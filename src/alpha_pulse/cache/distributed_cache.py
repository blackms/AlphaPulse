"""Distributed caching implementation for multi-node deployments."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis_async
from redis.asyncio.cluster import RedisCluster
from redis.exceptions import RedisError

from .redis_manager import RedisManager, CacheTier
from ..utils.logging_utils import get_logger
from ..monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class NodeRole(Enum):
    """Node roles in distributed cache."""
    
    PRIMARY = "primary"
    REPLICA = "replica"
    CACHE_ONLY = "cache_only"


class ShardingStrategy(Enum):
    """Sharding strategies for distributed cache."""
    
    CONSISTENT_HASH = "consistent_hash"
    RANGE_BASED = "range_based"
    TAG_BASED = "tag_based"


@dataclass
class CacheNode:
    """Represents a cache node in the distributed system."""
    
    id: str
    host: str
    port: int
    role: NodeRole
    weight: int = 1
    tags: Set[str] = None
    last_heartbeat: datetime = None
    is_healthy: bool = True


class ConsistentHashRing:
    """Consistent hashing implementation for cache distribution."""
    
    def __init__(self, virtual_nodes: int = 150):
        """Initialize consistent hash ring."""
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, CacheNode] = {}
        self.nodes: Dict[str, CacheNode] = {}
    
    def add_node(self, node: CacheNode) -> None:
        """Add node to the ring."""
        self.nodes[node.id] = node
        
        # Add virtual nodes
        for i in range(self.virtual_nodes * node.weight):
            virtual_key = f"{node.id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from the ring."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Remove virtual nodes
        keys_to_remove = []
        for hash_value, ring_node in self.ring.items():
            if ring_node.id == node_id:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        del self.nodes[node_id]
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        """Get node for a given key."""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node with hash >= key hash
        sorted_hashes = sorted(self.ring.keys())
        for node_hash in sorted_hashes:
            if node_hash >= hash_value:
                return self.ring[node_hash]
        
        # Wrap around to first node
        return self.ring[sorted_hashes[0]]
    
    def get_nodes(self, key: str, count: int = 3) -> List[CacheNode]:
        """Get multiple nodes for replication."""
        if not self.ring:
            return []
        
        nodes = []
        seen_ids = set()
        
        hash_value = self._hash(key)
        sorted_hashes = sorted(self.ring.keys())
        
        # Start from the primary node
        start_index = 0
        for i, node_hash in enumerate(sorted_hashes):
            if node_hash >= hash_value:
                start_index = i
                break
        
        # Collect unique nodes
        for i in range(len(sorted_hashes)):
            index = (start_index + i) % len(sorted_hashes)
            node = self.ring[sorted_hashes[index]]
            
            if node.id not in seen_ids:
                nodes.append(node)
                seen_ids.add(node.id)
                
                if len(nodes) >= count:
                    break
        
        return nodes
    
    def _hash(self, key: str) -> int:
        """Generate hash for key."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class DistributedCacheManager:
    """Manages distributed caching across multiple nodes."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        node_id: str,
        sharding_strategy: ShardingStrategy = ShardingStrategy.CONSISTENT_HASH,
        replication_factor: int = 3,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize distributed cache manager."""
        self.redis_manager = redis_manager
        self.node_id = node_id
        self.sharding_strategy = sharding_strategy
        self.replication_factor = replication_factor
        self.metrics = metrics_collector
        
        # Consistent hashing
        self.hash_ring = ConsistentHashRing()
        
        # Node management
        self.nodes: Dict[str, CacheNode] = {}
        self.local_node: Optional[CacheNode] = None
        
        # Connection pools for remote nodes
        self._remote_connections: Dict[str, redis_async.Redis] = {}
        
        # Synchronization
        self._sync_lock = asyncio.Lock()
        self._pending_syncs: Dict[str, Set[str]] = {}  # key -> set of node_ids
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
    
    async def initialize(self, local_node: CacheNode, remote_nodes: List[CacheNode]) -> None:
        """Initialize distributed cache."""
        try:
            # Set local node
            self.local_node = local_node
            self.nodes[local_node.id] = local_node
            self.hash_ring.add_node(local_node)
            
            # Add remote nodes
            for node in remote_nodes:
                await self.add_node(node)
            
            # Start background tasks
            await self.start_background_tasks()
            
            logger.info(f"Distributed cache initialized with {len(self.nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed cache: {e}")
            raise
    
    async def add_node(self, node: CacheNode) -> None:
        """Add a node to the distributed cache."""
        try:
            # Add to tracking
            self.nodes[node.id] = node
            self.hash_ring.add_node(node)
            
            # Create connection to remote node
            if node.id != self.node_id:
                conn = redis_async.Redis(
                    host=node.host,
                    port=node.port,
                    decode_responses=False,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self._remote_connections[node.id] = conn
                
                # Test connection
                await conn.ping()
            
            logger.info(f"Added node {node.id} to distributed cache")
            
            # Trigger rebalancing
            await self._rebalance_cache(node)
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            raise
    
    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the distributed cache."""
        try:
            if node_id not in self.nodes:
                return
            
            node = self.nodes[node_id]
            
            # Remove from hash ring
            self.hash_ring.remove_node(node_id)
            
            # Close connection
            if node_id in self._remote_connections:
                await self._remote_connections[node_id].close()
                del self._remote_connections[node_id]
            
            # Remove from tracking
            del self.nodes[node_id]
            
            logger.info(f"Removed node {node_id} from distributed cache")
            
            # Trigger rebalancing
            await self._rebalance_cache(node, removing=True)
            
        except Exception as e:
            logger.error(f"Failed to remove node {node_id}: {e}")
    
    async def get(self, key: str, tier: CacheTier = CacheTier.L3_DISTRIBUTED_REDIS) -> Optional[Any]:
        """Get value from distributed cache."""
        try:
            # Determine target nodes
            nodes = self._get_target_nodes(key)
            
            # Try local node first if it's a target
            if self.local_node in nodes:
                value = await self.redis_manager.get(key, tier)
                if value is not None:
                    if self.metrics:
                        self.metrics.increment("distributed_cache.hit", {"source": "local"})
                    return value
            
            # Try remote nodes
            for node in nodes:
                if node.id == self.node_id:
                    continue
                
                value = await self._get_from_remote(node, key)
                if value is not None:
                    if self.metrics:
                        self.metrics.increment("distributed_cache.hit", {"source": "remote"})
                    
                    # Cache locally for future access
                    await self.redis_manager.set(key, value, tier=CacheTier.L2_LOCAL_REDIS)
                    
                    return value
            
            if self.metrics:
                self.metrics.increment("distributed_cache.miss")
            
            return None
            
        except Exception as e:
            logger.error(f"Distributed get error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("distributed_cache.error", {"operation": "get"})
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.L3_DISTRIBUTED_REDIS
    ) -> bool:
        """Set value in distributed cache."""
        try:
            # Determine target nodes
            nodes = self._get_target_nodes(key)
            
            # Set on all target nodes
            tasks = []
            for node in nodes:
                if node.id == self.node_id:
                    # Set locally
                    task = self.redis_manager.set(key, value, ttl, tier)
                else:
                    # Set remotely
                    task = self._set_on_remote(node, key, value, ttl)
                
                tasks.append(task)
            
            # Wait for all sets to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            success_count = sum(1 for r in results if r is True)
            
            if self.metrics:
                self.metrics.increment(
                    "distributed_cache.set",
                    {"successes": success_count, "total": len(nodes)}
                )
            
            # Consider successful if written to majority
            return success_count > len(nodes) // 2
            
        except Exception as e:
            logger.error(f"Distributed set error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("distributed_cache.error", {"operation": "set"})
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from distributed cache."""
        try:
            # Determine target nodes
            nodes = self._get_target_nodes(key)
            
            # Delete from all target nodes
            tasks = []
            for node in nodes:
                if node.id == self.node_id:
                    # Delete locally
                    task = self.redis_manager.delete(key)
                else:
                    # Delete remotely
                    task = self._delete_from_remote(node, key)
                
                tasks.append(task)
            
            # Wait for all deletes to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            success_count = sum(1 for r in results if r is True)
            
            if self.metrics:
                self.metrics.increment(
                    "distributed_cache.delete",
                    {"successes": success_count, "total": len(nodes)}
                )
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Distributed delete error for key {key}: {e}")
            if self.metrics:
                self.metrics.increment("distributed_cache.error", {"operation": "delete"})
            return False
    
    def _get_target_nodes(self, key: str) -> List[CacheNode]:
        """Get target nodes for a key based on sharding strategy."""
        if self.sharding_strategy == ShardingStrategy.CONSISTENT_HASH:
            return self.hash_ring.get_nodes(key, self.replication_factor)
        
        elif self.sharding_strategy == ShardingStrategy.RANGE_BASED:
            # Simple range-based sharding
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            node_index = hash_value % len(self.nodes)
            
            nodes = list(self.nodes.values())
            targets = []
            
            for i in range(self.replication_factor):
                index = (node_index + i) % len(nodes)
                targets.append(nodes[index])
            
            return targets
        
        elif self.sharding_strategy == ShardingStrategy.TAG_BASED:
            # Extract tag from key (e.g., "tag:key")
            parts = key.split(":", 1)
            if len(parts) == 2:
                tag = parts[0]
                # Find nodes with matching tag
                matching_nodes = [
                    node for node in self.nodes.values()
                    if node.tags and tag in node.tags
                ]
                if matching_nodes:
                    return matching_nodes[:self.replication_factor]
            
            # Fallback to consistent hashing
            return self.hash_ring.get_nodes(key, self.replication_factor)
        
        return []
    
    async def _get_from_remote(self, node: CacheNode, key: str) -> Optional[Any]:
        """Get value from remote node."""
        try:
            if node.id not in self._remote_connections:
                return None
            
            conn = self._remote_connections[node.id]
            data = await conn.get(key)
            
            if data:
                # Deserialize using redis manager's deserializer
                return self.redis_manager._deserialize(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Remote get error from node {node.id}: {e}")
            return None
    
    async def _set_on_remote(self, node: CacheNode, key: str, value: Any, ttl: Optional[int]) -> bool:
        """Set value on remote node."""
        try:
            if node.id not in self._remote_connections:
                return False
            
            conn = self._remote_connections[node.id]
            
            # Serialize using redis manager's serializer
            data = self.redis_manager._serialize(value)
            
            await conn.set(key, data, ex=ttl)
            return True
            
        except Exception as e:
            logger.error(f"Remote set error on node {node.id}: {e}")
            return False
    
    async def _delete_from_remote(self, node: CacheNode, key: str) -> bool:
        """Delete value from remote node."""
        try:
            if node.id not in self._remote_connections:
                return False
            
            conn = self._remote_connections[node.id]
            result = await conn.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Remote delete error on node {node.id}: {e}")
            return False
    
    async def _rebalance_cache(self, node: CacheNode, removing: bool = False) -> None:
        """Rebalance cache after node addition/removal."""
        try:
            logger.info(f"Starting cache rebalancing for node {node.id} (removing={removing})")
            
            # This is a simplified rebalancing strategy
            # In production, you'd want more sophisticated migration
            
            if removing:
                # Node being removed - migrate its data to other nodes
                # This would require tracking which keys are on which nodes
                pass
            else:
                # Node being added - migrate some data to it
                # This would require rehashing existing keys
                pass
            
            logger.info("Cache rebalancing completed")
            
        except Exception as e:
            logger.error(f"Rebalancing error: {e}")
    
    async def _monitor_nodes(self) -> None:
        """Monitor node health and connectivity."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for node_id, node in list(self.nodes.items()):
                    if node_id == self.node_id:
                        continue
                    
                    # Check node health
                    is_healthy = await self._check_node_health(node)
                    
                    if is_healthy != node.is_healthy:
                        node.is_healthy = is_healthy
                        
                        if not is_healthy:
                            logger.warning(f"Node {node_id} is unhealthy")
                            if self.metrics:
                                self.metrics.increment("distributed_cache.node_failure")
                        else:
                            logger.info(f"Node {node_id} recovered")
                            if self.metrics:
                                self.metrics.increment("distributed_cache.node_recovery")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Node monitoring error: {e}")
    
    async def _check_node_health(self, node: CacheNode) -> bool:
        """Check if a node is healthy."""
        try:
            if node.id not in self._remote_connections:
                return False
            
            conn = self._remote_connections[node.id]
            await conn.ping()
            
            node.last_heartbeat = datetime.utcnow()
            return True
            
        except Exception:
            return False
    
    async def start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start node monitoring
        monitor_task = asyncio.create_task(self._monitor_nodes())
        self._background_tasks.append(monitor_task)
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
        
        # Close remote connections
        for conn in self._remote_connections.values():
            await conn.close()
        
        self._remote_connections.clear()
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get distributed cache cluster statistics."""
        stats = {
            "cluster_size": len(self.nodes),
            "healthy_nodes": sum(1 for n in self.nodes.values() if n.is_healthy),
            "unhealthy_nodes": sum(1 for n in self.nodes.values() if not n.is_healthy),
            "replication_factor": self.replication_factor,
            "sharding_strategy": self.sharding_strategy.value,
            "nodes": {}
        }
        
        for node_id, node in self.nodes.items():
            stats["nodes"][node_id] = {
                "host": node.host,
                "port": node.port,
                "role": node.role.value,
                "is_healthy": node.is_healthy,
                "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None
            }
        
        return stats