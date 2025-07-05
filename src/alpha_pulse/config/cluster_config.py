"""Configuration for distributed computing clusters."""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

class ClusterType(Enum):
    """Types of distributed computing clusters."""
    LOCAL = "local"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    HYBRID = "hybrid"


class ScalingPolicy(Enum):
    """Cluster scaling policies."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class ClusterNodeConfig:
    """Configuration for cluster nodes."""
    cpu_cores: int = 4
    memory_gb: int = 8
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    disk_size_gb: int = 100
    instance_type: Optional[str] = None
    spot_instance: bool = True
    preemptible: bool = True
    node_labels: Dict[str, str] = field(default_factory=dict)
    taints: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration."""
    enabled: bool = True
    min_nodes: int = 0
    max_nodes: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_rate: int = 2  # nodes per minute
    scale_down_rate: int = 1  # nodes per minute
    scale_down_delay: int = 300  # seconds
    scaling_policy: ScalingPolicy = ScalingPolicy.DYNAMIC


@dataclass
class NetworkConfig:
    """Network configuration for cluster."""
    enable_tls: bool = True
    enable_encryption: bool = True
    allowed_ips: List[str] = field(default_factory=list)
    proxy_settings: Optional[Dict[str, str]] = None
    bandwidth_limit_mbps: Optional[int] = None
    latency_threshold_ms: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring configuration for cluster."""
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    prometheus_endpoint: Optional[str] = None
    grafana_endpoint: Optional[str] = None
    alert_endpoints: List[str] = field(default_factory=list)


@dataclass
class StorageConfig:
    """Storage configuration for cluster."""
    shared_storage_type: str = "nfs"  # nfs, s3, gcs, azure_blob
    shared_storage_path: str = "/mnt/shared"
    cache_size_gb: int = 100
    persistent_volume_size_gb: int = 500
    enable_data_compression: bool = True
    enable_data_encryption: bool = True
    s3_bucket: Optional[str] = None
    gcs_bucket: Optional[str] = None
    azure_container: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration for cluster."""
    enable_rbac: bool = True
    enable_pod_security: bool = True
    enable_network_policies: bool = True
    service_account: Optional[str] = None
    secrets_manager: Optional[str] = None  # vault, aws_secrets, gcp_secrets
    certificate_authority: Optional[str] = None
    allowed_registries: List[str] = field(default_factory=list)


@dataclass
class ClusterConfig:
    """Main cluster configuration."""
    cluster_name: str = "alphapulse-cluster"
    cluster_type: ClusterType = ClusterType.LOCAL
    region: Optional[str] = None
    availability_zones: List[str] = field(default_factory=list)
    
    # Node configurations
    head_node: ClusterNodeConfig = field(default_factory=ClusterNodeConfig)
    worker_node: ClusterNodeConfig = field(default_factory=ClusterNodeConfig)
    
    # Scaling configuration
    autoscaling: AutoScalingConfig = field(default_factory=AutoScalingConfig)
    
    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Security configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Framework-specific settings
    ray_config: Dict[str, Any] = field(default_factory=dict)
    dask_config: Dict[str, Any] = field(default_factory=dict)
    
    # Cost optimization
    enable_spot_instances: bool = True
    spot_instance_max_price: Optional[float] = None
    idle_termination_minutes: int = 30
    
    # Advanced settings
    enable_gpu_support: bool = False
    enable_rdma: bool = False
    enable_nvlink: bool = False
    custom_docker_image: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "cluster_name": self.cluster_name,
            "cluster_type": self.cluster_type.value,
            "region": self.region,
            "availability_zones": self.availability_zones,
            "head_node": {
                "cpu_cores": self.head_node.cpu_cores,
                "memory_gb": self.head_node.memory_gb,
                "gpu_count": self.head_node.gpu_count,
                "gpu_type": self.head_node.gpu_type,
                "disk_size_gb": self.head_node.disk_size_gb,
                "instance_type": self.head_node.instance_type,
                "spot_instance": self.head_node.spot_instance,
            },
            "worker_node": {
                "cpu_cores": self.worker_node.cpu_cores,
                "memory_gb": self.worker_node.memory_gb,
                "gpu_count": self.worker_node.gpu_count,
                "gpu_type": self.worker_node.gpu_type,
                "disk_size_gb": self.worker_node.disk_size_gb,
                "instance_type": self.worker_node.instance_type,
                "spot_instance": self.worker_node.spot_instance,
            },
            "autoscaling": {
                "enabled": self.autoscaling.enabled,
                "min_nodes": self.autoscaling.min_nodes,
                "max_nodes": self.autoscaling.max_nodes,
                "target_cpu_utilization": self.autoscaling.target_cpu_utilization,
                "target_memory_utilization": self.autoscaling.target_memory_utilization,
                "scaling_policy": self.autoscaling.scaling_policy.value,
            },
            "network": {
                "enable_tls": self.network.enable_tls,
                "enable_encryption": self.network.enable_encryption,
                "allowed_ips": self.network.allowed_ips,
                "bandwidth_limit_mbps": self.network.bandwidth_limit_mbps,
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "enable_logging": self.monitoring.enable_logging,
                "enable_tracing": self.monitoring.enable_tracing,
                "metrics_port": self.monitoring.metrics_port,
                "log_level": self.monitoring.log_level,
            },
            "storage": {
                "shared_storage_type": self.storage.shared_storage_type,
                "shared_storage_path": self.storage.shared_storage_path,
                "cache_size_gb": self.storage.cache_size_gb,
                "enable_data_compression": self.storage.enable_data_compression,
            },
            "security": {
                "enable_rbac": self.security.enable_rbac,
                "enable_pod_security": self.security.enable_pod_security,
                "enable_network_policies": self.security.enable_network_policies,
            },
            "ray_config": self.ray_config,
            "dask_config": self.dask_config,
            "enable_spot_instances": self.enable_spot_instances,
            "idle_termination_minutes": self.idle_termination_minutes,
            "enable_gpu_support": self.enable_gpu_support,
            "custom_docker_image": self.custom_docker_image,
            "environment_variables": self.environment_variables,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Basic settings
        config.cluster_name = data.get("cluster_name", config.cluster_name)
        config.cluster_type = ClusterType(data.get("cluster_type", "local"))
        config.region = data.get("region")
        config.availability_zones = data.get("availability_zones", [])
        
        # Node configurations
        if "head_node" in data:
            head_data = data["head_node"]
            config.head_node = ClusterNodeConfig(
                cpu_cores=head_data.get("cpu_cores", 4),
                memory_gb=head_data.get("memory_gb", 8),
                gpu_count=head_data.get("gpu_count", 0),
                gpu_type=head_data.get("gpu_type"),
                disk_size_gb=head_data.get("disk_size_gb", 100),
                instance_type=head_data.get("instance_type"),
                spot_instance=head_data.get("spot_instance", True),
            )
            
        if "worker_node" in data:
            worker_data = data["worker_node"]
            config.worker_node = ClusterNodeConfig(
                cpu_cores=worker_data.get("cpu_cores", 4),
                memory_gb=worker_data.get("memory_gb", 8),
                gpu_count=worker_data.get("gpu_count", 0),
                gpu_type=worker_data.get("gpu_type"),
                disk_size_gb=worker_data.get("disk_size_gb", 100),
                instance_type=worker_data.get("instance_type"),
                spot_instance=worker_data.get("spot_instance", True),
            )
            
        # Autoscaling
        if "autoscaling" in data:
            auto_data = data["autoscaling"]
            config.autoscaling = AutoScalingConfig(
                enabled=auto_data.get("enabled", True),
                min_nodes=auto_data.get("min_nodes", 0),
                max_nodes=auto_data.get("max_nodes", 10),
                target_cpu_utilization=auto_data.get("target_cpu_utilization", 70.0),
                target_memory_utilization=auto_data.get("target_memory_utilization", 80.0),
                scaling_policy=ScalingPolicy(auto_data.get("scaling_policy", "dynamic")),
            )
            
        # Other configurations
        config.ray_config = data.get("ray_config", {})
        config.dask_config = data.get("dask_config", {})
        config.enable_spot_instances = data.get("enable_spot_instances", True)
        config.idle_termination_minutes = data.get("idle_termination_minutes", 30)
        config.enable_gpu_support = data.get("enable_gpu_support", False)
        config.custom_docker_image = data.get("custom_docker_image")
        config.environment_variables = data.get("environment_variables", {})
        
        return config
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".yaml" or ext == ".yml":
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif ext == ".json":
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
    @classmethod
    def load(cls, filepath: str) -> "ClusterConfig":
        """Load configuration from file."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".yaml" or ext == ".yml":
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif ext == ".json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        return cls.from_dict(data)


# Predefined configurations
def get_local_development_config() -> ClusterConfig:
    """Get configuration for local development."""
    config = ClusterConfig(
        cluster_name="alphapulse-local",
        cluster_type=ClusterType.LOCAL,
        head_node=ClusterNodeConfig(cpu_cores=2, memory_gb=4),
        worker_node=ClusterNodeConfig(cpu_cores=2, memory_gb=4),
    )
    config.autoscaling.enabled = False
    config.autoscaling.min_nodes = 4
    config.autoscaling.max_nodes = 4
    return config


def get_aws_production_config() -> ClusterConfig:
    """Get configuration for AWS production deployment."""
    config = ClusterConfig(
        cluster_name="alphapulse-prod",
        cluster_type=ClusterType.AWS,
        region="us-east-1",
        availability_zones=["us-east-1a", "us-east-1b", "us-east-1c"],
        head_node=ClusterNodeConfig(
            cpu_cores=8,
            memory_gb=32,
            disk_size_gb=200,
            instance_type="m5.2xlarge",
            spot_instance=False,
        ),
        worker_node=ClusterNodeConfig(
            cpu_cores=16,
            memory_gb=64,
            disk_size_gb=500,
            instance_type="m5.4xlarge",
            spot_instance=True,
        ),
    )
    
    config.autoscaling.enabled = True
    config.autoscaling.min_nodes = 2
    config.autoscaling.max_nodes = 50
    config.autoscaling.scaling_policy = ScalingPolicy.COST_OPTIMIZED
    
    config.storage.shared_storage_type = "s3"
    config.storage.s3_bucket = "alphapulse-distributed-storage"
    
    config.enable_spot_instances = True
    config.spot_instance_max_price = 0.5  # 50% of on-demand price
    
    return config


def get_gpu_cluster_config() -> ClusterConfig:
    """Get configuration for GPU-accelerated cluster."""
    config = ClusterConfig(
        cluster_name="alphapulse-gpu",
        cluster_type=ClusterType.AWS,
        region="us-west-2",
        head_node=ClusterNodeConfig(
            cpu_cores=8,
            memory_gb=32,
            disk_size_gb=200,
            instance_type="m5.2xlarge",
        ),
        worker_node=ClusterNodeConfig(
            cpu_cores=8,
            memory_gb=61,
            gpu_count=1,
            gpu_type="V100",
            disk_size_gb=500,
            instance_type="p3.2xlarge",
            spot_instance=True,
        ),
    )
    
    config.enable_gpu_support = True
    config.enable_nvlink = True
    
    config.ray_config["num_gpus"] = 1
    config.custom_docker_image = "alphapulse/gpu-worker:latest"
    
    return config


def validate_cluster_config(config: ClusterConfig) -> List[str]:
    """Validate cluster configuration.
    
    Args:
        config: Cluster configuration
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Validate node resources
    if config.head_node.cpu_cores < 2:
        errors.append("Head node must have at least 2 CPU cores")
        
    if config.head_node.memory_gb < 4:
        errors.append("Head node must have at least 4GB memory")
        
    if config.worker_node.cpu_cores < 1:
        errors.append("Worker nodes must have at least 1 CPU core")
        
    if config.worker_node.memory_gb < 2:
        errors.append("Worker nodes must have at least 2GB memory")
        
    # Validate autoscaling
    if config.autoscaling.enabled:
        if config.autoscaling.min_nodes > config.autoscaling.max_nodes:
            errors.append("Minimum nodes cannot exceed maximum nodes")
            
        if config.autoscaling.target_cpu_utilization <= 0 or config.autoscaling.target_cpu_utilization > 100:
            errors.append("CPU utilization target must be between 0 and 100")
            
    # Validate GPU configuration
    if config.enable_gpu_support:
        if config.worker_node.gpu_count == 0:
            errors.append("GPU support enabled but no GPUs configured for workers")
            
    # Validate storage
    if config.storage.cache_size_gb > config.worker_node.disk_size_gb:
        errors.append("Cache size exceeds worker disk size")
        
    return errors