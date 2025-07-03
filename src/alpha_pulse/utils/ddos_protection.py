"""
DDoS protection utilities for AlphaPulse API.

Implements comprehensive DDoS detection and mitigation:
- Traffic pattern analysis
- Anomaly detection
- Progressive response degradation
- Automatic IP blocking
"""

import time
import json
import asyncio
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import redis
import ipaddress
import geoip2.database
import user_agents

from alpha_pulse.config.rate_limits import DDOS_THRESHOLDS, IP_REPUTATION_WEIGHTS
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity
from alpha_pulse.models.rate_limit_tracking import SecurityEvent, IPReputationScore


@dataclass
class TrafficMetrics:
    """Traffic metrics for DDoS analysis."""
    requests_per_second: float
    unique_ips_per_minute: int
    error_rate_percentage: float
    new_connections_per_second: float
    avg_request_size: float
    suspicious_user_agents: int
    repeated_patterns: int


@dataclass
class ThreatIndicator:
    """Threat indicator with score and evidence."""
    indicator_type: str
    score: float  # 0-100
    evidence: Dict[str, any]
    confidence: float  # 0-1
    timestamp: datetime


class IPReputationManager:
    """Manages IP reputation scoring and classification."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize reputation manager."""
        self.redis = redis_client
        self.audit_logger = get_audit_logger()
        
        # Load GeoIP database if available
        try:
            self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-City.mmdb')
        except:
            self.geoip_reader = None
            
    async def get_ip_reputation(self, ip_address: str) -> Dict[str, any]:
        """Get comprehensive IP reputation information."""
        key = f"ip_reputation:{ip_address}"
        
        # Get cached reputation data
        cached_data = await self.redis.hgetall(key)
        
        if cached_data:
            return {
                'score': int(cached_data.get('score', 50)),
                'classification': cached_data.get('classification', 'unknown'),
                'country': cached_data.get('country'),
                'is_vpn': cached_data.get('is_vpn', 'false') == 'true',
                'is_proxy': cached_data.get('is_proxy', 'false') == 'true',
                'last_seen': cached_data.get('last_seen'),
                'threat_types': json.loads(cached_data.get('threat_types', '[]'))
            }
            
        # Calculate reputation for new IP
        reputation = await self._calculate_ip_reputation(ip_address)
        
        # Cache the result
        await self.redis.hmset(key, reputation)
        await self.redis.expire(key, 3600)  # 1 hour cache
        
        return reputation
        
    async def _calculate_ip_reputation(self, ip_address: str) -> Dict[str, any]:
        """Calculate IP reputation score."""
        score = 50  # Neutral starting score
        classification = "unknown"
        threat_types = []
        
        # Geographic analysis
        country, is_vpn, is_proxy = await self._analyze_geographic(ip_address)
        
        # Historical behavior analysis
        historical_score = await self._analyze_historical_behavior(ip_address)
        score = (score + historical_score) / 2
        
        # Check against threat intelligence feeds
        threat_intel_score = await self._check_threat_intelligence(ip_address)
        if threat_intel_score < 50:
            score = min(score, threat_intel_score)
            threat_types.append("threat_intel")
            
        # Network analysis
        if await self._is_suspicious_network(ip_address):
            score -= 20
            threat_types.append("suspicious_network")
            
        # Classify based on score
        if score >= 80:
            classification = "trusted"
        elif score >= 60:
            classification = "good"
        elif score >= 40:
            classification = "neutral"
        elif score >= 20:
            classification = "suspicious"
        else:
            classification = "malicious"
            
        return {
            'score': max(0, min(100, int(score))),
            'classification': classification,
            'country': country,
            'is_vpn': str(is_vpn),
            'is_proxy': str(is_proxy),
            'last_seen': datetime.now(timezone.utc).isoformat(),
            'threat_types': json.dumps(threat_types)
        }
        
    async def _analyze_geographic(self, ip_address: str) -> Tuple[Optional[str], bool, bool]:
        """Analyze geographic and network information."""
        if not self.geoip_reader:
            return None, False, False
            
        try:
            response = self.geoip_reader.city(ip_address)
            country = response.country.iso_code
            
            # Simple VPN/Proxy detection based on ASN
            asn = response.traits.autonomous_system_number
            org = response.traits.autonomous_system_organization or ""
            
            # Common VPN/Proxy indicators
            vpn_indicators = ['vpn', 'proxy', 'hosting', 'cloud', 'datacenter']
            is_vpn = any(indicator in org.lower() for indicator in vpn_indicators)
            is_proxy = 'proxy' in org.lower()
            
            return country, is_vpn, is_proxy
            
        except Exception:
            return None, False, False
            
    async def _analyze_historical_behavior(self, ip_address: str) -> float:
        """Analyze historical behavior of IP."""
        # Get recent security events for this IP
        events_key = f"security_events:{ip_address}"
        recent_events = await self.redis.lrange(events_key, 0, 100)
        
        if not recent_events:
            return 50  # Neutral for unknown IPs
            
        score = 50
        for event_json in recent_events:
            try:
                event = json.loads(event_json)
                event_type = event.get('type')
                
                # Apply scoring based on event type
                if event_type == 'rate_limit_violation':
                    score -= 5
                elif event_type == 'authentication_failure':
                    score -= 10
                elif event_type == 'malicious_pattern':
                    score -= 25
                elif event_type == 'successful_request':
                    score += 1
                    
            except json.JSONDecodeError:
                continue
                
        return max(0, min(100, score))
        
    async def _check_threat_intelligence(self, ip_address: str) -> float:
        """Check IP against threat intelligence feeds."""
        # This would integrate with external threat intelligence APIs
        # For now, we'll check against a simple blacklist
        
        blacklist_key = "ip_blacklist"
        is_blacklisted = await self.redis.sismember(blacklist_key, ip_address)
        
        if is_blacklisted:
            return 0  # Maximum threat score
            
        # Check against known malicious networks
        try:
            ip_obj = ipaddress.ip_address(ip_address)
            for malicious_network in await self._get_malicious_networks():
                if ip_obj in ipaddress.ip_network(malicious_network):
                    return 10  # High threat score
        except:
            pass
            
        return 50  # Neutral if no intelligence available
        
    async def _get_malicious_networks(self) -> List[str]:
        """Get list of known malicious networks."""
        # This would be populated from threat feeds
        return await self.redis.smembers("malicious_networks") or []
        
    async def _is_suspicious_network(self, ip_address: str) -> bool:
        """Check if IP belongs to suspicious network."""
        try:
            # Check if it's a Tor exit node
            if await self.redis.sismember("tor_exit_nodes", ip_address):
                return True
                
            # Check hosting providers known for abuse
            suspicious_asns = await self.redis.smembers("suspicious_asns")
            # Would need ASN lookup here
            
            return False
        except:
            return False
            
    async def update_reputation(self, ip_address: str, event_type: str, weight: int = 1):
        """Update IP reputation based on new event."""
        current_rep = await self.get_ip_reputation(ip_address)
        
        # Apply reputation change
        score_change = IP_REPUTATION_WEIGHTS.get(event_type, 0) * weight
        new_score = max(0, min(100, current_rep['score'] + score_change))
        
        # Update in Redis
        key = f"ip_reputation:{ip_address}"
        await self.redis.hset(key, 'score', new_score)
        
        # Log significant reputation changes
        if abs(score_change) >= 10:
            self.audit_logger.log(
                event_type=AuditEventType.DATA_MODIFIED,
                event_data={
                    'ip_address': ip_address,
                    'event_type': event_type,
                    'old_score': current_rep['score'],
                    'new_score': new_score,
                    'change': score_change
                },
                data_classification="restricted"
            )


class TrafficAnalyzer:
    """Analyzes traffic patterns for DDoS detection."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize traffic analyzer."""
        self.redis = redis_client
        self.audit_logger = get_audit_logger()
        
        # Traffic pattern buffers
        self.request_times = deque(maxlen=1000)
        self.ip_requests = defaultdict(list)
        self.user_agent_patterns = defaultdict(int)
        
    async def analyze_request(self, request_data: Dict[str, any]) -> List[ThreatIndicator]:
        """Analyze individual request for threat indicators."""
        indicators = []
        
        # Extract request information
        ip_address = request_data.get('ip_address')
        user_agent = request_data.get('user_agent', '')
        endpoint = request_data.get('endpoint', '')
        method = request_data.get('method', 'GET')
        
        # Analyze request frequency from IP
        frequency_indicator = await self._analyze_request_frequency(ip_address)
        if frequency_indicator:
            indicators.append(frequency_indicator)
            
        # Analyze user agent patterns
        ua_indicator = self._analyze_user_agent(user_agent)
        if ua_indicator:
            indicators.append(ua_indicator)
            
        # Analyze request patterns
        pattern_indicator = self._analyze_request_pattern(endpoint, method)
        if pattern_indicator:
            indicators.append(pattern_indicator)
            
        # Analyze payload characteristics
        payload_indicator = await self._analyze_payload(request_data)
        if payload_indicator:
            indicators.append(payload_indicator)
            
        return indicators
        
    async def _analyze_request_frequency(self, ip_address: str) -> Optional[ThreatIndicator]:
        """Analyze request frequency from single IP."""
        if not ip_address:
            return None
            
        # Track requests from this IP
        now = time.time()
        self.ip_requests[ip_address].append(now)
        
        # Remove old entries (older than 1 minute)
        cutoff = now - 60
        self.ip_requests[ip_address] = [
            t for t in self.ip_requests[ip_address] if t > cutoff
        ]
        
        requests_per_minute = len(self.ip_requests[ip_address])
        
        # Check against threshold
        if requests_per_minute > DDOS_THRESHOLDS["requests_per_second"] * 60:
            return ThreatIndicator(
                indicator_type="high_frequency",
                score=min(100, requests_per_minute / 10),
                evidence={
                    'requests_per_minute': requests_per_minute,
                    'threshold': DDOS_THRESHOLDS["requests_per_second"] * 60,
                    'ip_address': ip_address
                },
                confidence=0.8,
                timestamp=datetime.now(timezone.utc)
            )
            
        return None
        
    def _analyze_user_agent(self, user_agent: str) -> Optional[ThreatIndicator]:
        """Analyze user agent for bot patterns."""
        if not user_agent:
            return ThreatIndicator(
                indicator_type="missing_user_agent",
                score=30,
                evidence={'user_agent': user_agent},
                confidence=0.6,
                timestamp=datetime.now(timezone.utc)
            )
            
        # Parse user agent
        try:
            ua = user_agents.parse(user_agent)
            
            # Check for bot indicators
            bot_indicators = [
                'bot', 'crawler', 'spider', 'scraper', 'scanner',
                'wget', 'curl', 'python', 'requests', 'http'
            ]
            
            if any(indicator in user_agent.lower() for indicator in bot_indicators):
                return ThreatIndicator(
                    indicator_type="bot_user_agent",
                    score=60,
                    evidence={
                        'user_agent': user_agent,
                        'browser': ua.browser.family,
                        'os': ua.os.family
                    },
                    confidence=0.7,
                    timestamp=datetime.now(timezone.utc)
                )
                
            # Check for very old browsers (potential bot)
            if ua.browser.version and len(ua.browser.version) >= 2:
                major_version = ua.browser.version[0]
                if ua.browser.family == 'Chrome' and major_version < 70:
                    return ThreatIndicator(
                        indicator_type="outdated_browser",
                        score=40,
                        evidence={
                            'user_agent': user_agent,
                            'browser': f"{ua.browser.family} {major_version}"
                        },
                        confidence=0.5,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception:
            # Malformed user agent
            return ThreatIndicator(
                indicator_type="malformed_user_agent",
                score=50,
                evidence={'user_agent': user_agent},
                confidence=0.6,
                timestamp=datetime.now(timezone.utc)
            )
            
        return None
        
    def _analyze_request_pattern(self, endpoint: str, method: str) -> Optional[ThreatIndicator]:
        """Analyze request patterns for attacks."""
        # SQL injection patterns
        sql_patterns = [
            r"union\s+select", r"or\s+1\s*=\s*1", r"drop\s+table",
            r"--", r"/\*.*\*/", r"xp_cmdshell", r"sp_executesql"
        ]
        
        endpoint_lower = endpoint.lower()
        for pattern in sql_patterns:
            if re.search(pattern, endpoint_lower):
                return ThreatIndicator(
                    indicator_type="sql_injection_attempt",
                    score=90,
                    evidence={
                        'endpoint': endpoint,
                        'method': method,
                        'pattern': pattern
                    },
                    confidence=0.9,
                    timestamp=datetime.now(timezone.utc)
                )
                
        # Path traversal patterns
        if "../" in endpoint or "..%2f" in endpoint.lower():
            return ThreatIndicator(
                indicator_type="path_traversal_attempt",
                score=85,
                evidence={'endpoint': endpoint, 'method': method},
                confidence=0.8,
                timestamp=datetime.now(timezone.utc)
            )
            
        # Excessive path length
        if len(endpoint) > 500:
            return ThreatIndicator(
                indicator_type="excessive_path_length",
                score=60,
                evidence={
                    'endpoint_length': len(endpoint),
                    'method': method
                },
                confidence=0.7,
                timestamp=datetime.now(timezone.utc)
            )
            
        return None
        
    async def _analyze_payload(self, request_data: Dict[str, any]) -> Optional[ThreatIndicator]:
        """Analyze request payload for malicious content."""
        content_length = request_data.get('content_length', 0)
        
        # Excessive payload size
        if content_length > 10 * 1024 * 1024:  # 10MB
            return ThreatIndicator(
                indicator_type="excessive_payload_size",
                score=70,
                evidence={'content_length': content_length},
                confidence=0.6,
                timestamp=datetime.now(timezone.utc)
            )
            
        return None
        
    async def get_traffic_metrics(self) -> TrafficMetrics:
        """Get current traffic metrics."""
        now = time.time()
        
        # Calculate requests per second
        recent_requests = [t for t in self.request_times if now - t <= 1]
        rps = len(recent_requests)
        
        # Calculate unique IPs per minute
        unique_ips = set()
        for ip, times in self.ip_requests.items():
            if any(now - t <= 60 for t in times):
                unique_ips.add(ip)
        unique_ips_per_minute = len(unique_ips)
        
        # Get error rate from Redis
        error_count = int(await self.redis.get("metrics:errors:1min") or 0)
        total_count = int(await self.redis.get("metrics:requests:1min") or 1)
        error_rate = (error_count / total_count) * 100
        
        return TrafficMetrics(
            requests_per_second=rps,
            unique_ips_per_minute=unique_ips_per_minute,
            error_rate_percentage=error_rate,
            new_connections_per_second=0,  # Would need connection tracking
            avg_request_size=0,  # Would calculate from recent requests
            suspicious_user_agents=sum(1 for count in self.user_agent_patterns.values() if count > 100),
            repeated_patterns=0  # Would analyze for repeated attack patterns
        )


class DDoSMitigator:
    """Implements DDoS mitigation strategies."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize DDoS mitigator."""
        self.redis = redis_client
        self.audit_logger = get_audit_logger()
        self.ip_reputation = IPReputationManager(redis_client)
        self.traffic_analyzer = TrafficAnalyzer(redis_client)
        
    async def assess_threat_level(self, indicators: List[ThreatIndicator]) -> str:
        """Assess overall threat level from indicators."""
        if not indicators:
            return "none"
            
        max_score = max(indicator.score for indicator in indicators)
        avg_confidence = sum(indicator.confidence for indicator in indicators) / len(indicators)
        
        # Adjust score based on confidence
        adjusted_score = max_score * avg_confidence
        
        if adjusted_score >= 80:
            return "critical"
        elif adjusted_score >= 60:
            return "high"
        elif adjusted_score >= 40:
            return "medium"
        elif adjusted_score >= 20:
            return "low"
        else:
            return "minimal"
            
    async def apply_mitigation(
        self,
        ip_address: str,
        threat_level: str,
        indicators: List[ThreatIndicator]
    ) -> Dict[str, any]:
        """Apply appropriate mitigation strategy."""
        
        mitigation_actions = []
        
        if threat_level == "critical":
            # Immediate blocking
            await self._block_ip(ip_address, duration=3600)  # 1 hour
            mitigation_actions.append("ip_blocked")
            
        elif threat_level == "high":
            # Aggressive rate limiting
            await self._apply_aggressive_rate_limit(ip_address, duration=1800)  # 30 minutes
            mitigation_actions.append("aggressive_rate_limit")
            
        elif threat_level == "medium":
            # Progressive delay
            await self._apply_progressive_delay(ip_address, duration=900)  # 15 minutes
            mitigation_actions.append("progressive_delay")
            
        elif threat_level == "low":
            # Increase monitoring
            await self._increase_monitoring(ip_address)
            mitigation_actions.append("increased_monitoring")
            
        # Update IP reputation
        reputation_impact = {
            "critical": -50,
            "high": -25,
            "medium": -10,
            "low": -5
        }
        
        if threat_level in reputation_impact:
            await self.ip_reputation.update_reputation(
                ip_address, "malicious_patterns", reputation_impact[threat_level]
            )
            
        # Log security event
        await self._log_security_event(ip_address, threat_level, indicators, mitigation_actions)
        
        return {
            "threat_level": threat_level,
            "mitigation_actions": mitigation_actions,
            "indicators": [
                {
                    "type": ind.indicator_type,
                    "score": ind.score,
                    "confidence": ind.confidence
                } for ind in indicators
            ]
        }
        
    async def _block_ip(self, ip_address: str, duration: int):
        """Block IP address for specified duration."""
        block_key = f"blocked_ips:{ip_address}"
        await self.redis.setex(block_key, duration, "blocked")
        
        # Add to global blocklist
        await self.redis.sadd("global_blocklist", ip_address)
        
        self.audit_logger.log(
            event_type=AuditEventType.DATA_MODIFIED,
            event_data={
                'action': 'ip_blocked',
                'ip_address': ip_address,
                'duration': duration
            },
            severity=AuditSeverity.WARNING
        )
        
    async def _apply_aggressive_rate_limit(self, ip_address: str, duration: int):
        """Apply aggressive rate limiting to IP."""
        limit_key = f"aggressive_limit:{ip_address}"
        await self.redis.setex(limit_key, duration, "10")  # 10 requests per minute
        
    async def _apply_progressive_delay(self, ip_address: str, duration: int):
        """Apply progressive delays to IP requests."""
        delay_key = f"progressive_delay:{ip_address}"
        await self.redis.setex(delay_key, duration, "1000")  # 1 second delay
        
    async def _increase_monitoring(self, ip_address: str):
        """Increase monitoring for IP address."""
        monitor_key = f"monitor:{ip_address}"
        await self.redis.setex(monitor_key, 3600, "enhanced")  # 1 hour enhanced monitoring
        
    async def _log_security_event(
        self,
        ip_address: str,
        threat_level: str,
        indicators: List[ThreatIndicator],
        actions: List[str]
    ):
        """Log security event for analysis."""
        event_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ip_address': ip_address,
            'threat_level': threat_level,
            'indicators': [
                {
                    'type': ind.indicator_type,
                    'score': ind.score,
                    'confidence': ind.confidence,
                    'evidence': ind.evidence
                } for ind in indicators
            ],
            'mitigation_actions': actions
        }
        
        # Store in Redis for real-time analysis
        await self.redis.lpush(
            f"security_events:{ip_address}",
            json.dumps(event_data)
        )
        
        # Keep only last 100 events per IP
        await self.redis.ltrim(f"security_events:{ip_address}", 0, 99)
        
        # Store in global security events
        await self.redis.lpush("global_security_events", json.dumps(event_data))
        await self.redis.ltrim("global_security_events", 0, 9999)
        
        # Audit log critical events
        if threat_level in ["critical", "high"]:
            self.audit_logger.log(
                event_type=AuditEventType.DATA_ACCESS,
                event_data={
                    'security_event': True,
                    'ip_address': ip_address,
                    'threat_level': threat_level,
                    'action_count': len(actions)
                },
                severity=AuditSeverity.CRITICAL if threat_level == "critical" else AuditSeverity.ERROR,
                data_classification="restricted"
            )
            
    async def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is currently blocked."""
        block_key = f"blocked_ips:{ip_address}"
        is_blocked = await self.redis.exists(block_key)
        
        # Also check global blocklist
        in_global_blocklist = await self.redis.sismember("global_blocklist", ip_address)
        
        return bool(is_blocked or in_global_blocklist)
        
    async def get_mitigation_status(self, ip_address: str) -> Dict[str, any]:
        """Get current mitigation status for IP."""
        status = {
            'blocked': await self.is_ip_blocked(ip_address),
            'aggressive_limit': await self.redis.exists(f"aggressive_limit:{ip_address}"),
            'progressive_delay': await self.redis.exists(f"progressive_delay:{ip_address}"),
            'enhanced_monitoring': await self.redis.exists(f"monitor:{ip_address}"),
            'reputation': await self.ip_reputation.get_ip_reputation(ip_address)
        }
        
        return status