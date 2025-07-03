"""
IP filtering and reputation management for AlphaPulse API.

Provides comprehensive IP filtering capabilities:
- Whitelist/blacklist management
- Geographic filtering
- VPN/Proxy detection
- Dynamic reputation scoring
- Threat intelligence integration
"""

import ipaddress
import time
import json
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import redis
import geoip2.database
import requests

from alpha_pulse.config.rate_limits import GEO_RESTRICTIONS
from alpha_pulse.utils.audit_logger import get_audit_logger, AuditEventType, AuditSeverity


@dataclass
class IPInfo:
    """Comprehensive IP information."""
    ip_address: str
    country_code: Optional[str]
    country_name: Optional[str]
    city: Optional[str]
    region: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    isp: Optional[str]
    asn: Optional[int]
    is_vpn: bool
    is_proxy: bool
    is_tor: bool
    is_hosting: bool
    threat_score: float  # 0-100
    reputation_score: float  # 0-100


@dataclass
class FilterRule:
    """IP filtering rule."""
    rule_id: str
    rule_type: str  # whitelist, blacklist, geo_block, asn_block
    pattern: str    # IP, CIDR, country code, ASN
    reason: str
    created_at: datetime
    expires_at: Optional[datetime]
    created_by: str
    is_active: bool


class IPGeolocationService:
    """IP geolocation and enrichment service."""
    
    def __init__(self, geoip_db_path: Optional[str] = None):
        """Initialize geolocation service."""
        self.geoip_db_path = geoip_db_path or "/usr/share/GeoIP/GeoLite2-City.mmdb"
        self.geoip_reader = None
        self.audit_logger = get_audit_logger()
        
        try:
            self.geoip_reader = geoip2.database.Reader(self.geoip_db_path)
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.SYSTEM_START,
                event_data={
                    'warning': 'GeoIP database not available',
                    'error': str(e),
                    'path': self.geoip_db_path
                }
            )
            
    async def get_ip_info(self, ip_address: str) -> IPInfo:
        """Get comprehensive IP information."""
        # Basic IP info structure
        ip_info = IPInfo(
            ip_address=ip_address,
            country_code=None,
            country_name=None,
            city=None,
            region=None,
            latitude=None,
            longitude=None,
            timezone=None,
            isp=None,
            asn=None,
            is_vpn=False,
            is_proxy=False,
            is_tor=False,
            is_hosting=False,
            threat_score=0.0,
            reputation_score=50.0
        )
        
        # Enhance with GeoIP data
        if self.geoip_reader:
            try:
                response = self.geoip_reader.city(ip_address)
                
                ip_info.country_code = response.country.iso_code
                ip_info.country_name = response.country.name
                ip_info.city = response.city.name
                ip_info.region = response.subdivisions.most_specific.name
                ip_info.latitude = float(response.location.latitude) if response.location.latitude else None
                ip_info.longitude = float(response.location.longitude) if response.location.longitude else None
                ip_info.timezone = response.location.time_zone
                ip_info.isp = response.traits.isp
                ip_info.asn = response.traits.autonomous_system_number
                
                # Detect VPN/Proxy/Hosting based on ISP
                org = (response.traits.autonomous_system_organization or "").lower()
                ip_info.is_vpn = any(term in org for term in ['vpn', 'virtual private'])
                ip_info.is_proxy = any(term in org for term in ['proxy', 'anonymous'])
                ip_info.is_hosting = any(term in org for term in ['hosting', 'cloud', 'datacenter', 'server'])
                
            except Exception as e:
                # Log geolocation failure but continue
                pass
                
        # Enhance with threat intelligence
        await self._enhance_with_threat_intel(ip_info)
        
        return ip_info
        
    async def _enhance_with_threat_intel(self, ip_info: IPInfo):
        """Enhance IP info with threat intelligence."""
        # Check if IP is known Tor exit node
        ip_info.is_tor = await self._is_tor_exit_node(ip_info.ip_address)
        
        # Calculate threat score based on various factors
        threat_score = 0.0
        
        if ip_info.is_tor:
            threat_score += 30
        if ip_info.is_vpn:
            threat_score += 15
        if ip_info.is_proxy:
            threat_score += 20
        if ip_info.is_hosting:
            threat_score += 10
            
        # Geographic risk factors
        high_risk_countries = ['CN', 'RU', 'KP', 'IR']
        if ip_info.country_code in high_risk_countries:
            threat_score += 25
            
        ip_info.threat_score = min(100.0, threat_score)
        
    async def _is_tor_exit_node(self, ip_address: str) -> bool:
        """Check if IP is a known Tor exit node."""
        # This would typically query a Tor exit node list
        # For demo purposes, we'll check a simple list
        tor_exit_nodes = [
            # Example Tor exit nodes (would be populated from real data)
            "199.87.154.255",
            "185.220.101.182"
        ]
        
        return ip_address in tor_exit_nodes


class IPFilterManager:
    """Manages IP filtering rules and decisions."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize IP filter manager."""
        self.redis = redis_client
        self.geolocation = IPGeolocationService()
        self.audit_logger = get_audit_logger()
        
    async def should_allow_ip(self, ip_address: str) -> Tuple[bool, str, Dict[str, any]]:
        """
        Determine if IP should be allowed access.
        
        Returns:
            Tuple of (allowed, reason, details)
        """
        # Get IP information
        ip_info = await self.geolocation.get_ip_info(ip_address)
        
        # Check whitelist first (highest priority)
        if await self._is_whitelisted(ip_address):
            return True, "whitelisted", {"ip_info": ip_info}
            
        # Check blacklist
        is_blacklisted, blacklist_reason = await self._is_blacklisted(ip_address)
        if is_blacklisted:
            return False, f"blacklisted: {blacklist_reason}", {"ip_info": ip_info}
            
        # Check geographic restrictions
        geo_blocked, geo_reason = await self._check_geographic_restrictions(ip_info)
        if geo_blocked:
            return False, f"geo_blocked: {geo_reason}", {"ip_info": ip_info}
            
        # Check reputation score
        if ip_info.reputation_score < 20:
            return False, "low_reputation", {"ip_info": ip_info}
            
        # Check threat score
        if ip_info.threat_score > 80:
            return False, "high_threat_score", {"ip_info": ip_info}
            
        # Additional checks based on network type
        if ip_info.is_tor and not await self._allow_tor():
            return False, "tor_blocked", {"ip_info": ip_info}
            
        if ip_info.is_vpn and not await self._allow_vpn():
            return False, "vpn_blocked", {"ip_info": ip_info}
            
        # Check rate limiting violations
        if await self._has_excessive_violations(ip_address):
            return False, "excessive_violations", {"ip_info": ip_info}
            
        return True, "allowed", {"ip_info": ip_info}
        
    async def _is_whitelisted(self, ip_address: str) -> bool:
        """Check if IP is whitelisted."""
        # Check exact IP whitelist
        if await self.redis.sismember("ip_whitelist", ip_address):
            return True
            
        # Check CIDR whitelist
        whitelist_cidrs = await self.redis.smembers("cidr_whitelist")
        for cidr in whitelist_cidrs:
            try:
                if ipaddress.ip_address(ip_address) in ipaddress.ip_network(cidr):
                    return True
            except ValueError:
                continue
                
        return False
        
    async def _is_blacklisted(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP is blacklisted."""
        # Check exact IP blacklist
        if await self.redis.sismember("ip_blacklist", ip_address):
            return True, "explicit_blacklist"
            
        # Check CIDR blacklist
        blacklist_cidrs = await self.redis.smembers("cidr_blacklist")
        for cidr in blacklist_cidrs:
            try:
                if ipaddress.ip_address(ip_address) in ipaddress.ip_network(cidr):
                    return True, f"cidr_blacklist:{cidr}"
            except ValueError:
                continue
                
        # Check dynamic blacklist (recent violations)
        if await self.redis.sismember("dynamic_blacklist", ip_address):
            return True, "dynamic_blacklist"
            
        return False, ""
        
    async def _check_geographic_restrictions(self, ip_info: IPInfo) -> Tuple[bool, str]:
        """Check geographic access restrictions."""
        if not ip_info.country_code:
            return False, ""
            
        # Check if country is explicitly blocked
        if ip_info.country_code in GEO_RESTRICTIONS.get("blocked_countries", []):
            return True, f"country_blocked:{ip_info.country_code}"
            
        # Check if only specific countries are allowed
        allowed_countries = GEO_RESTRICTIONS.get("allowed_countries", [])
        if allowed_countries and ip_info.country_code not in allowed_countries:
            return True, f"country_not_allowed:{ip_info.country_code}"
            
        return False, ""
        
    async def _allow_tor(self) -> bool:
        """Check if Tor traffic is allowed."""
        return await self.redis.get("allow_tor") == "true"
        
    async def _allow_vpn(self) -> bool:
        """Check if VPN traffic is allowed."""
        return await self.redis.get("allow_vpn") == "true"
        
    async def _has_excessive_violations(self, ip_address: str) -> bool:
        """Check if IP has excessive recent violations."""
        violations_key = f"violations:{ip_address}"
        recent_violations = await self.redis.zcount(
            violations_key,
            time.time() - 3600,  # Last hour
            time.time()
        )
        
        return recent_violations > 50  # More than 50 violations in last hour
        
    async def add_filter_rule(self, rule: FilterRule) -> bool:
        """Add new filtering rule."""
        try:
            rule_data = {
                'rule_type': rule.rule_type,
                'pattern': rule.pattern,
                'reason': rule.reason,
                'created_at': rule.created_at.isoformat(),
                'expires_at': rule.expires_at.isoformat() if rule.expires_at else None,
                'created_by': rule.created_by,
                'is_active': str(rule.is_active)
            }
            
            # Store rule
            await self.redis.hmset(f"filter_rule:{rule.rule_id}", rule_data)
            
            # Add to appropriate set based on rule type
            if rule.rule_type == "whitelist":
                await self.redis.sadd("ip_whitelist", rule.pattern)
            elif rule.rule_type == "blacklist":
                await self.redis.sadd("ip_blacklist", rule.pattern)
            elif rule.rule_type == "cidr_whitelist":
                await self.redis.sadd("cidr_whitelist", rule.pattern)
            elif rule.rule_type == "cidr_blacklist":
                await self.redis.sadd("cidr_blacklist", rule.pattern)
                
            # Log rule creation
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGED,
                event_data={
                    'action': 'filter_rule_added',
                    'rule_id': rule.rule_id,
                    'rule_type': rule.rule_type,
                    'pattern': rule.pattern,
                    'reason': rule.reason
                },
                data_classification="confidential"
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGED,
                event_data={
                    'action': 'filter_rule_add_failed',
                    'rule_id': rule.rule_id,
                    'error': str(e)
                },
                severity=AuditSeverity.ERROR
            )
            return False
            
    async def remove_filter_rule(self, rule_id: str) -> bool:
        """Remove filtering rule."""
        try:
            # Get rule details
            rule_data = await self.redis.hgetall(f"filter_rule:{rule_id}")
            if not rule_data:
                return False
                
            rule_type = rule_data.get('rule_type')
            pattern = rule_data.get('pattern')
            
            # Remove from appropriate set
            if rule_type == "whitelist":
                await self.redis.srem("ip_whitelist", pattern)
            elif rule_type == "blacklist":
                await self.redis.srem("ip_blacklist", pattern)
            elif rule_type == "cidr_whitelist":
                await self.redis.srem("cidr_whitelist", pattern)
            elif rule_type == "cidr_blacklist":
                await self.redis.srem("cidr_blacklist", pattern)
                
            # Remove rule
            await self.redis.delete(f"filter_rule:{rule_id}")
            
            # Log rule removal
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGED,
                event_data={
                    'action': 'filter_rule_removed',
                    'rule_id': rule_id,
                    'rule_type': rule_type,
                    'pattern': pattern
                },
                data_classification="confidential"
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.CONFIG_CHANGED,
                event_data={
                    'action': 'filter_rule_remove_failed',
                    'rule_id': rule_id,
                    'error': str(e)
                },
                severity=AuditSeverity.ERROR
            )
            return False
            
    async def add_to_dynamic_blacklist(
        self,
        ip_address: str,
        reason: str,
        duration: int = 3600
    ):
        """Add IP to dynamic blacklist."""
        # Add to blacklist set
        await self.redis.sadd("dynamic_blacklist", ip_address)
        
        # Set expiry
        await self.redis.setex(f"blacklist_expiry:{ip_address}", duration, reason)
        
        # Log blacklisting
        self.audit_logger.log(
            event_type=AuditEventType.DATA_MODIFIED,
            event_data={
                'action': 'dynamic_blacklist_add',
                'ip_address': ip_address,
                'reason': reason,
                'duration': duration
            },
            severity=AuditSeverity.WARNING,
            data_classification="restricted"
        )
        
    async def get_filter_stats(self) -> Dict[str, any]:
        """Get filtering statistics."""
        stats = {
            'whitelist_count': await self.redis.scard("ip_whitelist"),
            'blacklist_count': await self.redis.scard("ip_blacklist"),
            'cidr_whitelist_count': await self.redis.scard("cidr_whitelist"),
            'cidr_blacklist_count': await self.redis.scard("cidr_blacklist"),
            'dynamic_blacklist_count': await self.redis.scard("dynamic_blacklist"),
            'total_rules': 0
        }
        
        # Count total rules
        rule_keys = await self.redis.keys("filter_rule:*")
        stats['total_rules'] = len(rule_keys)
        
        return stats
        
    async def cleanup_expired_rules(self):
        """Clean up expired filtering rules."""
        now = datetime.now(timezone.utc)
        
        # Get all rules
        rule_keys = await self.redis.keys("filter_rule:*")
        
        for key in rule_keys:
            rule_data = await self.redis.hgetall(key)
            expires_at_str = rule_data.get('expires_at')
            
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if now > expires_at:
                        rule_id = key.split(':', 1)[1]
                        await self.remove_filter_rule(rule_id)
                except ValueError:
                    continue


class ThreatIntelligenceIntegration:
    """Integration with external threat intelligence sources."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize threat intelligence integration."""
        self.redis = redis_client
        self.audit_logger = get_audit_logger()
        
    async def update_threat_feeds(self):
        """Update threat intelligence feeds."""
        feeds_updated = 0
        
        # Update malicious IP feeds
        if await self._update_malicious_ips():
            feeds_updated += 1
            
        # Update Tor exit nodes
        if await self._update_tor_exit_nodes():
            feeds_updated += 1
            
        # Update malicious ASNs
        if await self._update_malicious_asns():
            feeds_updated += 1
            
        self.audit_logger.log(
            event_type=AuditEventType.DATA_MODIFIED,
            event_data={
                'action': 'threat_feeds_updated',
                'feeds_updated': feeds_updated
            }
        )
        
    async def _update_malicious_ips(self) -> bool:
        """Update malicious IP list from threat feeds."""
        try:
            # This would fetch from real threat intelligence APIs
            # For demo, we'll use a placeholder
            malicious_ips = [
                "1.2.3.4",
                "5.6.7.8"
            ]
            
            # Clear existing and add new
            await self.redis.delete("threat_ips")
            if malicious_ips:
                await self.redis.sadd("threat_ips", *malicious_ips)
                
            return True
            
        except Exception as e:
            self.audit_logger.log(
                event_type=AuditEventType.DATA_MODIFIED,
                event_data={
                    'action': 'malicious_ips_update_failed',
                    'error': str(e)
                },
                severity=AuditSeverity.ERROR
            )
            return False
            
    async def _update_tor_exit_nodes(self) -> bool:
        """Update Tor exit nodes list."""
        try:
            # Fetch from Tor directory
            # This is a simplified example
            tor_nodes = [
                "199.87.154.255",
                "185.220.101.182"
            ]
            
            await self.redis.delete("tor_exit_nodes")
            if tor_nodes:
                await self.redis.sadd("tor_exit_nodes", *tor_nodes)
                
            return True
            
        except Exception:
            return False
            
    async def _update_malicious_asns(self) -> bool:
        """Update malicious ASN list."""
        try:
            # Known malicious ASNs
            malicious_asns = [
                "AS1234",  # Example malicious ASN
                "AS5678"
            ]
            
            await self.redis.delete("malicious_asns")
            if malicious_asns:
                await self.redis.sadd("malicious_asns", *malicious_asns)
                
            return True
            
        except Exception:
            return False