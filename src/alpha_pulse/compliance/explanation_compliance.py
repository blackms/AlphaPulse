"""
Explainable AI compliance integration for regulatory reporting.

This module bridges explainable AI capabilities with the compliance
and audit logging systems to ensure regulatory requirements are met.
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from loguru import logger

from ..utils.audit_logger import AuditLogger, AuditEventType
from ..services.explainability_service import ExplainabilityService
from ..models.explanation_result import ExplanationResult


class ComplianceRequirement(Enum):
    """Regulatory compliance requirements for explanations."""
    MIFID_II = "mifid_ii"
    SOX = "sox"
    GDPR = "gdpr"
    BASEL_III = "basel_iii"
    DODD_FRANK = "dodd_frank"


@dataclass
class ComplianceMetrics:
    """Metrics for compliance reporting."""
    total_decisions: int
    explained_decisions: int
    explanation_coverage: float
    avg_explanation_quality: float
    regulatory_flags: Dict[str, int]
    compliance_score: float


@dataclass
class ExplanationAuditEntry:
    """Audit entry for explanations."""
    explanation_id: str
    trade_decision_id: str
    agent_type: str
    symbol: str
    explanation_method: str
    quality_score: float
    compliance_flags: List[str]
    regulatory_requirements: List[ComplianceRequirement]
    timestamp: datetime


class ExplanationComplianceManager:
    """
    Manages compliance integration for explainable AI.
    
    This class ensures that AI trading decisions meet regulatory
    requirements for transparency and explainability.
    """
    
    def __init__(self):
        """Initialize compliance manager."""
        self.audit_logger = AuditLogger()
        self.explainability_service = ExplainabilityService()
        self.compliance_cache = {}
        self.quality_thresholds = {
            ComplianceRequirement.MIFID_II: 0.7,
            ComplianceRequirement.SOX: 0.8,
            ComplianceRequirement.GDPR: 0.6,
            ComplianceRequirement.BASEL_III: 0.75,
            ComplianceRequirement.DODD_FRANK: 0.75
        }
    
    async def audit_trading_decision_with_explanation(
        self,
        trade_decision_id: str,
        agent_type: str,
        symbol: str,
        decision_data: Dict[str, Any],
        explanation: ExplanationResult,
        regulatory_requirements: List[ComplianceRequirement] = None
    ) -> ExplanationAuditEntry:
        """
        Audit a trading decision with its explanation for compliance.
        
        Args:
            trade_decision_id: Unique identifier for the trading decision
            agent_type: Type of trading agent that made the decision
            symbol: Trading symbol
            decision_data: Data about the trading decision
            explanation: Generated explanation for the decision
            regulatory_requirements: Applicable regulatory requirements
            
        Returns:
            Audit entry for the explained decision
        """
        try:
            # Assess explanation quality
            quality_score = await self._assess_explanation_quality(
                explanation, regulatory_requirements or []
            )
            
            # Check compliance flags
            compliance_flags = self._check_compliance_flags(
                explanation, quality_score, regulatory_requirements or []
            )
            
            # Create audit entry
            audit_entry = ExplanationAuditEntry(
                explanation_id=explanation.explanation_id,
                trade_decision_id=trade_decision_id,
                agent_type=agent_type,
                symbol=symbol,
                explanation_method=explanation.method,
                quality_score=quality_score,
                compliance_flags=compliance_flags,
                regulatory_requirements=regulatory_requirements or [],
                timestamp=datetime.now()
            )
            
            # Log to audit system with explanation details
            await self._log_explained_decision(audit_entry, decision_data, explanation)
            
            # Cache for compliance reporting
            self._cache_compliance_entry(audit_entry)
            
            logger.info(f"Audited explained decision {trade_decision_id} with quality {quality_score:.2f}")
            return audit_entry
            
        except Exception as e:
            logger.error(f"Error auditing explained decision {trade_decision_id}: {e}")
            # Log compliance failure
            self.audit_logger.log(
                event_type=AuditEventType.COMPLIANCE_VIOLATION,
                event_data={
                    "trade_decision_id": trade_decision_id,
                    "error": str(e),
                    "explanation_id": explanation.explanation_id if explanation else None,
                    "compliance_status": "FAILED"
                }
            )
            raise
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        regulatory_requirement: Optional[ComplianceRequirement] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report with explanation metrics.
        
        Args:
            start_date: Start date for report period
            end_date: End date for report period
            regulatory_requirement: Specific regulatory requirement to focus on
            symbol: Optional symbol filter
            
        Returns:
            Comprehensive compliance report
        """
        try:
            # Gather explanation audit data
            explanation_audits = await self._get_explanation_audits(
                start_date, end_date, regulatory_requirement, symbol
            )
            
            # Calculate compliance metrics
            metrics = self._calculate_compliance_metrics(explanation_audits)
            
            # Identify compliance issues
            issues = self._identify_compliance_issues(explanation_audits)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(metrics, issues)
            
            # Create comprehensive report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "regulatory_requirement": regulatory_requirement.value if regulatory_requirement else "ALL",
                    "symbol_filter": symbol,
                    "total_entries": len(explanation_audits)
                },
                "compliance_metrics": {
                    "explanation_coverage": metrics.explanation_coverage,
                    "avg_explanation_quality": metrics.avg_explanation_quality,
                    "total_decisions": metrics.total_decisions,
                    "explained_decisions": metrics.explained_decisions,
                    "compliance_score": metrics.compliance_score,
                    "regulatory_flags": metrics.regulatory_flags
                },
                "quality_analysis": {
                    "high_quality_explanations": len([a for a in explanation_audits if a.quality_score >= 0.8]),
                    "medium_quality_explanations": len([a for a in explanation_audits if 0.6 <= a.quality_score < 0.8]),
                    "low_quality_explanations": len([a for a in explanation_audits if a.quality_score < 0.6]),
                    "quality_distribution": self._get_quality_distribution(explanation_audits)
                },
                "compliance_issues": issues,
                "recommendations": recommendations,
                "regulatory_summary": self._generate_regulatory_summary(
                    explanation_audits, regulatory_requirement
                ),
                "audit_trail": [
                    {
                        "explanation_id": audit.explanation_id,
                        "trade_decision_id": audit.trade_decision_id,
                        "symbol": audit.symbol,
                        "quality_score": audit.quality_score,
                        "compliance_flags": audit.compliance_flags,
                        "timestamp": audit.timestamp.isoformat()
                    }
                    for audit in explanation_audits[-50:]  # Last 50 for summary
                ]
            }
            
            # Log compliance report generation
            self.audit_logger.log(
                event_type=AuditEventType.COMPLIANCE_REPORT,
                event_data={
                    "report_type": "explanation_compliance",
                    "period_days": (end_date - start_date).days,
                    "explanation_coverage": metrics.explanation_coverage,
                    "compliance_score": metrics.compliance_score,
                    "regulatory_requirement": regulatory_requirement.value if regulatory_requirement else "ALL"
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    async def check_explanation_adequacy(
        self,
        explanation: ExplanationResult,
        regulatory_requirements: List[ComplianceRequirement]
    ) -> Tuple[bool, List[str]]:
        """
        Check if an explanation meets regulatory adequacy requirements.
        
        Args:
            explanation: Explanation to check
            regulatory_requirements: Applicable regulatory requirements
            
        Returns:
            Tuple of (is_adequate, list_of_issues)
        """
        issues = []
        
        # Check explanation quality against thresholds
        for requirement in regulatory_requirements:
            threshold = self.quality_thresholds.get(requirement, 0.7)
            if explanation.confidence < threshold:
                issues.append(f"Explanation quality {explanation.confidence:.2f} below {requirement.value} threshold {threshold}")
        
        # Check for required explanation components
        if not explanation.feature_importance:
            issues.append("Missing feature importance data")
        
        if not explanation.explanation_text:
            issues.append("Missing human-readable explanation")
        
        if len(explanation.feature_importance) < 3:
            issues.append("Insufficient feature analysis (less than 3 features)")
        
        # Check for regulatory-specific requirements
        for requirement in regulatory_requirements:
            requirement_issues = self._check_regulatory_specific_requirements(
                explanation, requirement
            )
            issues.extend(requirement_issues)
        
        is_adequate = len(issues) == 0
        return is_adequate, issues
    
    async def export_compliance_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format_type: str = "json",
        regulatory_requirement: Optional[ComplianceRequirement] = None
    ) -> Dict[str, Any]:
        """
        Export compliance data for regulatory submission.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            format_type: Export format (json, csv, xml)
            regulatory_requirement: Specific regulatory focus
            
        Returns:
            Exported compliance data
        """
        try:
            # Generate comprehensive compliance report
            report = await self.generate_compliance_report(
                start_date, end_date, regulatory_requirement
            )
            
            # Add regulatory-specific formatting
            if regulatory_requirement:
                report = self._format_for_regulatory_requirement(report, regulatory_requirement)
            
            # Log export activity
            self.audit_logger.log(
                event_type=AuditEventType.DATA_EXPORT,
                event_data={
                    "export_type": "compliance_explanation_data",
                    "format": format_type,
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "regulatory_requirement": regulatory_requirement.value if regulatory_requirement else "ALL",
                    "records_exported": len(report.get("audit_trail", []))
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error exporting compliance data: {e}")
            raise
    
    # Private methods
    
    async def _assess_explanation_quality(
        self,
        explanation: ExplanationResult,
        regulatory_requirements: List[ComplianceRequirement]
    ) -> float:
        """Assess the quality of an explanation for compliance purposes."""
        base_score = explanation.confidence
        
        # Adjust based on completeness
        completeness_score = 0.0
        if explanation.feature_importance:
            completeness_score += 0.3
        if explanation.explanation_text:
            completeness_score += 0.2
        if explanation.visualization_data:
            completeness_score += 0.1
        
        # Adjust based on regulatory requirements
        regulatory_adjustment = 0.0
        for requirement in regulatory_requirements:
            if requirement in [ComplianceRequirement.SOX, ComplianceRequirement.DODD_FRANK]:
                # Higher standards for financial reporting
                regulatory_adjustment -= 0.1
        
        final_score = min(1.0, base_score + completeness_score + regulatory_adjustment)
        return max(0.0, final_score)
    
    def _check_compliance_flags(
        self,
        explanation: ExplanationResult,
        quality_score: float,
        regulatory_requirements: List[ComplianceRequirement]
    ) -> List[str]:
        """Check for compliance flags based on explanation quality."""
        flags = []
        
        if quality_score < 0.6:
            flags.append("LOW_QUALITY")
        
        if quality_score < 0.5:
            flags.append("INADEQUATE_EXPLANATION")
        
        if not explanation.feature_importance:
            flags.append("MISSING_FEATURE_ANALYSIS")
        
        for requirement in regulatory_requirements:
            threshold = self.quality_thresholds.get(requirement, 0.7)
            if quality_score < threshold:
                flags.append(f"BELOW_{requirement.value.upper()}_THRESHOLD")
        
        return flags
    
    async def _log_explained_decision(
        self,
        audit_entry: ExplanationAuditEntry,
        decision_data: Dict[str, Any],
        explanation: ExplanationResult
    ) -> None:
        """Log the explained decision to audit system."""
        self.audit_logger.log(
            event_type=AuditEventType.TRADE_DECISION,
            event_data={
                **decision_data,
                "explanation_id": audit_entry.explanation_id,
                "explanation_method": audit_entry.explanation_method,
                "explanation_quality": audit_entry.quality_score,
                "compliance_flags": audit_entry.compliance_flags,
                "regulatory_requirements": [req.value for req in audit_entry.regulatory_requirements],
                "key_features": explanation.feature_importance[:5] if explanation.feature_importance else [],
                "explanation_summary": explanation.explanation_text[:200] if explanation.explanation_text else ""
            }
        )
    
    def _cache_compliance_entry(self, audit_entry: ExplanationAuditEntry) -> None:
        """Cache compliance entry for reporting."""
        date_key = audit_entry.timestamp.date().isoformat()
        if date_key not in self.compliance_cache:
            self.compliance_cache[date_key] = []
        self.compliance_cache[date_key].append(audit_entry)
    
    async def _get_explanation_audits(
        self,
        start_date: datetime,
        end_date: datetime,
        regulatory_requirement: Optional[ComplianceRequirement],
        symbol: Optional[str]
    ) -> List[ExplanationAuditEntry]:
        """Get explanation audit entries for the specified period."""
        # This would typically query the database
        # For now, return cached entries that match criteria
        audits = []
        
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_key = current_date.isoformat()
            if date_key in self.compliance_cache:
                for audit in self.compliance_cache[date_key]:
                    if symbol and audit.symbol != symbol:
                        continue
                    if regulatory_requirement and regulatory_requirement not in audit.regulatory_requirements:
                        continue
                    audits.append(audit)
            
            current_date += timedelta(days=1)
        
        return audits
    
    def _calculate_compliance_metrics(self, audits: List[ExplanationAuditEntry]) -> ComplianceMetrics:
        """Calculate compliance metrics from audit entries."""
        if not audits:
            return ComplianceMetrics(0, 0, 0.0, 0.0, {}, 0.0)
        
        total_decisions = len(audits)
        explained_decisions = len([a for a in audits if a.quality_score > 0])
        explanation_coverage = explained_decisions / total_decisions if total_decisions > 0 else 0
        avg_quality = sum(a.quality_score for a in audits) / len(audits)
        
        # Count regulatory flags
        flag_counts = {}
        for audit in audits:
            for flag in audit.compliance_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Calculate overall compliance score
        compliance_score = min(1.0, explanation_coverage * avg_quality)
        
        return ComplianceMetrics(
            total_decisions=total_decisions,
            explained_decisions=explained_decisions,
            explanation_coverage=explanation_coverage,
            avg_explanation_quality=avg_quality,
            regulatory_flags=flag_counts,
            compliance_score=compliance_score
        )
    
    def _identify_compliance_issues(self, audits: List[ExplanationAuditEntry]) -> List[Dict[str, Any]]:
        """Identify compliance issues from audit entries."""
        issues = []
        
        # Check for low quality explanations
        low_quality_count = len([a for a in audits if a.quality_score < 0.6])
        if low_quality_count > 0:
            issues.append({
                "type": "LOW_QUALITY_EXPLANATIONS",
                "count": low_quality_count,
                "severity": "HIGH" if low_quality_count > len(audits) * 0.1 else "MEDIUM",
                "description": f"{low_quality_count} explanations below quality threshold"
            })
        
        # Check for missing explanations
        missing_explanations = len([a for a in audits if not a.explanation_id])
        if missing_explanations > 0:
            issues.append({
                "type": "MISSING_EXPLANATIONS",
                "count": missing_explanations,
                "severity": "CRITICAL",
                "description": f"{missing_explanations} trading decisions without explanations"
            })
        
        return issues
    
    def _generate_compliance_recommendations(
        self,
        metrics: ComplianceMetrics,
        issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if metrics.explanation_coverage < 0.95:
            recommendations.append("Increase explanation coverage to meet regulatory requirements")
        
        if metrics.avg_explanation_quality < 0.75:
            recommendations.append("Improve explanation quality through better feature engineering")
        
        for issue in issues:
            if issue["severity"] == "CRITICAL":
                recommendations.append(f"URGENT: Address {issue['description']}")
        
        return recommendations
    
    def _get_quality_distribution(self, audits: List[ExplanationAuditEntry]) -> Dict[str, int]:
        """Get quality score distribution."""
        distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        
        for audit in audits:
            score = audit.quality_score
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution
    
    def _generate_regulatory_summary(
        self,
        audits: List[ExplanationAuditEntry],
        regulatory_requirement: Optional[ComplianceRequirement]
    ) -> Dict[str, Any]:
        """Generate regulatory-specific summary."""
        summary = {
            "total_applicable_decisions": len(audits),
            "compliance_rate": 0.0,
            "average_quality": 0.0,
            "regulatory_notes": []
        }
        
        if audits:
            applicable_audits = audits
            if regulatory_requirement:
                applicable_audits = [
                    a for a in audits 
                    if regulatory_requirement in a.regulatory_requirements
                ]
            
            if applicable_audits:
                threshold = self.quality_thresholds.get(regulatory_requirement, 0.7)
                compliant_count = len([a for a in applicable_audits if a.quality_score >= threshold])
                summary["compliance_rate"] = compliant_count / len(applicable_audits)
                summary["average_quality"] = sum(a.quality_score for a in applicable_audits) / len(applicable_audits)
                
                if regulatory_requirement:
                    summary["regulatory_notes"].append(
                        f"Compliance with {regulatory_requirement.value} requirements: "
                        f"{summary['compliance_rate']:.1%}"
                    )
        
        return summary
    
    def _check_regulatory_specific_requirements(
        self,
        explanation: ExplanationResult,
        requirement: ComplianceRequirement
    ) -> List[str]:
        """Check regulatory-specific requirements."""
        issues = []
        
        if requirement == ComplianceRequirement.MIFID_II:
            # MiFID II requires clear explanation of automated decisions
            if not explanation.explanation_text or len(explanation.explanation_text) < 50:
                issues.append("MiFID II: Insufficient human-readable explanation")
        
        elif requirement == ComplianceRequirement.SOX:
            # SOX requires detailed audit trail
            if not explanation.audit_trail:
                issues.append("SOX: Missing detailed audit trail")
        
        elif requirement == ComplianceRequirement.GDPR:
            # GDPR requires clear explanation of automated decision-making
            if not explanation.explanation_text:
                issues.append("GDPR: Missing explanation for automated decision")
        
        return issues
    
    def _format_for_regulatory_requirement(
        self,
        report: Dict[str, Any],
        requirement: ComplianceRequirement
    ) -> Dict[str, Any]:
        """Format report for specific regulatory requirement."""
        if requirement == ComplianceRequirement.MIFID_II:
            # Add MiFID II specific sections
            report["mifid_ii_compliance"] = {
                "automated_decision_transparency": report["compliance_metrics"]["explanation_coverage"],
                "client_notification_compliance": True,  # Would be calculated based on actual requirements
                "record_keeping_compliance": True
            }
        
        return report