"""
Rule evaluation logic for the alerting system.
"""
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import re

from .models import Alert, AlertRule


class RuleEvaluator:
    """Evaluates metrics against rules to determine if alerts should be triggered."""
    
    def __init__(self):
        """Initialize the rule evaluator."""
        self.condition_cache: Dict[str, Callable[[Any], bool]] = {}
    
    def evaluate(self, rule: AlertRule, metric_value: Any) -> Optional[Alert]:
        """Evaluate a metric against a rule.
        
        Args:
            rule: The rule to evaluate
            metric_value: The value of the metric
            
        Returns:
            Optional[Alert]: An alert if the rule is triggered, None otherwise
        """
        # Skip disabled rules
        if not rule.enabled:
            return None
        
        # Check cooldown period
        if (
            rule.last_triggered_at is not None and 
            (datetime.now() - rule.last_triggered_at).total_seconds() < rule.cooldown_period
        ):
            return None
        
        # Parse condition if not in cache
        if rule.condition not in self.condition_cache:
            condition_func = self.parse_condition(rule.condition)
            self.condition_cache[rule.condition] = condition_func
        
        # Evaluate condition
        condition_func = self.condition_cache[rule.condition]
        if condition_func(metric_value):
            # Rule triggered
            message = rule.message_template.format(value=metric_value)
            
            # Update last triggered time
            rule.last_triggered_at = datetime.now()
            
            # Create alert
            return Alert(
                rule_id=rule.rule_id,
                metric_name=rule.metric_name,
                metric_value=metric_value,
                severity=rule.severity,
                message=message
            )
        
        return None
    
    def parse_condition(self, condition: str) -> Callable[[Any], bool]:
        """Parse a condition string into a callable.
        
        Args:
            condition: Condition string (e.g., "> 0.8", "< 100")
            
        Returns:
            Callable[[Any], bool]: Function that takes a value and returns True if condition met
            
        Raises:
            ValueError: If the condition is invalid
        """
        # Regex to match comparison operators
        pattern = r'^(<|<=|>|>=|==|!=)\s*(-?\d+(\.\d+)?)$'
        match = re.match(pattern, condition.strip())
        
        if not match:
            raise ValueError(f"Invalid condition format: {condition}")
        
        operator = match.group(1)
        threshold = float(match.group(2))
        
        def compare(value: Any) -> bool:
            try:
                # Convert value to float for comparison
                numeric_value = float(value)
                
                if operator == "<":
                    return numeric_value < threshold
                elif operator == "<=":
                    return numeric_value <= threshold
                elif operator == ">":
                    return numeric_value > threshold
                elif operator == ">=":
                    return numeric_value >= threshold
                elif operator == "==":
                    return numeric_value == threshold
                elif operator == "!=":
                    return numeric_value != threshold
                else:
                    return False
            except (ValueError, TypeError):
                # If value can't be converted to float, comparison fails
                return False
        
        return compare
    
    def evaluate_metrics(self, rules: List[AlertRule], metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate multiple metrics against multiple rules.
        
        Args:
            rules: List of rules to evaluate
            metrics: Dictionary of metrics
            
        Returns:
            List[Alert]: List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in rules:
            # Check if the metric exists
            if rule.metric_name in metrics:
                metric_value = metrics[rule.metric_name]
                alert = self.evaluate(rule, metric_value)
                if alert:
                    triggered_alerts.append(alert)
        
        return triggered_alerts