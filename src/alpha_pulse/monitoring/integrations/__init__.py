"""
Integration modules for the monitoring system.

This package contains modules that integrate the monitoring system with
other components of the AlphaPulse system, such as alerting, dashboard,
and external systems.
"""

from .alerting import (
    alerting_integration,
    initialize_alerting,
    process_metrics_for_alerts,
    shutdown_alerting
)

__all__ = [
    'alerting_integration',
    'initialize_alerting',
    'process_metrics_for_alerts',
    'shutdown_alerting'
]