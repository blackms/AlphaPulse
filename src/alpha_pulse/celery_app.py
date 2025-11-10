"""
Celery application configuration for AlphaPulse.

Configures distributed task queue for background jobs like credential health checks.
"""

from celery import Celery
from celery.schedules import crontab
from loguru import logger

from alpha_pulse.config.secure_settings import get_settings

settings = get_settings()

# Create Celery app
app = Celery(
    "alpha_pulse",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "alpha_pulse.tasks.credential_health",
    ],
)

# Celery configuration
app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_routes={
        "alpha_pulse.tasks.credential_health.*": {"queue": "health_checks"},
    },
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={"master_name": "mymaster"},
    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetching for fair distribution
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    # Task execution limits
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Retry if worker crashes
)

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    "check-credentials-health": {
        "task": "alpha_pulse.tasks.credential_health.check_all_credentials_health",
        "schedule": crontab(
            minute=0,
            hour=f"*/{settings.credential_health_check_interval_hours}",
        ),  # Every N hours
        "options": {"queue": "health_checks"},
    },
}

logger.info(
    f"Celery app configured: broker={settings.celery_broker_url}, "
    f"health_check_interval={settings.credential_health_check_interval_hours}h"
)


if __name__ == "__main__":
    app.start()
