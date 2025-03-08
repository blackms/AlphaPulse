"""
Type definitions for the exchange data synchronization module.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union


class SyncStatus(Enum):
    """Status of a synchronization task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __str__(self):
        return self.value