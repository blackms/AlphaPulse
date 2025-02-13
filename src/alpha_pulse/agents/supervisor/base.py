"""
Base implementations for the supervisor agent system.
"""
import psutil
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger

from ..interfaces import BaseTradeAgent, MarketData, TradeSignal
from .interfaces import (
    ISelfSupervisedAgent,
    AgentState,
    AgentHealth,
    Task
)


class BaseSelfSupervisedAgent(BaseTradeAgent, ISelfSupervisedAgent):
    """
    Base implementation of a self-supervised agent.
    Extends BaseTradeAgent with self-supervision capabilities.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize base self-supervised agent."""
        config = config or {}
        super().__init__(agent_id, config)
        self._state = AgentState.INITIALIZING
        self._last_active = datetime.now()
        self._error_count = 0
        self._last_error = None
        self._process = psutil.Process()
        self._optimization_threshold = config.get("optimization_threshold", 0.7)
        self._performance_history: List[Dict[str, float]] = []
        self._default_config = {
            "optimization_threshold": 0.7,
            "monitoring_interval": 60,
            "max_errors": 10,
            "recovery_timeout": 300,
            "memory_limit_mb": 1000,
            "cpu_limit_percent": 80
        }
        self.config.update({
            k: config.get(k, v)
            for k, v in self._default_config.items()
        })
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the agent with configuration."""
        await super().initialize(config)
        self._state = AgentState.ACTIVE
        self._last_active = datetime.now()
        logger.info(f"Initialized self-supervised agent '{self.agent_id}'")
        
    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """Generate trading signals with self-supervision."""
        try:
            self._last_active = datetime.now()
            
            # Check if optimization is needed
            metrics = await self.self_evaluate()
            if metrics.get("performance_score", 1.0) < self._optimization_threshold:
                logger.info(f"Agent '{self.agent_id}' triggering self-optimization")
                self._state = AgentState.OPTIMIZING
                await self.optimize()
                self._state = AgentState.ACTIVE
            
            # Generate signals using parent implementation
            signals = await super().generate_signals(market_data)
            
            return signals
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            self._state = AgentState.ERROR
            logger.error(f"Error in agent '{self.agent_id}': {str(e)}")
            raise
            
    async def self_evaluate(self) -> Dict[str, float]:
        """Basic self-evaluation implementation."""
        if not self.metrics:
            return {"performance_score": 1.0}  # Default score when no metrics
            
        # Calculate performance score based on metrics
        performance_score = (
            self.metrics.signal_accuracy * 0.4 +
            self.metrics.profit_factor * 0.3 +
            self.metrics.sharpe_ratio * 0.2 +
            (1 - self.metrics.max_drawdown) * 0.1
        )
        
        metrics = {
            "performance_score": performance_score,
            "signal_accuracy": self.metrics.signal_accuracy,
            "profit_factor": self.metrics.profit_factor,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "max_drawdown": self.metrics.max_drawdown
        }
        
        self._performance_history.append(metrics)
        return metrics
        
    async def optimize(self) -> None:
        """Basic optimization implementation."""
        if not self._performance_history:
            return
            
        # Analyze performance history
        recent_scores = [
            m["performance_score"]
            for m in self._performance_history[-5:]  # Look at last 5 evaluations
        ]
        
        if len(recent_scores) < 5:
            return
            
        # Check for declining performance
        if all(recent_scores[i] < recent_scores[i-1] for i in range(1, len(recent_scores))):
            logger.warning(f"Agent {self.agent_id} detected declining performance")
            # Implement optimization logic in derived classes
            
    async def get_health_status(self) -> AgentHealth:
        """Get current health metrics."""
        try:
            memory_info = self._process.memory_info()
            cpu_percent = self._process.cpu_percent()
            
            metrics = await self.self_evaluate()
            
            return AgentHealth(
                state=self._state,
                last_active=self._last_active,
                error_count=self._error_count,
                last_error=self._last_error,
                memory_usage=memory_info.rss / 1024 / 1024,  # Convert to MB
                cpu_usage=cpu_percent,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error getting health status for agent {self.agent_id}: {str(e)}")
            return AgentHealth(
                state=AgentState.ERROR,
                last_active=self._last_active,
                error_count=self._error_count + 1,
                last_error=str(e),
                memory_usage=0.0,
                cpu_usage=0.0,
                metrics={}
            )
            
    async def execute_task(self, task: Task) -> None:
        """Execute a supervisor-assigned task."""
        try:
            self._last_active = datetime.now()
            
            if task.task_type == "optimize":
                await self.optimize()
            elif task.task_type == "evaluate":
                metrics = await self.self_evaluate()
                task.result = metrics
            else:
                # Propagate error instead of handling it
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            task.completed_at = datetime.now()
            task.status = "completed"
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            task.error = str(e)
            task.status = "failed"
            logger.error(f"Error executing task {task.task_id} for agent {self.agent_id}: {str(e)}")
            raise  # Re-raise the exception to propagate it
            
    async def pause(self) -> None:
        """Pause agent operations."""
        if self._state == AgentState.ACTIVE:
            self._state = AgentState.INACTIVE
            logger.info(f"Agent '{self.agent_id}' paused")
            
    async def resume(self) -> None:
        """Resume agent operations."""
        if self._state == AgentState.INACTIVE:
            self._state = AgentState.ACTIVE
            self._last_active = datetime.now()
            logger.info(f"Agent {self.agent_id} resumed")
            
    async def stop(self) -> None:
        """Stop agent operations completely."""
        await self.pause()  # First pause the agent
        self._state = AgentState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")