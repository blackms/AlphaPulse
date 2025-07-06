"""
Integration layer between online learning and trading agents.
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from alpha_pulse.ml.online.online_learning_service import OnlineLearningService
from alpha_pulse.ml.online.online_model import (
    LearningSessionRequest,
    PredictionRequest,
    StreamingBatch
)
from alpha_pulse.agents.interfaces import TradeSignal, SignalDirection
from alpha_pulse.monitoring.collector import EnhancedMetricsCollector


class OnlineLearningIntegration:
    """
    Integrates online learning with trading agents for continuous improvement.
    """
    
    def __init__(
        self,
        online_learning_service: OnlineLearningService,
        metrics_collector: Optional[EnhancedMetricsCollector] = None
    ):
        self.online_service = online_learning_service
        self.metrics_collector = metrics_collector
        self.agent_sessions: Dict[str, str] = {}  # agent_id -> session_id
        self.learning_config = {
            'update_frequency': 10,  # Update every 10 signals
            'min_confidence_threshold': 0.6,
            'performance_window': 100
        }
        self.signal_buffer: Dict[str, List[Dict[str, Any]]] = {}
        
    async def initialize_agent_learning(self, agent_id: str, strategy: str = "adaptive_forest") -> str:
        """
        Initialize online learning session for an agent.
        
        Args:
            agent_id: Trading agent identifier
            strategy: Learning strategy to use
            
        Returns:
            Session ID
        """
        request = LearningSessionRequest(
            agent_id=agent_id,
            strategy=strategy,
            config={
                'validation': {
                    'method': 'prequential',
                    'window_size': 100
                },
                'adaptive_control': {
                    'learning_rate': 0.01,
                    'enable_drift_detection': True,
                    'drift_threshold': 0.3
                },
                'model_params': {
                    'n_estimators': 10,
                    'max_features': 'sqrt'
                }
            }
        )
        
        response = await self.online_service.start_session(request)
        self.agent_sessions[agent_id] = response.session_id
        self.signal_buffer[agent_id] = []
        
        logger.info(f"Initialized online learning for agent {agent_id} with session {response.session_id}")
        return response.session_id
        
    async def process_agent_signal(
        self,
        agent_id: str,
        signal: TradeSignal,
        market_features: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process agent signal through online learning.
        
        Args:
            agent_id: Trading agent identifier
            signal: Generated trade signal
            market_features: Current market features
            
        Returns:
            Enhanced prediction with confidence adjustment
        """
        if agent_id not in self.agent_sessions:
            # Initialize learning for this agent
            await self.initialize_agent_learning(agent_id)
            
        session_id = self.agent_sessions[agent_id]
        
        # Prepare features for prediction
        features = self._prepare_features(signal, market_features)
        
        # Get online learning prediction
        pred_request = PredictionRequest(
            features=features,
            return_proba=True,
            include_uncertainty=True
        )
        
        try:
            prediction = await self.online_service.predict(session_id, pred_request)
            
            # Adjust signal confidence based on model confidence
            adjusted_confidence = signal.confidence * prediction.confidence
            
            # Store signal for later update
            self.signal_buffer[agent_id].append({
                'features': features,
                'signal': signal,
                'prediction': prediction,
                'timestamp': datetime.utcnow()
            })
            
            # Update model if buffer is full
            if len(self.signal_buffer[agent_id]) >= self.learning_config['update_frequency']:
                await self._update_model(agent_id)
            
            return {
                'original_confidence': signal.confidence,
                'model_confidence': prediction.confidence,
                'adjusted_confidence': adjusted_confidence,
                'prediction': prediction.prediction,
                'uncertainty': prediction.uncertainty
            }
            
        except Exception as e:
            logger.error(f"Error in online learning prediction for agent {agent_id}: {str(e)}")
            return None
            
    async def update_with_outcome(
        self,
        agent_id: str,
        signal_id: str,
        actual_outcome: Dict[str, Any]
    ) -> None:
        """
        Update model with actual trading outcome.
        
        Args:
            agent_id: Trading agent identifier
            signal_id: Original signal identifier
            actual_outcome: Actual trading result
        """
        if agent_id not in self.agent_sessions:
            return
            
        session_id = self.agent_sessions[agent_id]
        
        # Find the corresponding signal in buffer
        signal_data = None
        for data in self.signal_buffer.get(agent_id, []):
            if data['signal'].metadata.get('signal_id') == signal_id:
                signal_data = data
                break
                
        if not signal_data:
            logger.warning(f"Signal {signal_id} not found in buffer for agent {agent_id}")
            return
            
        # Prepare update data
        label = self._compute_label(signal_data['signal'], actual_outcome)
        
        # Create batch update
        batch = StreamingBatch(
            session_id=session_id,
            data_points=[signal_data['features']],
            labels=[label],
            timestamp=datetime.utcnow()
        )
        
        try:
            metrics = await self.online_service.update_batch(session_id, batch)
            
            # Track performance
            if self.metrics_collector:
                await self.metrics_collector.collect_and_store(
                    agent_data={
                        f"{agent_id}_online_learning": {
                            'accuracy': metrics.get('accuracy', 0),
                            'loss': metrics.get('loss', 0),
                            'drift_score': metrics.get('drift_score', 0)
                        }
                    }
                )
                
            logger.debug(f"Updated online learning for agent {agent_id}: {metrics}")
            
        except Exception as e:
            logger.error(f"Error updating online learning for agent {agent_id}: {str(e)}")
            
    async def _update_model(self, agent_id: str) -> None:
        """
        Perform batch update of the model.
        """
        if not self.signal_buffer.get(agent_id):
            return
            
        session_id = self.agent_sessions[agent_id]
        buffer = self.signal_buffer[agent_id]
        
        # Prepare batch data
        features = [data['features'] for data in buffer]
        # For now, use predicted direction as pseudo-label
        labels = [1 if data['signal'].direction == SignalDirection.LONG else -1 for data in buffer]
        
        batch = StreamingBatch(
            session_id=session_id,
            data_points=features,
            labels=labels,
            timestamp=datetime.utcnow()
        )
        
        try:
            await self.online_service.update_batch(session_id, batch)
            # Clear buffer after update
            self.signal_buffer[agent_id] = []
            logger.debug(f"Batch update completed for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error in batch update for agent {agent_id}: {str(e)}")
            
    def _prepare_features(self, signal: TradeSignal, market_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for online learning model.
        """
        features = {
            'confidence': signal.confidence,
            'direction': 1 if signal.direction == SignalDirection.LONG else -1,
            'has_target': 1 if signal.target_price else 0,
            'has_stop': 1 if signal.stop_loss else 0,
            **market_features
        }
        
        # Add agent-specific features from metadata
        if signal.metadata:
            for key, value in signal.metadata.items():
                if isinstance(value, (int, float)):
                    features[f'meta_{key}'] = value
                    
        return features
        
    def _compute_label(self, signal: TradeSignal, outcome: Dict[str, Any]) -> float:
        """
        Compute label from actual trading outcome.
        """
        # Simple binary classification: profitable or not
        profit = outcome.get('profit', 0)
        return 1.0 if profit > 0 else 0.0
        
    async def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get online learning performance metrics for an agent.
        """
        if agent_id not in self.agent_sessions:
            return None
            
        session_id = self.agent_sessions[agent_id]
        
        try:
            metrics = await self.online_service.get_learning_metrics(
                session_id,
                window_size=self.learning_config['performance_window']
            )
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance for agent {agent_id}: {str(e)}")
            return None
            
    async def stop_all_sessions(self) -> None:
        """
        Stop all active learning sessions.
        """
        for agent_id, session_id in self.agent_sessions.items():
            try:
                await self.online_service.stop_session(session_id, save_checkpoint=True)
                logger.info(f"Stopped online learning session for agent {agent_id}")
            except Exception as e:
                logger.error(f"Error stopping session for agent {agent_id}: {str(e)}")