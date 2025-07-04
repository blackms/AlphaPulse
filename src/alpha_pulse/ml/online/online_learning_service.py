"""Service layer for online learning system."""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
import uuid
import json
import pickle
from pathlib import Path

from alpha_pulse.ml.online.online_learner import (
    BaseOnlineLearner, OnlineLearnerEnsemble, AdaptiveLearningController,
    OnlineDataPoint, LearningState
)
from alpha_pulse.ml.online.incremental_models import (
    IncrementalSGD, IncrementalNaiveBayes, IncrementalPassiveAggressive,
    HoeffdingTree, AdaptiveRandomForest, OnlineGradientBoosting
)
from alpha_pulse.ml.online.adaptive_algorithms import (
    AdaptiveLearningRateScheduler, AdaptiveOptimizer, MultiArmedBandit,
    AdaptiveMetaLearner
)
from alpha_pulse.ml.online.memory_manager import MemoryManager
from alpha_pulse.ml.online.streaming_validation import StreamingValidator
from alpha_pulse.ml.online.online_model import (
    OnlineLearningSession, DriftEvent, ModelCheckpoint, StreamingMetrics,
    OnlineDataPointModel, StreamingBatch, LearningSessionRequest,
    LearningSessionResponse, PredictionRequest, PredictionResponse,
    DriftDetectionAlert, LearningMetrics, ModelUpdateNotification,
    OnlineLearningConfig
)

logger = logging.getLogger(__name__)


class OnlineLearningService:
    """Main service for managing online learning operations."""
    
    def __init__(self, db: Session, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Active sessions
        self.active_sessions: Dict[str, 'OnlineLearningSession'] = {}
        self.learners: Dict[str, Union[BaseOnlineLearner, OnlineLearnerEnsemble]] = {}
        self.validators: Dict[str, StreamingValidator] = {}
        self.controllers: Dict[str, AdaptiveLearningController] = {}
        
        # Background tasks
        self.background_tasks = set()
        self.shutdown_event = asyncio.Event()
        
    async def start_session(self, request: LearningSessionRequest) -> LearningSessionResponse:
        """Start a new online learning session."""
        session_id = str(uuid.uuid4())
        
        # Create session record
        session = OnlineLearningSession(
            session_id=session_id,
            agent_id=request.agent_id,
            strategy=request.strategy,
            metadata=request.config
        )
        self.db.add(session)
        self.db.commit()
        
        # Initialize learner
        learner = self._create_learner(request.strategy, request.config)
        self.learners[session_id] = learner
        
        # Initialize validator
        validator_config = request.config.get('validation', {})
        self.validators[session_id] = StreamingValidator(validator_config)
        
        # Initialize adaptive controller
        controller_config = request.config.get('adaptive_control', {})
        self.controllers[session_id] = AdaptiveLearningController(controller_config)
        
        # Store in active sessions
        self.active_sessions[session_id] = session
        
        # Start background monitoring
        task = asyncio.create_task(self._monitor_session(session_id))
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Started online learning session {session_id} for agent {request.agent_id}")
        
        return LearningSessionResponse(
            session_id=session_id,
            status="active",
            start_time=session.start_time,
            config=request.config
        )
        
    def _create_learner(self, strategy: str, config: Dict[str, Any]) -> Union[BaseOnlineLearner, OnlineLearnerEnsemble]:
        """Create appropriate learner based on strategy."""
        model_type = config.get('model_type', 'sgd')
        
        if strategy == 'ensemble':
            # Create ensemble of learners
            ensemble = OnlineLearnerEnsemble(config)
            
            # Add diverse learners
            learners_config = config.get('learners', [
                {'type': 'sgd', 'config': {}},
                {'type': 'naive_bayes', 'config': {}},
                {'type': 'hoeffding_tree', 'config': {}}
            ])
            
            for learner_spec in learners_config:
                learner = self._create_single_learner(learner_spec['type'], learner_spec['config'])
                ensemble.add_learner(learner)
                
            return ensemble
            
        else:
            # Single learner
            return self._create_single_learner(model_type, config)
            
    def _create_single_learner(self, model_type: str, config: Dict[str, Any]) -> BaseOnlineLearner:
        """Create a single learner instance."""
        if model_type == 'sgd':
            return IncrementalSGD(config)
        elif model_type == 'naive_bayes':
            return IncrementalNaiveBayes(config)
        elif model_type == 'passive_aggressive':
            return IncrementalPassiveAggressive(config)
        elif model_type == 'hoeffding_tree':
            return HoeffdingTree(config)
        elif model_type == 'adaptive_forest':
            return AdaptiveRandomForest(config)
        elif model_type == 'gradient_boosting':
            return OnlineGradientBoosting(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    async def process_batch(self, session_id: str, batch: StreamingBatch) -> Dict[str, Any]:
        """Process a batch of streaming data."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        learner = self.learners[session_id]
        validator = self.validators[session_id]
        controller = self.controllers[session_id]
        
        results = {
            'session_id': session_id,
            'batch_id': batch.batch_id,
            'n_processed': 0,
            'predictions': [],
            'metrics': {}
        }
        
        # Process each data point
        for data_point_model in batch.data_points:
            # Convert to internal format
            data_point = OnlineDataPoint(
                timestamp=data_point_model.timestamp,
                features=np.array(data_point_model.features),
                label=data_point_model.label,
                weight=data_point_model.weight,
                metadata=data_point_model.metadata
            )
            
            # Learn and predict
            prediction = await self._process_single_point(
                session_id, learner, data_point
            )
            
            if prediction is not None:
                results['predictions'].append(prediction)
                
            results['n_processed'] += 1
            
        # Update session
        session = self.active_sessions[session_id]
        session.n_samples_processed += results['n_processed']
        self.db.commit()
        
        # Validate if enough data
        if len(results['predictions']) > 0 and data_point_model.label is not None:
            actuals = [dp.label for dp in batch.data_points if dp.label is not None]
            validation_result = validator.validate_stream(
                session_id,
                np.array(results['predictions']),
                np.array(actuals)
            )
            results['validation'] = validation_result
            
        return results
        
    async def _process_single_point(self, session_id: str, 
                                   learner: Union[BaseOnlineLearner, OnlineLearnerEnsemble],
                                   data_point: OnlineDataPoint) -> Optional[float]:
        """Process a single data point."""
        try:
            # Check if adaptation needed
            controller = self.controllers[session_id]
            
            if hasattr(learner, 'state') and controller.should_adapt(learner.state.current_accuracy):
                await self._adapt_learner(session_id, learner)
                
            # Learn from data point
            prediction = learner.learn_one(data_point)
            
            # Record metrics
            if prediction is not None and data_point.label is not None:
                error = abs(prediction - data_point.label)
                await self._record_metric(session_id, 'prediction_error', error)
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing data point in session {session_id}: {e}")
            return None
            
    async def _adapt_learner(self, session_id: str, 
                            learner: Union[BaseOnlineLearner, OnlineLearnerEnsemble]) -> None:
        """Adapt learner based on performance."""
        controller = self.controllers[session_id]
        
        # Get market conditions (simplified)
        market_conditions = {
            'volatility': np.random.random(),  # Would use real market data
            'trend_strength': np.random.random(),
            'regime': 'normal'
        }
        
        # Select new strategy
        new_strategy = controller.select_strategy(market_conditions)
        
        # Apply adaptations
        if hasattr(learner, 'learning_rate'):
            learner.learning_rate = controller.adapt_learning_rate(
                learner.learning_rate,
                market_conditions['volatility'],
                learner.state.current_accuracy
            )
            
        # Record adaptation
        await self._record_adaptation(session_id, new_strategy, market_conditions)
        
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make predictions using online model."""
        if request.session_id not in self.active_sessions:
            raise ValueError(f"Session {request.session_id} not found")
            
        learner = self.learners[request.session_id]
        features = np.array(request.features)
        
        # Make predictions
        predictions = learner.predict(features)
        
        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            try:
                probabilities = learner.predict_proba(features)
            except:
                logger.warning("Probabilities not available for this model")
                
        # Get model version
        model_version = 1
        if hasattr(learner, 'state'):
            model_version = learner.state.model_version
            
        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist() if probabilities is not None else None,
            model_version=model_version,
            timestamp=datetime.now()
        )
        
    async def checkpoint_model(self, session_id: str, is_best: bool = False) -> str:
        """Save model checkpoint."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        learner = self.learners[session_id]
        session = self.active_sessions[session_id]
        
        # Generate checkpoint path
        checkpoint_path = self.checkpoint_dir / f"{session_id}_{datetime.now().timestamp()}.pkl"
        
        # Save model
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(learner, f)
            
        # Get performance metrics
        metrics = {}
        if hasattr(learner, 'state'):
            metrics['accuracy'] = learner.state.current_accuracy
            metrics['n_samples'] = learner.state.n_samples_seen
            
        # Record checkpoint
        checkpoint = ModelCheckpoint(
            session_id=session_id,
            model_version=learner.state.model_version if hasattr(learner, 'state') else 1,
            performance_metrics=metrics,
            model_path=str(checkpoint_path),
            is_best=is_best
        )
        self.db.add(checkpoint)
        self.db.commit()
        
        logger.info(f"Saved checkpoint for session {session_id} at {checkpoint_path}")
        
        return str(checkpoint_path)
        
    async def load_checkpoint(self, checkpoint_path: str) -> str:
        """Load model from checkpoint."""
        session_id = str(uuid.uuid4())
        
        # Load model
        with open(checkpoint_path, 'rb') as f:
            learner = pickle.load(f)
            
        # Create new session
        session = OnlineLearningSession(
            session_id=session_id,
            agent_id="restored",
            strategy="restored",
            metadata={'checkpoint_path': checkpoint_path}
        )
        self.db.add(session)
        self.db.commit()
        
        # Register learner
        self.learners[session_id] = learner
        self.active_sessions[session_id] = session
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} as session {session_id}")
        
        return session_id
        
    async def _monitor_session(self, session_id: str) -> None:
        """Background monitoring of learning session."""
        monitor_interval = 30  # seconds
        
        while not self.shutdown_event.is_set():
            try:
                if session_id not in self.active_sessions:
                    break
                    
                learner = self.learners[session_id]
                
                # Get current metrics
                if hasattr(learner, 'get_info'):
                    info = learner.get_info()
                    
                    # Check for drift
                    if info.get('drift_detected', False):
                        await self._handle_drift(session_id)
                        
                    # Update metrics
                    metrics = LearningMetrics(
                        session_id=session_id,
                        timestamp=datetime.now(),
                        n_samples_seen=info.get('n_samples_seen', 0),
                        n_updates=info.get('n_updates', 0),
                        current_accuracy=info.get('current_accuracy', 0),
                        memory_usage_mb=info.get('memory_usage', {}).get('managed_mb', 0),
                        processing_rate_hz=0,  # Would calculate from timestamps
                        drift_detected=info.get('drift_detected', False),
                        model_version=info.get('model_version', 1)
                    )
                    
                    await self._record_metrics(session_id, metrics)
                    
                # Periodic checkpoint
                if hasattr(learner, 'state') and learner.state.n_samples_seen % 1000 == 0:
                    await self.checkpoint_model(session_id)
                    
            except Exception as e:
                logger.error(f"Error monitoring session {session_id}: {e}")
                
            await asyncio.sleep(monitor_interval)
            
    async def _handle_drift(self, session_id: str) -> None:
        """Handle detected concept drift."""
        learner = self.learners[session_id]
        
        # Record drift event
        drift_event = DriftEvent(
            session_id=session_id,
            timestamp=datetime.now(),
            sample_index=learner.state.n_samples_seen if hasattr(learner, 'state') else 0,
            drift_type='sudden',  # Would determine from detector
            drift_level=1.0,
            confidence=0.95,
            detector_method='adwin',
            action_taken='adaptive',
            metadata={}
        )
        self.db.add(drift_event)
        self.db.commit()
        
        # Send alert
        alert = DriftDetectionAlert(
            session_id=session_id,
            drift_type=drift_event.drift_type,
            drift_level=drift_event.drift_level,
            confidence=drift_event.confidence,
            timestamp=drift_event.timestamp,
            recommended_action="Increase learning rate and monitor performance"
        )
        
        # Would send to notification system
        logger.warning(f"Drift detected in session {session_id}: {alert}")
        
    async def _record_metric(self, session_id: str, metric_type: str, value: float) -> None:
        """Record streaming metric."""
        metric = StreamingMetrics(
            session_id=session_id,
            metric_type=metric_type,
            value=value,
            window_size=100
        )
        self.db.add(metric)
        self.db.commit()
        
    async def _record_metrics(self, session_id: str, metrics: LearningMetrics) -> None:
        """Record comprehensive metrics."""
        # Would send to monitoring system
        logger.debug(f"Metrics for session {session_id}: {metrics}")
        
    async def _record_adaptation(self, session_id: str, strategy: str, 
                                conditions: Dict[str, float]) -> None:
        """Record strategy adaptation."""
        # Would record to database
        logger.info(f"Adapted strategy for session {session_id} to {strategy}")
        
    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop an online learning session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        # Update session record
        session = self.active_sessions[session_id]
        session.end_time = datetime.now()
        
        # Get final performance
        learner = self.learners[session_id]
        if hasattr(learner, 'state'):
            session.final_performance = learner.state.current_accuracy
            
        self.db.commit()
        
        # Save final checkpoint
        checkpoint_path = await self.checkpoint_model(session_id, is_best=True)
        
        # Clean up
        del self.active_sessions[session_id]
        del self.learners[session_id]
        del self.validators[session_id]
        del self.controllers[session_id]
        
        logger.info(f"Stopped online learning session {session_id}")
        
        return {
            'session_id': session_id,
            'duration_seconds': (session.end_time - session.start_time).total_seconds(),
            'n_samples_processed': session.n_samples_processed,
            'final_performance': session.final_performance,
            'checkpoint_path': checkpoint_path
        }
        
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a learning session."""
        if session_id not in self.active_sessions:
            # Check database for completed sessions
            session = self.db.query(OnlineLearningSession).filter_by(
                session_id=session_id
            ).first()
            
            if session:
                return {
                    'session_id': session_id,
                    'status': 'completed' if session.end_time else 'inactive',
                    'start_time': session.start_time,
                    'end_time': session.end_time,
                    'n_samples_processed': session.n_samples_processed,
                    'final_performance': session.final_performance
                }
            else:
                raise ValueError(f"Session {session_id} not found")
                
        # Active session
        session = self.active_sessions[session_id]
        learner = self.learners[session_id]
        
        status = {
            'session_id': session_id,
            'status': 'active',
            'start_time': session.start_time,
            'n_samples_processed': session.n_samples_processed
        }
        
        if hasattr(learner, 'get_info'):
            status.update(learner.get_info())
            
        return status
        
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get status of all sessions."""
        sessions = []
        
        # Active sessions
        for session_id in self.active_sessions:
            try:
                status = await self.get_session_status(session_id)
                sessions.append(status)
            except Exception as e:
                logger.error(f"Error getting status for session {session_id}: {e}")
                
        # Recent completed sessions
        recent_completed = self.db.query(OnlineLearningSession).filter(
            OnlineLearningSession.end_time.isnot(None),
            OnlineLearningSession.end_time > datetime.now() - timedelta(hours=24)
        ).limit(10).all()
        
        for session in recent_completed:
            sessions.append({
                'session_id': session.session_id,
                'status': 'completed',
                'start_time': session.start_time,
                'end_time': session.end_time,
                'n_samples_processed': session.n_samples_processed,
                'final_performance': session.final_performance
            })
            
        return sessions
        
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down online learning service...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Save all active sessions
        for session_id in list(self.active_sessions.keys()):
            try:
                await self.stop_session(session_id)
            except Exception as e:
                logger.error(f"Error stopping session {session_id}: {e}")
                
        # Wait for background tasks
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        logger.info("Online learning service shutdown complete")