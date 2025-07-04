"""Demo of online learning capabilities for trading agents."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from alpha_pulse.ml.online import (
    OnlineDataPoint,
    IncrementalSGD,
    AdaptiveRandomForest,
    OnlineLearnerEnsemble,
    AdaptiveLearningRateScheduler,
    ConceptDriftDetector,
    StreamingValidator,
    PrequentialEvaluator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_market_data_stream(n_samples: int = 1000, drift_point: int = 500) -> List[Dict[str, Any]]:
    """Generate synthetic market data with concept drift."""
    np.random.seed(42)
    data_stream = []
    
    for i in range(n_samples):
        timestamp = datetime.now() + timedelta(minutes=i)
        
        # Market features
        features = {
            'price_momentum': np.random.randn(),
            'volume_ratio': np.random.exponential(1),
            'volatility': np.random.gamma(2, 2),
            'rsi': np.random.uniform(20, 80),
            'macd_signal': np.random.randn(),
            'order_flow_imbalance': np.random.randn()
        }
        
        # Target: future return
        if i < drift_point:
            # Regime 1: Momentum-driven market
            target = (
                0.5 * features['price_momentum'] +
                0.3 * features['macd_signal'] +
                0.1 * np.random.randn()
            )
        else:
            # Regime 2: Mean-reversion market (concept drift)
            target = (
                -0.4 * features['price_momentum'] +
                0.5 * features['volatility'] +
                0.1 * np.random.randn()
            )
            
        data_stream.append({
            'timestamp': timestamp,
            'features': features,
            'target': target
        })
        
    return data_stream


def demo_incremental_learning():
    """Demonstrate basic incremental learning."""
    logger.info("=== Demo: Incremental Learning ===")
    
    # Configure learner
    config = {
        'learning_rate': 0.01,
        'penalty': 'l2',
        'alpha': 0.0001,
        'drift_detection': {
            'method': 'adwin',
            'delta': 0.002
        }
    }
    
    learner = IncrementalSGD(config, task_type='regression')
    evaluator = PrequentialEvaluator({'task_type': 'regression', 'window_size': 100})
    
    # Generate data stream
    data_stream = generate_market_data_stream(n_samples=1000)
    
    # Process stream
    predictions = []
    actuals = []
    timestamps = []
    drift_points = []
    
    for i, data in enumerate(data_stream):
        # Create data point
        features = np.array(list(data['features'].values()))
        dp = OnlineDataPoint(
            timestamp=data['timestamp'],
            features=features,
            label=data['target']
        )
        
        # Learn and predict
        prediction = learner.learn_one(dp)
        
        if prediction is not None:
            predictions.append(prediction)
            actuals.append(data['target'])
            timestamps.append(data['timestamp'])
            
            # Update evaluator
            evaluator.add_result(prediction, data['target'])
            
            # Check for drift
            if learner.drift_detector.detected_change():
                drift_points.append(i)
                logger.info(f"Drift detected at sample {i}")
                
        # Log progress
        if i % 100 == 0 and i > 0:
            metrics = evaluator.get_current_performance()
            mse = metrics['metrics'].get('mse', {}).get('current', 0)
            logger.info(f"Sample {i}: MSE = {mse:.4f}")
            
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Predictions vs Actuals
    plt.subplot(2, 1, 1)
    plt.plot(predictions[-200:], label='Predictions', alpha=0.7)
    plt.plot(actuals[-200:], label='Actuals', alpha=0.7)
    for dp in drift_points:
        if dp > len(predictions) - 200:
            plt.axvline(x=dp - (len(predictions) - 200), color='r', 
                       linestyle='--', label='Drift' if dp == drift_points[0] else '')
    plt.legend()
    plt.title('Online Learning: Predictions vs Actuals (Last 200 samples)')
    plt.ylabel('Return')
    
    # Subplot 2: Rolling Error
    window = 50
    errors = np.abs(np.array(predictions) - np.array(actuals))
    rolling_error = pd.Series(errors).rolling(window).mean()
    
    plt.subplot(2, 1, 2)
    plt.plot(rolling_error, label=f'Rolling MAE (window={window})')
    for dp in drift_points:
        plt.axvline(x=dp, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title('Rolling Mean Absolute Error')
    plt.xlabel('Sample')
    plt.ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig('online_learning_demo.png')
    logger.info("Saved plot to online_learning_demo.png")
    
    # Final metrics
    final_metrics = evaluator.get_current_performance()
    logger.info("\nFinal Performance Metrics:")
    for metric, info in final_metrics['metrics'].items():
        logger.info(f"  {metric}: {info['current']:.4f} (trend: {info['trend']})")


def demo_adaptive_ensemble():
    """Demonstrate adaptive ensemble learning."""
    logger.info("\n=== Demo: Adaptive Ensemble Learning ===")
    
    # Configure ensemble
    ensemble_config = {
        'max_models': 5,
        'combination_method': 'weighted_average',
        'max_workers': 4
    }
    
    ensemble = OnlineLearnerEnsemble(ensemble_config)
    
    # Add diverse learners
    learners = [
        ('SGD', IncrementalSGD({'learning_rate': 0.01}, task_type='regression')),
        ('SGD_aggressive', IncrementalSGD({'learning_rate': 0.1}, task_type='regression')),
        ('Passive_Aggressive', IncrementalSGD({
            'loss': 'epsilon_insensitive',
            'learning_rate': 'optimal'
        }, task_type='regression')),
        ('Adaptive_Forest', AdaptiveRandomForest({
            'n_estimators': 3,
            'drift_detection': True
        }))
    ]
    
    for name, learner in learners:
        ensemble.add_learner(learner)
        logger.info(f"Added {name} to ensemble")
        
    # Generate data stream
    data_stream = generate_market_data_stream(n_samples=800, drift_point=400)
    
    # Process stream
    ensemble_predictions = []
    individual_predictions = {i: [] for i in range(len(learners))}
    
    for i, data in enumerate(data_stream):
        features = np.array(list(data['features'].values()))
        dp = OnlineDataPoint(
            timestamp=data['timestamp'],
            features=features,
            label=data['target']
        )
        
        # Ensemble prediction
        pred = ensemble.learn_one(dp)
        if pred is not None:
            ensemble_predictions.append(pred)
            
        # Track individual learner performance
        for j, (name, learner) in enumerate(learners):
            if hasattr(learner, 'state') and learner.state.n_samples_seen > 0:
                try:
                    ind_pred = learner.predict(features.reshape(1, -1))[0]
                    individual_predictions[j].append(ind_pred)
                except:
                    individual_predictions[j].append(np.nan)
                    
        # Update weights periodically
        if i % 100 == 0 and i > 0:
            ensemble.update_weights()
            logger.info(f"Sample {i}: Updated ensemble weights = {ensemble.learner_weights}")
            
    # Check for drift in ensemble
    drift_indices = ensemble.detect_drift()
    if drift_indices:
        logger.info(f"Drift detected in learners: {drift_indices}")
        
    # Get ensemble info
    info = ensemble.get_ensemble_info()
    logger.info(f"\nEnsemble Summary:")
    logger.info(f"  Number of learners: {info['n_learners']}")
    logger.info(f"  Best learner index: {info['best_learner_idx']}")
    
    for i, learner_info in enumerate(info['learners']):
        logger.info(f"\n  Learner {i}:")
        logger.info(f"    Weight: {learner_info['weight']:.3f}")
        logger.info(f"    Samples seen: {learner_info['n_samples_seen']}")
        logger.info(f"    Accuracy: {learner_info['current_accuracy']:.3f}")


def demo_adaptive_learning_rate():
    """Demonstrate adaptive learning rate scheduling."""
    logger.info("\n=== Demo: Adaptive Learning Rate Scheduling ===")
    
    # Different scheduling strategies
    schedules = {
        'exponential_decay': {
            'schedule_type': 'exponential_decay',
            'initial_rate': 0.1,
            'decay_rate': 0.95,
            'decay_steps': 100
        },
        'cosine_annealing': {
            'schedule_type': 'cosine_annealing',
            'initial_rate': 0.1,
            'T_max': 200,
            'eta_min': 0.001
        },
        'adaptive': {
            'schedule_type': 'adaptive',
            'initial_rate': 0.1,
            'adapt_to_performance': True,
            'adapt_to_volatility': True
        }
    }
    
    plt.figure(figsize=(12, 6))
    
    for i, (name, config) in enumerate(schedules.items()):
        scheduler = AdaptiveLearningRateScheduler(config)
        
        rates = []
        for step in range(500):
            # Simulate performance for adaptive scheduler
            if name == 'adaptive':
                performance = 0.8 + 0.1 * np.sin(step / 50)
                rate = scheduler.step(performance)
            else:
                rate = scheduler.step()
            rates.append(rate)
            
        plt.plot(rates, label=name)
        
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Adaptive Learning Rate Schedules')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_schedules.png')
    logger.info("Saved learning rate schedules plot")


def demo_concept_drift_detection():
    """Demonstrate various concept drift detection methods."""
    logger.info("\n=== Demo: Concept Drift Detection ===")
    
    # Test different drift detectors
    detectors = {
        'ADWIN': ConceptDriftDetector({'method': 'adwin', 'delta': 0.002}),
        'DDM': ConceptDriftDetector({'method': 'ddm', 'warning_level': 2.0}),
        'Page-Hinkley': ConceptDriftDetector({'method': 'page_hinkley', 'threshold': 50}),
        'KSWIN': ConceptDriftDetector({'method': 'kswin', 'window_size': 100})
    }
    
    # Generate data with multiple drift points
    np.random.seed(42)
    n_samples = 1000
    
    # Create error signal with drifts
    errors = []
    for i in range(n_samples):
        if i < 300:
            error = np.random.normal(0.1, 0.05)  # Low error
        elif i < 600:
            error = np.random.normal(0.3, 0.1)   # Medium error (drift 1)
        else:
            error = np.random.normal(0.15, 0.05) # Low-medium error (drift 2)
        errors.append(max(0, error))
        
    # Detect drifts
    drift_results = {name: [] for name in detectors}
    
    for i, error in enumerate(errors):
        for name, detector in detectors.items():
            detector.add_element(error)
            
            if detector.detected_change():
                drift_results[name].append(i)
                detector.reset()  # Reset after detection
                
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot error signal
    plt.subplot(2, 1, 1)
    plt.plot(errors, alpha=0.7)
    plt.axvline(x=300, color='r', linestyle='--', label='True drift 1')
    plt.axvline(x=600, color='r', linestyle='--', label='True drift 2')
    plt.ylabel('Error')
    plt.title('Error Signal with Concept Drifts')
    plt.legend()
    
    # Plot detections
    plt.subplot(2, 1, 2)
    y_positions = list(range(len(detectors)))
    
    for i, (name, detections) in enumerate(drift_results.items()):
        plt.scatter(detections, [i] * len(detections), label=name, s=100)
        logger.info(f"{name} detected drifts at: {detections}")
        
    plt.axvline(x=300, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=600, color='r', linestyle='--', alpha=0.5)
    plt.yticks(y_positions, list(detectors.keys()))
    plt.xlabel('Sample')
    plt.title('Drift Detection Results')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drift_detection_comparison.png')
    logger.info("Saved drift detection comparison plot")


async def demo_streaming_validation():
    """Demonstrate streaming validation and monitoring."""
    logger.info("\n=== Demo: Streaming Validation ===")
    
    # Configure validator
    validator_config = {
        'validation_interval': 50,
        'stability_threshold': 0.05,
        'task_type': 'regression',
        'anomaly_detection': {
            'method': 'zscore',
            'threshold': 3.0
        }
    }
    
    validator = StreamingValidator(validator_config)
    
    # Generate predictions with occasional anomalies
    np.random.seed(42)
    n_samples = 500
    
    predictions = []
    actuals = []
    
    for i in range(n_samples):
        # Normal predictions
        if i % 50 == 0 and i > 0:  # Inject anomaly
            pred = np.random.uniform(-5, 5)
            actual = np.random.uniform(-5, 5)
        else:
            actual = np.random.randn()
            pred = actual + np.random.randn() * 0.2
            
        predictions.append(pred)
        actuals.append(actual)
        
    # Validate in batches
    batch_size = 50
    validation_results = []
    
    for i in range(0, n_samples, batch_size):
        batch_preds = np.array(predictions[i:i+batch_size])
        batch_actuals = np.array(actuals[i:i+batch_size])
        
        result = validator.validate_stream('model_1', batch_preds, batch_actuals)
        validation_results.append(result)
        
        # Log results
        logger.info(f"\nBatch {i//batch_size + 1}:")
        logger.info(f"  Is stable: {result['is_stable']}")
        logger.info(f"  Anomalies detected: {len(result['anomalies'])}")
        
        if result['anomalies']:
            logger.info(f"  Anomaly indices: {[a['index'] + i for a in result['anomalies']]}")
            
    # Get final report
    report = validator.get_validation_report('model_1')
    if report:
        logger.info("\nFinal Validation Report:")
        logger.info(f"  Stability: {report['stability']['overall_stable']}")
        
        # Show recommendations
        if report['recommendations']:
            logger.info("  Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"    - {rec}")


def main():
    """Run all demos."""
    logger.info("Starting Online Learning Demos")
    logger.info("=" * 50)
    
    # Run demos
    demo_incremental_learning()
    demo_adaptive_ensemble()
    demo_adaptive_learning_rate()
    demo_concept_drift_detection()
    
    # Run async demo
    asyncio.run(demo_streaming_validation())
    
    logger.info("\n" + "=" * 50)
    logger.info("Online Learning Demos Complete!")
    logger.info("\nKey Capabilities Demonstrated:")
    logger.info("1. Incremental learning with real-time adaptation")
    logger.info("2. Concept drift detection and handling")
    logger.info("3. Adaptive ensemble learning")
    logger.info("4. Dynamic learning rate scheduling")
    logger.info("5. Streaming validation and anomaly detection")
    logger.info("\nCheck generated plots for visualizations.")


if __name__ == "__main__":
    main()