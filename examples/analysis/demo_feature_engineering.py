"""
Demo Feature Engineering and Model Training

This script demonstrates the ML pipeline using the refactored modules:
1. Generate sample price data
2. Feature engineering using technical indicators
3. Model training and evaluation
4. Results visualization

The demo showcases the usage of:
- Data generation module for synthetic data creation
- Feature engineering module for technical analysis
- Model training module for ML model management
- Visualization module for analysis and evaluation
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from alpha_pulse.features.data_generation import (
    create_sample_data,
    create_target_variable
)
from alpha_pulse.features.feature_engineering import (
    calculate_technical_indicators,
    FeatureStore
)
from alpha_pulse.features.visualization import FeatureVisualizer
from alpha_pulse.features.model_training import ModelTrainer, ModelFactory


def main():
    """Run feature engineering and model training demonstration."""
    logger.info("Starting feature engineering demonstration...")

    # Initialize components
    feature_store = FeatureStore()
    visualizer = FeatureVisualizer()
    
    try:
        # 1. Generate sample data
        df = create_sample_data(days=365)
        logger.info(f"Generated {len(df)} days of sample data")
        
        # 2. Feature Engineering
        features = calculate_technical_indicators(df)
        logger.info(f"Generated {len(features.columns)} features")
        
        # Cache computed features
        feature_store.add_features('demo_features', features)
        logger.info("Cached features for future use")
        
        # 3. Create target variable (next day returns)
        target = create_target_variable(df, forward_returns_days=1)
        
        # 4. Initialize model trainer
        trainer = ModelTrainer()
        
        # Prepare and split data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            features.fillna(0),  # Fill NaN values for demonstration
            target.fillna(0)     # Fill NaN values for demonstration
        )
        logger.info(f"Training data size: {len(X_train)} samples")
        logger.info(f"Test data size: {len(X_test)} samples")
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate(X_test, y_test)
        
        # Get and display feature importance
        importance = trainer.get_feature_importance()
        logger.info("\nTop 10 Most Important Features:")
        logger.info(importance.nlargest(10))
        
        # 5. Visualizations
        # Plot feature importance
        visualizer.plot_feature_importance(
            importance,
            title="Feature Importance Scores"
        )
        
        # Plot predictions vs actual
        predictions = trainer.model.predict(X_test)
        visualizer.plot_predictions_vs_actual(
            y_test,
            predictions,
            title="Predicted vs Actual Returns"
        )
        
        # Plot feature distributions
        visualizer.plot_feature_distributions(features)
        
        # Plot correlation matrix
        visualizer.plot_correlation_matrix(features)
        
        # Save trained model
        model_path = trainer.save_model('demo_model')
        logger.info(f"Saved trained model to {model_path}")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()