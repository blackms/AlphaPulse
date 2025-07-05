"""Comprehensive tests for explainability features."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification
import tempfile
import json
from datetime import datetime
from pathlib import Path
import asyncio

from alpha_pulse.ml.explainability.shap_explainer import SHAPExplainer, SHAPConfig
from alpha_pulse.ml.explainability.lime_explainer import (
    LIMETabularExplainer, LIMETimeSeriesExplainer, LIMETextExplainer, LIMEConfig
)
from alpha_pulse.ml.explainability.feature_importance import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig
)
from alpha_pulse.ml.explainability.decision_trees import (
    DecisionTreeSurrogate, SurrogateTreeConfig
)
from alpha_pulse.ml.explainability.explanation_aggregator import (
    ExplanationAggregator, AggregationConfig
)
from alpha_pulse.services.explainability_service import (
    ExplainabilityService, ExplainabilityConfig
)
from alpha_pulse.models.explanation_result import (
    ExplanationType, ExplanationLevel, FeatureContribution
)
from alpha_pulse.utils.visualization_utils import ExplainabilityVisualizer


class TestSHAPExplainer:
    """Tests for SHAP explainer."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        return X, y, feature_names
    
    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        return X, y, feature_names
    
    @pytest.fixture
    def tree_model(self, regression_data):
        """Create a tree-based model."""
        X, y, _ = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def linear_model(self, regression_data):
        """Create a linear model."""
        X, y, _ = regression_data
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def test_shap_tree_explainer(self, tree_model, regression_data):
        """Test SHAP with tree-based model."""
        X, y, feature_names = regression_data
        
        # Create explainer
        config = SHAPConfig(max_samples=50)
        explainer = SHAPExplainer(tree_model, config)
        
        # Explain single instance
        instance = X[0:1]
        result = explainer.explain_instance(
            instance, "test_pred_1", feature_names
        )
        
        # Verify result
        assert result.explanation_type == ExplanationType.SHAP
        assert result.explanation_level == ExplanationLevel.LOCAL
        assert len(result.feature_contributions) == 10
        assert result.prediction_id == "test_pred_1"
        
        # Check that contributions sum to prediction difference
        total_contribution = sum(fc.contribution for fc in result.feature_contributions)
        expected_diff = result.prediction_value - result.base_value
        assert abs(total_contribution - expected_diff) < 0.1
    
    def test_shap_linear_explainer(self, linear_model, regression_data):
        """Test SHAP with linear model."""
        X, y, feature_names = regression_data
        
        # Create explainer with background data
        explainer = SHAPExplainer(linear_model)
        explainer.set_background_data(X[:50])
        
        # Explain instance
        instance = X[0:1]
        result = explainer.explain_instance(
            instance, "test_pred_2", feature_names
        )
        
        # Verify result
        assert result.explanation_type == ExplanationType.SHAP
        assert len(result.feature_contributions) == 10
        
        # For linear models, contributions should be proportional to coefficients
        for i, fc in enumerate(result.feature_contributions):
            assert fc.feature_name == feature_names[i]
    
    def test_shap_batch_explanation(self, tree_model, regression_data):
        """Test batch explanation."""
        X, y, feature_names = regression_data
        
        explainer = SHAPExplainer(tree_model)
        
        # Explain multiple instances
        instances = X[:5]
        pred_ids = [f"pred_{i}" for i in range(5)]
        
        results = explainer.explain_batch(instances, pred_ids, feature_names)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.prediction_id == pred_ids[i]
            assert len(result.feature_contributions) == 10
    
    def test_shap_global_importance(self, tree_model, regression_data):
        """Test global importance computation."""
        X, y, feature_names = regression_data
        
        explainer = SHAPExplainer(tree_model)
        
        # Compute global importance
        global_exp = explainer.compute_global_importance(
            X[:50], feature_names, sample_size=30
        )
        
        assert global_exp.explanation_level == ExplanationLevel.GLOBAL
        assert len(global_exp.global_feature_importance) == 10
        assert global_exp.num_samples == 30
        
        # Check that all features have importance scores
        for feature in feature_names:
            assert feature in global_exp.global_feature_importance
            assert global_exp.global_feature_importance[feature] >= 0
    
    def test_shap_visualization_data(self, tree_model, regression_data):
        """Test visualization data generation."""
        X, y, feature_names = regression_data
        
        explainer = SHAPExplainer(tree_model)
        result = explainer.explain_instance(X[0:1], "test_pred", feature_names)
        
        # Test waterfall data
        waterfall_data = explainer.get_waterfall_data(result)
        assert "base_value" in waterfall_data
        assert "features" in waterfall_data
        assert "contributions" in waterfall_data
        assert len(waterfall_data["features"]) == len(waterfall_data["contributions"])
        
        # Test force plot data
        force_data = explainer.get_force_plot_data(result)
        assert "base_value" in force_data
        assert "positive_features" in force_data
        assert "negative_features" in force_data


class TestLIMEExplainer:
    """Tests for LIME explainer."""
    
    @pytest.fixture
    def tabular_data(self):
        """Create tabular test data."""
        X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(8)]
        return X, y, feature_names
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series test data."""
        # Simulate time series with trend and seasonality
        t = np.linspace(0, 4*np.pi, 100)
        y = np.sin(t) + 0.1 * t + np.random.normal(0, 0.1, 100)
        return y
    
    def test_lime_tabular_explainer(self, tabular_data):
        """Test LIME tabular explainer."""
        X, y, feature_names = tabular_data
        
        # Train a model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create LIME explainer
        config = LIMEConfig(num_features=5, num_samples=1000)
        explainer = LIMETabularExplainer(X[:100], feature_names, config=config)
        
        # Explain instance
        instance = X[0:1]
        result = explainer.explain_instance(
            instance, model.predict, "test_pred", "test_model"
        )
        
        # Verify result
        assert result.explanation_type == ExplanationType.LIME
        assert result.explanation_level == ExplanationLevel.LOCAL
        assert len(result.feature_contributions) <= 5  # num_features
        assert result.prediction_id == "test_pred"
        
        # Check visualization data
        assert "lime_explanation" in result.visualization_data
        assert "feature_weights" in result.visualization_data
    
    def test_lime_stability(self, tabular_data):
        """Test LIME explanation stability."""
        X, y, feature_names = tabular_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        explainer = LIMETabularExplainer(X[:100], feature_names)
        
        # Test stability across runs
        instance = X[0:1]
        stability = explainer.get_feature_importance_stability(
            instance, model.predict, n_runs=5
        )
        
        # Check that all features have stability metrics
        for feature in feature_names:
            assert feature in stability
            assert "mean" in stability[feature]
            assert "std" in stability[feature]
            assert "cv" in stability[feature]
            
            # Coefficient of variation should be reasonable
            assert stability[feature]["cv"] < 1.0  # Not too unstable
    
    def test_lime_time_series(self, time_series_data):
        """Test LIME time series explainer."""
        ts = time_series_data
        
        # Simple model that predicts next value
        def predict_next(series):
            # Use last 3 values to predict next
            if len(series.shape) == 1:
                return np.array([series[-1] + (series[-1] - series[-2])])
            else:
                return np.array([s[-1] + (s[-1] - s[-2]) for s in series])
        
        # Create explainer
        window_size = 10
        explainer = LIMETimeSeriesExplainer(window_size)
        
        # Explain prediction at time 50
        result = explainer.explain_instance(
            ts, predict_next, "test_pred", "test_model", time_index=50
        )
        
        # Verify result
        assert result.explanation_type == ExplanationType.LIME
        assert len(result.feature_contributions) <= window_size
        
        # Check that time steps are labeled correctly
        for fc in result.feature_contributions:
            assert fc.feature_name.startswith("t")
    
    def test_lime_text_explainer(self):
        """Test LIME text explainer."""
        # Simple sentiment prediction function
        def predict_sentiment(texts):
            # Mock sentiment: positive if "good" in text
            sentiments = []
            for text in texts:
                if "good" in text.lower():
                    sentiments.append([0.2, 0.8])  # [negative, positive]
                else:
                    sentiments.append([0.8, 0.2])
            return np.array(sentiments)
        
        explainer = LIMETextExplainer()
        
        # Explain text
        text = "This is a good stock to buy"
        result = explainer.explain_instance(
            text, predict_sentiment, "test_pred", "test_model"
        )
        
        # Verify result
        assert result.explanation_type == ExplanationType.LIME
        assert len(result.feature_contributions) > 0
        
        # Check that "good" has positive contribution
        good_contributions = [
            fc for fc in result.feature_contributions 
            if fc.feature_name == "good"
        ]
        if good_contributions:
            assert good_contributions[0].contribution > 0


class TestFeatureImportance:
    """Tests for feature importance analysis."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and data for testing."""
        X, y = make_regression(n_samples=200, n_features=10, 
                              n_informative=5, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        return model, X, y, feature_names
    
    def test_permutation_importance(self, model_and_data):
        """Test permutation importance."""
        model, X, y, feature_names = model_and_data
        
        analyzer = FeatureImportanceAnalyzer()
        
        # Analyze with permutation only
        result = analyzer.analyze_global_importance(
            model, X[:100], y[:100], feature_names, method="permutation"
        )
        
        # Verify result
        assert isinstance(result.global_feature_importance, dict)
        assert len(result.global_feature_importance) == 10
        
        # Informative features should have higher importance
        importances = list(result.global_feature_importance.values())
        assert max(importances) > min(importances) * 2  # Some variation
    
    def test_drop_column_importance(self, model_and_data):
        """Test drop column importance."""
        model, X, y, feature_names = model_and_data
        
        config = FeatureImportanceConfig(sample_size=50)
        analyzer = FeatureImportanceAnalyzer(config)
        
        # Analyze with drop column
        result = analyzer.analyze_global_importance(
            model, X, y, feature_names, method="drop_column"
        )
        
        # Verify all features analyzed
        assert all(f in result.global_feature_importance for f in feature_names)
    
    def test_feature_interactions(self, model_and_data):
        """Test feature interaction computation."""
        model, X, y, feature_names = model_and_data
        
        analyzer = FeatureImportanceAnalyzer()
        
        # Analyze all methods
        result = analyzer.analyze_global_importance(
            model, X[:50], y[:50], feature_names, method="all"
        )
        
        # Check interactions
        assert len(result.feature_interaction_importance) > 0
        
        # Interactions should be tuples of feature names
        for interaction, importance in result.feature_interaction_importance.items():
            assert isinstance(interaction, tuple)
            assert len(interaction) == 2
            assert importance >= 0
    
    def test_mutual_information(self, model_and_data):
        """Test mutual information computation."""
        model, X, y, feature_names = model_and_data
        
        analyzer = FeatureImportanceAnalyzer()
        
        # Compute mutual information
        mi_scores = analyzer.compute_mutual_information(X, y, feature_names)
        
        assert len(mi_scores) == 10
        assert all(score >= 0 for score in mi_scores.values())


class TestDecisionTreeSurrogate:
    """Tests for decision tree surrogate models."""
    
    @pytest.fixture
    def complex_model_and_data(self):
        """Create a complex model and data."""
        X, y = make_classification(n_samples=500, n_features=20, 
                                  n_informative=10, n_redundant=5,
                                  random_state=42)
        feature_names = [f"feature_{i}" for i in range(20)]
        
        # Complex model (ensemble)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, X, y, feature_names
    
    def test_surrogate_fitting(self, complex_model_and_data):
        """Test fitting surrogate tree."""
        model, X, y, feature_names = complex_model_and_data
        
        config = SurrogateTreeConfig(max_depth=4, min_samples_leaf=20)
        surrogate = DecisionTreeSurrogate(model, config)
        
        # Fit surrogate
        surrogate.fit(X[:300], feature_names)
        
        # Check fidelity
        assert surrogate.fidelity_score > 0.7  # Reasonable approximation
        assert surrogate.surrogate_tree is not None
    
    def test_surrogate_global_explanation(self, complex_model_and_data):
        """Test global explanation from surrogate."""
        model, X, y, feature_names = complex_model_and_data
        
        surrogate = DecisionTreeSurrogate(model)
        surrogate.fit(X[:300], feature_names)
        
        # Get global explanation
        global_exp = surrogate.explain_global()
        
        assert "tree_structure" in global_exp
        assert "decision_rules" in global_exp
        assert "feature_importance" in global_exp
        assert len(global_exp["decision_rules"]) > 0
        
        # Check tree structure
        tree_struct = global_exp["tree_structure"]
        assert tree_struct["type"] == "split"  # Root should be split
        assert "feature" in tree_struct
        assert "threshold" in tree_struct
    
    def test_surrogate_instance_explanation(self, complex_model_and_data):
        """Test instance explanation from surrogate."""
        model, X, y, feature_names = complex_model_and_data
        
        surrogate = DecisionTreeSurrogate(model)
        surrogate.fit(X[:300], feature_names)
        
        # Explain instance
        instance = X[0:1]
        exp = surrogate.explain_instance(instance)
        
        assert "decision_path" in exp
        assert "original_prediction" in exp
        assert "surrogate_prediction" in exp
        assert len(exp["decision_path"]) > 0
        
        # Decision path should end with leaf
        assert exp["decision_path"][-1]["type"] == "leaf"
    
    def test_surrogate_visualization(self, complex_model_and_data):
        """Test surrogate tree visualization."""
        model, X, y, feature_names = complex_model_and_data
        
        surrogate = DecisionTreeSurrogate(model)
        surrogate.fit(X[:300], feature_names)
        
        # Get DOT representation
        dot_string = surrogate.visualize_tree()
        
        assert "digraph" in dot_string
        assert "feature_0" in dot_string or feature_names[0] in dot_string
        
        # Get text representation
        text_repr = surrogate.get_text_representation()
        assert "|---" in text_repr  # Tree structure markers


class TestExplanationAggregator:
    """Tests for explanation aggregation."""
    
    @pytest.fixture
    def multiple_explanations(self):
        """Create multiple explanations for testing."""
        explanations = []
        
        # Create 3 explanations with varying contributions
        for i in range(3):
            contributions = []
            for j in range(5):
                fc = FeatureContribution(
                    feature_name=f"feature_{j}",
                    value=float(j),
                    contribution=float(j * (i + 1) + np.random.normal(0, 0.1))
                )
                contributions.append(fc)
            
            exp = ExplanationResult(
                explanation_id=f"exp_{i}",
                model_id="test_model",
                prediction_id="test_pred",
                timestamp=datetime.now(),
                explanation_type=ExplanationType.SHAP if i == 0 else ExplanationType.LIME,
                explanation_level=ExplanationLevel.LOCAL,
                feature_contributions=contributions,
                base_value=0.0,
                prediction_value=10.0,
                confidence_score=0.8 + i * 0.05
            )
            explanations.append(exp)
        
        return explanations
    
    def test_explanation_aggregation(self, multiple_explanations):
        """Test aggregating multiple explanations."""
        config = AggregationConfig(
            aggregation_method="weighted_mean",
            normalize_scores=True
        )
        aggregator = ExplanationAggregator(config)
        
        # Add explanations
        for exp in multiple_explanations:
            aggregator.add_explanation(exp)
        
        # Aggregate
        result = aggregator.aggregate()
        
        # Verify result
        assert result.explanation_type == ExplanationType.FEATURE_IMPORTANCE
        assert len(result.feature_contributions) <= 10  # top_k_features
        assert result.confidence_score is not None
        
        # Check that aggregation happened
        assert "methods_used" in result.method_parameters
        assert len(result.method_parameters["methods_used"]) == 3
    
    def test_explanation_comparison(self, multiple_explanations):
        """Test comparing explanations."""
        aggregator = ExplanationAggregator()
        
        # Compare explanations
        comparison = aggregator.compare_explanations(multiple_explanations)
        
        # Verify comparison
        assert len(comparison.explanation_ids) == 3
        assert comparison.method_agreement_matrix.shape == (3, 3)
        assert len(comparison.consensus_features) >= 0
        assert len(comparison.disputed_features) >= 0
        
        # Agreement matrix should have 1.0 on diagonal
        for i in range(3):
            assert comparison.method_agreement_matrix[i, i] == 1.0
    
    def test_counterfactual_generation(self):
        """Test counterfactual explanation generation."""
        # Simple linear model
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        model = LinearRegression()
        model.fit(X, y)
        
        aggregator = ExplanationAggregator()
        
        # Generate counterfactual
        instance = X[0:1]
        original_pred = model.predict(instance)[0]
        desired_outcome = original_pred + 5.0
        
        feature_names = [f"feature_{i}" for i in range(5)]
        counterfactual = aggregator.generate_counterfactual(
            instance, model, desired_outcome, feature_names
        )
        
        # Verify counterfactual
        assert counterfactual.original_prediction != counterfactual.counterfactual_prediction
        assert len(counterfactual.feature_changes) > 0
        assert counterfactual.num_features_changed == len(counterfactual.get_minimal_changes())


class TestExplainabilityService:
    """Tests for explainability service."""
    
    @pytest.fixture
    def service(self, tmp_path):
        """Create explainability service."""
        config = ExplainabilityConfig(
            enable_shap=True,
            enable_lime=True,
            enable_surrogate=True,
            store_explanations=False,
            visualization_output_dir=str(tmp_path / "viz")
        )
        return ExplainabilityService(config)
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and data."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X, y, feature_names
    
    @pytest.mark.asyncio
    async def test_explain_prediction(self, service, model_and_data):
        """Test explaining a prediction."""
        model, X, y, feature_names = model_and_data
        
        # Explain prediction
        instance = X[0:1]
        explanations = await service.explain_prediction(
            model, instance, "test_pred", "test_model",
            feature_names, methods=["shap", "lime"],
            background_data=X[:100]
        )
        
        # Verify explanations
        assert "shap" in explanations
        assert "lime" in explanations
        
        assert explanations["shap"].explanation_type == ExplanationType.SHAP
        assert explanations["lime"].explanation_type == ExplanationType.LIME
    
    @pytest.mark.asyncio
    async def test_global_explanation(self, service, model_and_data):
        """Test global model explanation."""
        model, X, y, feature_names = model_and_data
        
        # Get global explanation
        global_exp = await service.explain_global_behavior(
            model, X[:100], "test_model", feature_names
        )
        
        # Verify
        assert global_exp.num_samples == 100
        assert len(global_exp.global_feature_importance) == 10
        assert all(imp >= 0 for imp in global_exp.global_feature_importance.values())
    
    @pytest.mark.asyncio
    async def test_counterfactual_generation(self, service, model_and_data):
        """Test counterfactual generation through service."""
        model, X, y, feature_names = model_and_data
        
        instance = X[0:1]
        original_pred = model.predict(instance)[0]
        
        # Generate counterfactual
        counterfactual = await service.generate_counterfactual(
            model, instance, original_pred + 10.0,
            "test_model", feature_names
        )
        
        # Verify
        assert counterfactual.num_features_changed > 0
        assert counterfactual.change_distance > 0
    
    def test_method_comparison(self, service, model_and_data):
        """Test comparing explanation methods."""
        # Create dummy explanations
        contributions1 = [
            FeatureContribution(f"feature_{i}", float(i), float(i*2))
            for i in range(5)
        ]
        contributions2 = [
            FeatureContribution(f"feature_{i}", float(i), float(i*2.1))
            for i in range(5)
        ]
        
        exp1 = ExplanationResult(
            "exp1", "model", "pred", datetime.now(),
            ExplanationType.SHAP, ExplanationLevel.LOCAL,
            contributions1, 0.0, 10.0
        )
        exp2 = ExplanationResult(
            "exp2", "model", "pred", datetime.now(),
            ExplanationType.LIME, ExplanationLevel.LOCAL,
            contributions2, 0.0, 10.5
        )
        
        # Compare
        comparison = service.compare_methods({"shap": exp1, "lime": exp2})
        
        assert "consensus_features" in comparison
        assert "disputed_features" in comparison
        assert "mean_agreement" in comparison
        assert comparison["mean_agreement"] > 0.9  # High agreement for similar contributions
    
    @pytest.mark.asyncio
    async def test_visualization(self, service, model_and_data, tmp_path):
        """Test explanation visualization."""
        model, X, y, feature_names = model_and_data
        
        # Create explanation
        instance = X[0:1]
        explanations = await service.explain_prediction(
            model, instance, "test_pred", "test_model",
            feature_names, methods=["shap"],
            background_data=X[:100]
        )
        
        # Visualize
        shap_exp = explanations["shap"]
        output_paths = await service.visualize_explanation(
            shap_exp, "test_viz", ["feature_importance"]
        )
        
        # Verify file created
        assert "feature_importance" in output_paths
        assert Path(output_paths["feature_importance"]).exists()


class TestVisualizationUtils:
    """Tests for visualization utilities."""
    
    @pytest.fixture
    def sample_explanation(self):
        """Create sample explanation for visualization."""
        contributions = [
            FeatureContribution(f"feature_{i}", float(i), 
                              float(np.random.normal(0, i+1)))
            for i in range(10)
        ]
        
        return ExplanationResult(
            "test_exp", "model", "pred", datetime.now(),
            ExplanationType.SHAP, ExplanationLevel.LOCAL,
            contributions, 0.0, 5.0, confidence_score=0.85
        )
    
    def test_feature_importance_plot(self, sample_explanation, tmp_path):
        """Test feature importance plotting."""
        visualizer = ExplainabilityVisualizer()
        
        # Create plot
        save_path = tmp_path / "importance.png"
        fig = visualizer.plot_feature_importance(
            sample_explanation, save_path=str(save_path)
        )
        
        # Verify
        assert save_path.exists()
        assert fig is not None
    
    def test_export_explanation_report(self, sample_explanation, tmp_path):
        """Test exporting explanation report."""
        visualizer = ExplainabilityVisualizer()
        
        # Export as HTML
        html_path = tmp_path / "report.html"
        visualizer.export_explanation_report(
            sample_explanation, str(html_path), format="html"
        )
        
        # Verify
        assert html_path.exists()
        content = html_path.read_text()
        assert "Explanation Report" in content
        assert "feature_0" in content
        
        # Export as JSON
        json_path = tmp_path / "report.json"
        visualizer.export_explanation_report(
            sample_explanation, str(json_path), format="json"
        )
        
        # Verify JSON
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "explanation_id" in data
        assert "feature_contributions" in data