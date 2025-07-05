# Explainable AI Features

## Overview

AlphaPulse includes comprehensive explainable AI (XAI) capabilities to provide transparency and interpretability for ML-driven trading decisions. This ensures regulatory compliance, builds trust, and helps traders understand the factors driving algorithmic decisions.

## Key Features

### 1. SHAP (SHapley Additive exPlanations)

SHAP provides unified framework for interpreting model predictions based on game theory concepts.

#### Features:
- **TreeExplainer**: Exact SHAP values for tree-based models (XGBoost, Random Forest)
- **DeepExplainer**: Neural network explanations using DeepLIFT
- **KernelExplainer**: Model-agnostic explanations for any model type
- **LinearExplainer**: Efficient explanations for linear models

#### Usage:
```python
from alpha_pulse.ml.explainability import SHAPExplainer, SHAPConfig

# Configure SHAP
config = SHAPConfig(
    max_samples=100,
    check_additivity=True,
    n_jobs=-1
)

# Create explainer
explainer = SHAPExplainer(model, config)
explainer.set_background_data(training_data)

# Explain single prediction
explanation = explainer.explain_instance(
    instance=features,
    prediction_id="trade_123",
    feature_names=feature_names
)

# Get global importance
global_exp = explainer.compute_global_importance(
    data=test_data,
    feature_names=feature_names
)
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)

LIME provides local explanations by approximating the model locally with interpretable models.

#### Features:
- **Tabular Data**: Explanations for structured trading data
- **Time Series**: Temporal explanations for price predictions
- **Text Data**: Explanations for sentiment analysis models

#### Usage:
```python
from alpha_pulse.ml.explainability import LIMETabularExplainer, LIMEConfig

# Configure LIME
config = LIMEConfig(
    num_features=10,
    num_samples=5000,
    mode="regression"
)

# Create explainer
explainer = LIMETabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    config=config
)

# Explain prediction
explanation = explainer.explain_instance(
    instance=features,
    predict_fn=model.predict,
    prediction_id="trade_123",
    model_id="momentum_model"
)
```

### 3. Feature Importance Analysis

Comprehensive feature importance using multiple methods for robustness.

#### Methods:
- **Permutation Importance**: Model-agnostic importance by permuting features
- **Drop Column Importance**: Importance by removing features
- **Model-based Importance**: Native importance from tree models
- **Mutual Information**: Information-theoretic importance

#### Usage:
```python
from alpha_pulse.ml.explainability import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()

# Analyze global importance
global_importance = analyzer.analyze_global_importance(
    model=model,
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    method="all"  # Use all methods
)

# Get feature interactions
interactions = global_importance.feature_interaction_importance
```

### 4. Decision Tree Surrogates

Interpretable decision tree approximations of complex models.

#### Features:
- **Global Interpretability**: Extract decision rules from complex models
- **Fidelity Scoring**: Measure how well surrogate approximates original
- **Instance Explanations**: Trace decision paths for individual predictions
- **Visualization**: Tree diagrams and text representations

#### Usage:
```python
from alpha_pulse.ml.explainability import DecisionTreeSurrogate, SurrogateTreeConfig

# Configure surrogate
config = SurrogateTreeConfig(
    max_depth=5,
    min_samples_leaf=20
)

# Create and fit surrogate
surrogate = DecisionTreeSurrogate(complex_model, config)
surrogate.fit(X_train, feature_names)

# Get global explanation
global_exp = surrogate.explain_global()
print(f"Fidelity: {surrogate.fidelity_score:.3f}")

# Explain instance
instance_exp = surrogate.explain_instance(instance)
decision_path = instance_exp["decision_path"]
```

### 5. Counterfactual Explanations

Show minimal changes needed for different outcomes.

#### Features:
- **What-if Analysis**: Explore alternative scenarios
- **Minimal Changes**: Find smallest modifications needed
- **Constraint Handling**: Respect feature constraints
- **Quality Scoring**: Evaluate counterfactual quality

#### Usage:
```python
# Generate counterfactual
counterfactual = service.generate_counterfactual(
    model=model,
    instance=current_features,
    desired_outcome=target_return,
    feature_constraints={
        "position_size": (0, 1),
        "leverage": (1, 3)
    }
)

# Get minimal changes
changes = counterfactual.get_minimal_changes()
```

## Explainability Service

The unified `ExplainabilityService` integrates all explanation methods.

### Configuration:
```python
from alpha_pulse.services import ExplainabilityService, ExplainabilityConfig

config = ExplainabilityConfig(
    enable_shap=True,
    enable_lime=True,
    enable_surrogate=True,
    enable_counterfactual=True,
    
    # Storage
    store_explanations=True,
    explanation_cache_size=1000,
    
    # Performance
    async_explanations=True,
    max_concurrent_explanations=5,
    
    # Compliance
    require_explanation_for_trades=True,
    min_confidence_threshold=0.7,
    audit_all_decisions=True
)

service = ExplainabilityService(config, db_session)
```

### Multi-Method Explanations:
```python
# Explain with multiple methods
explanations = await service.explain_prediction(
    model=trading_model,
    instance=features,
    prediction_id=f"trade_{timestamp}",
    model_id="momentum_strategy",
    feature_names=feature_names,
    methods=["shap", "lime", "surrogate"],
    background_data=training_data
)

# Aggregate explanations
aggregated = service.aggregate_explanations(explanations)

# Compare methods
comparison = service.compare_methods(explanations)
print(f"Method agreement: {comparison['mean_agreement']:.3f}")
```

## Visualization

Rich visualization support for explanations.

### Available Visualizations:
- **Feature Importance Plots**: Bar, horizontal bar, lollipop charts
- **SHAP Plots**: Waterfall, force plots, summary plots
- **LIME Plots**: Local explanation visualizations
- **Counterfactual Comparisons**: Before/after comparisons
- **Method Comparisons**: Agreement matrices, consistency scores

### Example:
```python
from alpha_pulse.utils import ExplainabilityVisualizer

visualizer = ExplainabilityVisualizer()

# Create visualizations
fig = visualizer.plot_feature_importance(
    explanation,
    top_k=10,
    plot_type="waterfall"
)

# Create interactive dashboard
dashboard = visualizer.create_interactive_dashboard(
    explanations=explanation_history
)

# Export report
visualizer.export_explanation_report(
    explanation,
    output_path="reports/trade_explanation.html",
    format="html",
    include_visualizations=True
)
```

## Regulatory Compliance

### Features:
- **Decision Audit Trails**: Complete record of all decisions
- **Explanation Reports**: Automated regulatory documentation
- **Bias Detection**: Monitor for discriminatory patterns
- **Model Cards**: Comprehensive model documentation

### Compliance Checking:
```python
# Check explanation compliance
compliance = await service.check_explanation_compliance(explanation)

if not compliance["is_compliant"]:
    logger.warning(f"Compliance issues: {compliance['issues']}")
    
# Generate regulatory report
report = service.generate_regulatory_report(
    model_id="momentum_strategy",
    start_date=start,
    end_date=end
)
```

## Integration with Trading System

### Trade Decision Explanations:
```python
# In trading agent
decision = agent.make_trading_decision(market_data)

# Generate explanation
explanation = await explainability_service.explain_prediction(
    model=agent.model,
    instance=decision.features,
    prediction_id=decision.id,
    model_id=agent.model_id
)

# Attach to trade
trade.explanation = explanation
trade.explanation_summary = explanation.get_top_features(5)
```

### Risk Management Integration:
```python
# Explain risk assessment
risk_explanation = await service.explain_prediction(
    model=risk_model,
    instance=portfolio_features,
    prediction_id=f"risk_{timestamp}",
    model_id="var_model"
)

# Use in risk decisions
if risk_explanation.confidence_score < 0.6:
    logger.warning("Low confidence in risk assessment")
    risk_manager.increase_safety_margin()
```

## Performance Considerations

### Optimization Tips:
1. **Caching**: Explanations are cached to avoid recomputation
2. **Async Processing**: Use async methods for concurrent explanations
3. **Sampling**: Use sampling for large datasets
4. **Method Selection**: Choose appropriate methods for model types

### Performance Monitoring:
```python
# Monitor explanation metrics
metrics = {
    "explanation_time": explanation.computation_time,
    "confidence": explanation.confidence_score,
    "cache_hit_rate": service.get_cache_stats()["hit_rate"]
}
```

## Best Practices

### 1. Method Selection
- Use SHAP for tree-based models (exact values)
- Use LIME for quick local explanations
- Use surrogates for extracting rules
- Combine methods for robustness

### 2. Feature Engineering
- Ensure feature names are meaningful
- Group related features for clarity
- Document feature transformations

### 3. Validation
- Validate explanations with domain experts
- Check explanation stability
- Monitor explanation drift over time

### 4. User Communication
- Present top features clearly
- Use appropriate visualizations
- Provide confidence intervals
- Include "what-if" scenarios

## Examples

### Complete Trading Explanation:
```python
async def explain_trading_decision(trade_signal):
    """Generate comprehensive explanation for trading decision."""
    
    # 1. Generate multi-method explanations
    explanations = await explainability_service.explain_prediction(
        model=trade_signal.model,
        instance=trade_signal.features,
        prediction_id=trade_signal.id,
        model_id=trade_signal.strategy_id,
        methods=["shap", "lime"]
    )
    
    # 2. Aggregate for consensus
    aggregated = explainability_service.aggregate_explanations(explanations)
    
    # 3. Generate counterfactual
    counterfactual = await explainability_service.generate_counterfactual(
        model=trade_signal.model,
        instance=trade_signal.features,
        desired_outcome=0.0,  # No trade
        feature_constraints=get_feature_constraints()
    )
    
    # 4. Create visualizations
    viz_paths = await explainability_service.visualize_explanation(
        aggregated,
        output_name=f"trade_{trade_signal.id}",
        plot_types=["feature_importance", "shap_waterfall"]
    )
    
    # 5. Check compliance
    compliance = await explainability_service.check_explanation_compliance(
        aggregated
    )
    
    return {
        "explanation": aggregated,
        "counterfactual": counterfactual,
        "visualizations": viz_paths,
        "compliance": compliance,
        "top_factors": aggregated.get_top_features(5)
    }
```

This comprehensive explainable AI system ensures that AlphaPulse's trading decisions are transparent, interpretable, and compliant with regulatory requirements.