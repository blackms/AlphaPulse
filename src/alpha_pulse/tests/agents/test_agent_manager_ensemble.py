import sys
from types import ModuleType, SimpleNamespace

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


def _noop_decorator(*_args, **_kwargs):
    def wrapper(func):
        return func
    return wrapper


if "alpha_pulse.decorators.audit_decorators" not in sys.modules:
    stub_module = ModuleType("alpha_pulse.decorators.audit_decorators")
    stub_module.audit_trade_decision = _noop_decorator
    stub_module.audit_risk_check = _noop_decorator
    stub_module.audit_portfolio_action = _noop_decorator
    stub_module.audit_agent_signal = _noop_decorator
    stub_module.audit_data_access = _noop_decorator
    stub_module.audit_config_change = _noop_decorator
    stub_module.audit_secret_access = _noop_decorator
    sys.modules["alpha_pulse.decorators.audit_decorators"] = stub_module

if "alpha_pulse.services.regime_detection_service" not in sys.modules:
    regime_module = ModuleType("alpha_pulse.services.regime_detection_service")

    class _StubRegimeDetectionService:
        def __init__(self, *args, **kwargs):
            self.config = kwargs.get("config")

        async def initialize(self):
            return None

        async def start(self):
            return None

        async def stop(self):
            return None

    class _StubRegimeDetectionConfig(dict):
        pass

    regime_module.RegimeDetectionService = _StubRegimeDetectionService
    regime_module.RegimeDetectionConfig = _StubRegimeDetectionConfig
    sys.modules["alpha_pulse.services.regime_detection_service"] = regime_module

if "alpha_pulse.ml.regime.regime_classifier" not in sys.modules:
    classifier_module = ModuleType("alpha_pulse.ml.regime.regime_classifier")

    class _StubRegimeInfo:
        pass

    class _StubRegimeType:
        BULL = "bull"
        BEAR = "bear"
        SIDEWAYS = "sideways"

    classifier_module.RegimeInfo = _StubRegimeInfo
    classifier_module.RegimeType = _StubRegimeType
    sys.modules["alpha_pulse.ml.regime.regime_classifier"] = classifier_module

if "alpha_pulse.agents.factory" not in sys.modules:
    factory_module = ModuleType("alpha_pulse.agents.factory")

    class _StubAgentFactory:
        @staticmethod
        async def create_all_agents(_config):
            return {}

    factory_module.AgentFactory = _StubAgentFactory
    sys.modules["alpha_pulse.agents.factory"] = factory_module

if "alpha_pulse.services.ensemble_service" not in sys.modules:
    ensemble_module = ModuleType("alpha_pulse.services.ensemble_service")

    class _StubEnsembleService:
        def __init__(self, *args, **kwargs):
            pass

    ensemble_module.EnsembleService = _StubEnsembleService
    sys.modules["alpha_pulse.services.ensemble_service"] = ensemble_module

if "alpha_pulse.agents.gpu_signal_processor" not in sys.modules:
    gpu_module = ModuleType("alpha_pulse.agents.gpu_signal_processor")

    class _StubGPUProcessor:
        def __init__(self, *_args, **_kwargs):
            self.enabled = False

        async def process_signals(self, signals):
            return signals

    gpu_module.GPUSignalProcessor = _StubGPUProcessor
    sys.modules["alpha_pulse.agents.gpu_signal_processor"] = gpu_module

if "alpha_pulse.services.explainability_service" not in sys.modules:
    explainability_module = ModuleType("alpha_pulse.services.explainability_service")

    class _StubExplainabilityService:
        def __init__(self, *args, **kwargs):
            pass

    explainability_module.ExplainabilityService = _StubExplainabilityService
    sys.modules["alpha_pulse.services.explainability_service"] = explainability_module

if "alpha_pulse.compliance.explanation_compliance" not in sys.modules:
    compliance_module = ModuleType("alpha_pulse.compliance.explanation_compliance")

    class _StubComplianceRequirement:
        MIFID_II = "MiFID_II"
        SOX = "SOX"

    class _StubComplianceManager:
        def __init__(self):
            pass

        def register_requirement(self, *_args, **_kwargs):
            return None

    compliance_module.ComplianceRequirement = _StubComplianceRequirement
    compliance_module.ExplanationComplianceManager = _StubComplianceManager
    sys.modules["alpha_pulse.compliance.explanation_compliance"] = compliance_module

if "alpha_pulse.data.quality.data_validator" not in sys.modules:
    quality_module = ModuleType("alpha_pulse.data.quality.data_validator")

    class _StubQualityScore:
        def __init__(self):
            self.completeness = 1.0
            self.accuracy = 1.0
            self.consistency = 1.0
            self.timeliness = 1.0
            self.validity = 1.0
            self.uniqueness = 1.0

    class _StubDataValidator:
        async def validate_market_data(self, *_args, **_kwargs):
            return SimpleNamespace(quality_score=_StubQualityScore())

    quality_module.DataValidator = _StubDataValidator
    quality_module.QualityScore = _StubQualityScore
    sys.modules["alpha_pulse.data.quality.data_validator"] = quality_module

if "alpha_pulse.data.quality.quality_metrics" not in sys.modules:
    metrics_module = ModuleType("alpha_pulse.data.quality.quality_metrics")

    def _stub_get_quality_metrics_service(*_args, **_kwargs):
        return SimpleNamespace(record_metrics=lambda *a, **k: None)

    metrics_module.get_quality_metrics_service = _stub_get_quality_metrics_service
    sys.modules["alpha_pulse.data.quality.quality_metrics"] = metrics_module


from alpha_pulse.agents.interfaces import TradeSignal, SignalDirection
from alpha_pulse.agents.manager import AgentManager


@pytest.mark.asyncio
async def test_agent_manager_awaits_ensemble_prediction():
    """Ensure AgentManager awaits ensemble predictions and returns aggregated signals."""
    ensemble_prediction = SimpleNamespace(
        id=1,
        ensemble_id="ensemble-1",
        timestamp=datetime.utcnow(),
        signal="buy",
        confidence=0.8,
        contributing_agents=["agent-123"],
        weights={"agent-123": 1.0},
        metadata={"strategy": "voting"},
        execution_time_ms=12.5
    )

    ensemble_service = MagicMock()
    ensemble_service.get_ensemble_prediction = AsyncMock(return_value=ensemble_prediction)

    manager = AgentManager(
        config={"use_ensemble": True},
        ensemble_service=ensemble_service
    )
    manager.ensemble_id = "ensemble-1"
    manager.agent_registry = {"technical": "agent-123"}

    signals = [
        TradeSignal(
            agent_id="agent-alpha",
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=0.9,
            timestamp=datetime.utcnow(),
            target_price=150.0,
            stop_loss=145.0,
            metadata={
                "agent_type": "technical",
                "target_price": 150.0,
                "stop_loss": 145.0
            }
        )
    ]

    with patch('alpha_pulse.agents.manager.AgentSignalCreate') as MockAgentSignalCreate, \
         patch('alpha_pulse.agents.manager.TradeSignal') as MockTradeSignal:
        MockAgentSignalCreate.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        created_signal = MagicMock()
        MockTradeSignal.return_value = created_signal

        aggregated = await manager._aggregate_signals_with_ensemble(signals)

    ensemble_service.get_ensemble_prediction.assert_awaited_once()
    assert aggregated == [created_signal]

    trade_signal_kwargs = MockTradeSignal.call_args.kwargs
    assert "metadata" in trade_signal_kwargs
    assert trade_signal_kwargs["metadata"].get("ensemble_id") == "ensemble-1"
