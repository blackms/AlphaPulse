"""
Basic import tests for CI/CD validation.
"""
import pytest


def test_alpha_pulse_import():
    """Test that alpha_pulse package can be imported."""
    import alpha_pulse
    assert hasattr(alpha_pulse, '__version__')
    print(f"AlphaPulse version: {alpha_pulse.__version__}")


def test_core_modules_import():
    """Test that core modules can be imported."""
    try:
        # Test config module
        from alpha_pulse import config
        assert config is not None
        print("✓ config module imported")
        
        # Test data_pipeline module  
        from alpha_pulse import data_pipeline
        assert data_pipeline is not None
        print("✓ data_pipeline module imported")
        
        # Test features module
        from alpha_pulse import features
        assert features is not None
        print("✓ features module imported")
        
        # Test exchanges module
        from alpha_pulse import exchanges
        assert exchanges is not None
        print("✓ exchanges module imported")
        
    except ImportError as e:
        pytest.fail(f"Failed to import core module: {e}")


def test_exchanges_submodules():
    """Test that exchange submodules can be imported."""
    from alpha_pulse.exchanges import ExchangeFactory, ExchangeType
    assert ExchangeFactory is not None
    assert ExchangeType is not None


def test_models_module():
    """Test that models module can be imported."""
    from alpha_pulse.models import ModelTrainer
    assert ModelTrainer is not None


def test_environment_variables():
    """Test that required environment variables are set."""
    import os
    
    # Check test environment variables
    assert os.getenv('DATABASE_URL') is not None
    assert os.getenv('REDIS_URL') is not None
    assert os.getenv('ENVIRONMENT') == 'test'
    print("Environment variables verified")


def test_database_connection():
    """Test that database URL is properly configured."""
    import os
    database_url = os.getenv('DATABASE_URL')
    assert database_url.startswith('postgresql://')
    assert 'alphapulse_test' in database_url
    print(f"Database URL: {database_url}")


if __name__ == "__main__":
    # Run tests when executed directly
    test_alpha_pulse_import()
    test_core_modules_import()
    test_exchanges_submodules()
    test_models_module()
    test_environment_variables()
    test_database_connection()
    print("All basic import tests passed!")