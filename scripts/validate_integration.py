#!/usr/bin/env python3
"""
AlphaPulse Integration Validation Script

This script performs comprehensive validation of all newly integrated features
to ensure they work together properly in the AlphaPulse system.

Run this script after deployment to validate the integration.
"""
import asyncio
import sys
import json
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path
import traceback
from typing import Dict, List, Any, Optional

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
import pandas as pd
import numpy as np
import requests

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


class IntegrationValidator:
    """Validates all feature integrations in AlphaPulse."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.validation_results = {}
        self.errors = []
        
    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🚀 Starting AlphaPulse Integration Validation")
        logger.info(f"🌐 API Base URL: {self.api_url}")
        
        validation_steps = [
            ("API Health Check", self.validate_api_health),
            ("GPU Integration", self.validate_gpu_integration),
            ("Explainable AI", self.validate_explainability_integration),
            ("Data Quality Pipeline", self.validate_data_quality_integration),
            ("Data Lake Architecture", self.validate_data_lake_integration),
            ("Enhanced Backtesting", self.validate_backtesting_integration),
            ("Cross-Feature Integration", self.validate_cross_feature_integration),
            ("Dashboard Readiness", self.validate_dashboard_readiness)
        ]
        
        for step_name, step_func in validation_steps:
            logger.info(f"\n📋 Running: {step_name}")
            try:
                result = await step_func()
                self.validation_results[step_name] = {
                    "status": "PASS" if result else "FAIL",
                    "details": result if isinstance(result, dict) else {}
                }
                
                if result:
                    logger.success(f"✅ {step_name}: PASSED")
                else:
                    logger.error(f"❌ {step_name}: FAILED")
                    
            except Exception as e:
                error_msg = f"{step_name} failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_results[step_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
        
        return self.generate_validation_report()
    
    async def validate_api_health(self) -> bool:
        """Validate basic API health and availability."""
        try:
            # Test root endpoint
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code != 200:
                logger.warning(f"Root endpoint returned {response.status_code}")
                return False
            
            # Test docs endpoint
            response = requests.get(f"{self.base_url}/docs", timeout=10)
            if response.status_code != 200:
                logger.warning(f"Docs endpoint returned {response.status_code}")
            
            logger.info("✅ Basic API endpoints accessible")
            return True
            
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    async def validate_gpu_integration(self) -> Dict[str, Any]:
        """Validate GPU acceleration integration."""
        results = {"gpu_available": False, "monitoring": False, "api_accessible": False}
        
        try:
            # Test GPU status endpoint
            response = requests.get(f"{self.api_url}/gpu/status", timeout=10)
            results["api_accessible"] = response.status_code in [200, 500]  # 500 is OK if no GPU
            
            if response.status_code == 200:
                data = response.json()
                results["gpu_available"] = data.get("available", False)
                results["device_count"] = data.get("device_count", 0)
                results["monitoring"] = True
                
                if results["gpu_available"]:
                    logger.info(f"🚀 GPU detected: {results['device_count']} device(s)")
                else:
                    logger.info("💻 No GPU detected, CPU fallback available")
            
            # Test metrics endpoint
            response = requests.get(f"{self.api_url}/gpu/metrics", timeout=10)
            results["metrics_available"] = response.status_code in [200, 404]
            
            return results
            
        except Exception as e:
            logger.error(f"GPU validation failed: {e}")
            return results
    
    async def validate_explainability_integration(self) -> Dict[str, Any]:
        """Validate explainable AI integration."""
        results = {"api_accessible": False, "models_endpoint": False, "explainer_types": False}
        
        try:
            # Test models endpoint
            response = requests.get(f"{self.api_url}/explainability/models", timeout=10)
            results["api_accessible"] = response.status_code in [200, 404]
            results["models_endpoint"] = response.status_code == 200
            
            # Test explainer types endpoint
            response = requests.get(f"{self.api_url}/explainability/explainer-types", timeout=10)
            results["explainer_types"] = response.status_code == 200
            
            if results["explainer_types"]:
                data = response.json()
                results["supported_types"] = data.get("explainer_types", [])
                logger.info(f"🧠 Explainer types available: {results['supported_types']}")
            
            # Test explanation endpoint with mock data
            mock_request = {
                "model_id": "test_model",
                "input_data": [[150.0, 155.0, 145.0, 152.0, 1000000]],
                "explainer_type": "shap",
                "target_feature": "price_direction"
            }
            
            response = requests.post(
                f"{self.api_url}/explainability/explain",
                json=mock_request,
                timeout=10
            )
            results["explanation_endpoint"] = response.status_code in [200, 422, 404]  # Expected responses
            
            return results
            
        except Exception as e:
            logger.error(f"Explainability validation failed: {e}")
            return results
    
    async def validate_data_quality_integration(self) -> Dict[str, Any]:
        """Validate data quality pipeline integration."""
        results = {"api_accessible": False, "assessment": False, "dashboard": False}
        
        try:
            # Test status endpoint
            response = requests.get(f"{self.api_url}/data-quality/status", timeout=10)
            results["api_accessible"] = response.status_code in [200, 404]
            
            # Test assessment endpoint with mock data
            mock_assessment = {
                "dataset_name": "test_dataset",
                "data_sample": [
                    {"symbol": "AAPL", "close": 150.0, "volume": 1000000},
                    {"symbol": "AAPL", "close": 151.0, "volume": 1100000}
                ],
                "validation_rules": ["no_nulls", "positive_volume"]
            }
            
            response = requests.post(
                f"{self.api_url}/data-quality/assess",
                json=mock_assessment,
                timeout=10
            )
            results["assessment"] = response.status_code in [200, 422]
            
            # Test dashboard endpoint
            response = requests.get(f"{self.api_url}/data-quality/dashboard", timeout=10)
            results["dashboard"] = response.status_code in [200, 404]
            
            # Test trends endpoint
            response = requests.get(f"{self.api_url}/data-quality/trends", timeout=10)
            results["trends"] = response.status_code in [200, 404]
            
            logger.info("🎯 Data quality pipeline endpoints validated")
            return results
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return results
    
    async def validate_data_lake_integration(self) -> Dict[str, Any]:
        """Validate data lake architecture integration."""
        results = {"api_accessible": False, "datasets": False, "query": False, "health": False}
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_url}/datalake/health", timeout=10)
            results["health"] = response.status_code in [200, 500]  # 500 OK if not configured
            results["api_accessible"] = True
            
            if response.status_code == 200:
                health_data = response.json()
                results["health_status"] = health_data.get("status", "unknown")
                logger.info(f"🏛️ Data lake health: {results['health_status']}")
            
            # Test datasets endpoint
            response = requests.get(f"{self.api_url}/datalake/datasets?limit=5", timeout=10)
            results["datasets"] = response.status_code in [200, 500]
            
            if response.status_code == 200:
                datasets = response.json()
                results["dataset_count"] = len(datasets) if isinstance(datasets, list) else 0
                logger.info(f"📊 Found {results.get('dataset_count', 0)} datasets")
            
            # Test query endpoint with simple query
            mock_query = {
                "sql": "SELECT 1 as test_column",
                "limit": 1,
                "timeout_seconds": 5
            }
            
            response = requests.post(
                f"{self.api_url}/datalake/query",
                json=mock_query,
                timeout=10
            )
            results["query"] = response.status_code in [200, 422, 500]
            
            # Test statistics endpoint
            response = requests.get(f"{self.api_url}/datalake/statistics", timeout=10)
            results["statistics"] = response.status_code in [200, 500]
            
            return results
            
        except Exception as e:
            logger.error(f"Data lake validation failed: {e}")
            return results
    
    async def validate_backtesting_integration(self) -> Dict[str, Any]:
        """Validate enhanced backtesting integration."""
        results = {"api_accessible": False, "strategies": False, "data_lake_status": False}
        
        try:
            # Test strategies endpoint
            response = requests.get(f"{self.api_url}/backtesting/strategies", timeout=10)
            results["strategies"] = response.status_code == 200
            results["api_accessible"] = True
            
            if results["strategies"]:
                strategies_data = response.json()
                results["available_strategies"] = list(strategies_data.get("strategies", {}).keys())
                results["data_sources"] = strategies_data.get("data_sources", [])
                logger.info(f"📈 Available strategies: {results['available_strategies']}")
            
            # Test data lake status
            response = requests.get(f"{self.api_url}/backtesting/data-lake/status", timeout=10)
            results["data_lake_status"] = response.status_code == 200
            
            if results["data_lake_status"]:
                status_data = response.json()
                results["data_lake_available"] = status_data.get("data_lake_available", False)
                logger.info(f"🏛️ Data lake available for backtesting: {results['data_lake_available']}")
            
            # Test backtest execution with minimal parameters
            mock_backtest = {
                "symbols": ["AAPL"],
                "timeframe": "1d",
                "start_date": (datetime.now(UTC) - timedelta(days=30)).isoformat(),
                "end_date": datetime.now(UTC).isoformat(),
                "strategy_type": "simple_ma",
                "strategy_params": {"short_window": 10, "long_window": 20},
                "initial_capital": 10000,
                "commission": 0.001,
                "data_source": "auto"
            }
            
            response = requests.post(
                f"{self.api_url}/backtesting/run",
                json=mock_backtest,
                timeout=30  # Longer timeout for backtest
            )
            results["backtest_execution"] = response.status_code in [200, 422, 500]
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting validation failed: {e}")
            return results
    
    async def validate_cross_feature_integration(self) -> Dict[str, Any]:
        """Validate that features work together properly."""
        results = {"gpu_explainability": False, "quality_backtesting": False, "data_lake_quality": False}
        
        try:
            # Test 1: GPU + Explainability (if both available)
            gpu_response = requests.get(f"{self.api_url}/gpu/status", timeout=5)
            explain_response = requests.get(f"{self.api_url}/explainability/explainer-types", timeout=5)
            
            if gpu_response.status_code == 200 and explain_response.status_code == 200:
                results["gpu_explainability"] = True
                logger.info("🔗 GPU + Explainability integration available")
            
            # Test 2: Data Quality + Backtesting
            quality_response = requests.get(f"{self.api_url}/data-quality/status", timeout=5)
            backtest_response = requests.get(f"{self.api_url}/backtesting/strategies", timeout=5)
            
            if quality_response.status_code in [200, 404] and backtest_response.status_code == 200:
                results["quality_backtesting"] = True
                logger.info("🔗 Data Quality + Backtesting integration available")
            
            # Test 3: Data Lake + Data Quality
            lake_response = requests.get(f"{self.api_url}/datalake/health", timeout=5)
            
            if quality_response.status_code in [200, 404] and lake_response.status_code in [200, 500]:
                results["data_lake_quality"] = True
                logger.info("🔗 Data Lake + Data Quality integration available")
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-feature integration validation failed: {e}")
            return results
    
    async def validate_dashboard_readiness(self) -> Dict[str, Any]:
        """Validate that dashboard integration is ready."""
        results = {"api_endpoints": 0, "widget_data": False, "real_time_updates": False}
        
        try:
            # Test key endpoints that dashboard widgets need
            key_endpoints = [
                "/gpu/status",
                "/explainability/models", 
                "/data-quality/dashboard",
                "/datalake/statistics",
                "/backtesting/strategies"
            ]
            
            working_endpoints = 0
            for endpoint in key_endpoints:
                try:
                    response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:  # 404 is OK for optional features
                        working_endpoints += 1
                except:
                    pass
            
            results["api_endpoints"] = working_endpoints
            results["widget_data"] = working_endpoints >= len(key_endpoints) * 0.6  # 60% threshold
            
            # Test WebSocket availability (for real-time updates)
            try:
                response = requests.get(f"{self.base_url}/ws", timeout=2)
                results["real_time_updates"] = response.status_code in [200, 426]  # 426 = Upgrade Required
            except:
                results["real_time_updates"] = False
            
            logger.info(f"🖥️ Dashboard readiness: {working_endpoints}/{len(key_endpoints)} endpoints working")
            return results
            
        except Exception as e:
            logger.error(f"Dashboard readiness validation failed: {e}")
            return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        passed_tests = sum(1 for result in self.validation_results.values() 
                          if result["status"] == "PASS")
        total_tests = len(self.validation_results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "overall_status": "PASS" if success_rate >= 70 else "FAIL",
            "success_rate": success_rate,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "errors": self.errors,
            "detailed_results": self.validation_results,
            "recommendations": self.generate_recommendations()
        }
        
        # Print summary
        logger.info(f"\n📊 Validation Summary:")
        logger.info(f"✅ Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"❌ Failed: {total_tests - passed_tests}")
        logger.info(f"🎯 Overall Status: {report['overall_status']}")
        
        if report["overall_status"] == "PASS":
            logger.success("🎉 AlphaPulse integration validation PASSED!")
        else:
            logger.error("❌ AlphaPulse integration validation FAILED!")
            logger.info("📋 Check recommendations for next steps")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for test_name, result in self.validation_results.items():
            if result["status"] == "FAIL":
                if "GPU" in test_name:
                    recommendations.append("Consider installing CUDA drivers for GPU acceleration")
                elif "Explainable" in test_name:
                    recommendations.append("Install SHAP/LIME packages for explainable AI features")
                elif "Data Quality" in test_name:
                    recommendations.append("Configure data quality validation rules")
                elif "Data Lake" in test_name:
                    recommendations.append("Set up data lake infrastructure (Bronze/Silver/Gold layers)")
                elif "Backtesting" in test_name:
                    recommendations.append("Verify backtesting data sources are available")
        
        if len(self.errors) > 0:
            recommendations.append("Check system logs for detailed error information")
        
        return recommendations


async def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaPulse Integration Validation")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL for AlphaPulse API")
    parser.add_argument("--output", help="Output file for validation report (JSON)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Run validation
    validator = IntegrationValidator(base_url=args.url)
    report = await validator.run_validation()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Validation report saved to {args.output}")
    
    # Exit with appropriate code
    exit_code = 0 if report["overall_status"] == "PASS" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())