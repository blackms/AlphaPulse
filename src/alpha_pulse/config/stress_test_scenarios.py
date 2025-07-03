"""
Predefined stress test scenarios for portfolio risk analysis.

Contains historical and hypothetical scenarios with calibrated shocks.
"""

STRESS_TEST_SCENARIOS = {
    "historical": [
        {
            "name": "2008 Financial Crisis",
            "start_date": "2008-09-01",
            "end_date": "2009-03-31",
            "description": "Global financial crisis triggered by subprime mortgage collapse",
            "probability": 0.02,
            "severity": "extreme",
            "fallback_shocks": {
                "equity": -0.55,
                "bond": -0.15,
                "commodity": -0.60,
                "fx": -0.20,
                "crypto": -0.70,
                "default": -0.40
            },
            "volatility_multiplier": 3.5,
            "correlation_increase": 0.8,
            "metadata": {
                "sp500_drawdown": -56.8,
                "vix_peak": 80.86,
                "duration_days": 212
            }
        },
        {
            "name": "COVID-19 Market Crash",
            "start_date": "2020-02-20",
            "end_date": "2020-03-23",
            "description": "Pandemic-induced market selloff and liquidity crisis",
            "probability": 0.03,
            "severity": "severe",
            "fallback_shocks": {
                "equity": -0.35,
                "bond": -0.08,
                "commodity": -0.40,
                "fx": -0.15,
                "crypto": -0.50,
                "default": -0.30
            },
            "volatility_multiplier": 3.0,
            "correlation_increase": 0.7,
            "metadata": {
                "sp500_drawdown": -33.9,
                "vix_peak": 82.69,
                "duration_days": 33
            }
        },
        {
            "name": "Dot-Com Bubble Burst",
            "start_date": "2000-03-10",
            "end_date": "2002-10-09",
            "description": "Technology stock bubble collapse",
            "probability": 0.03,
            "severity": "extreme",
            "fallback_shocks": {
                "equity": -0.49,
                "tech": -0.78,
                "bond": 0.15,
                "commodity": -0.20,
                "fx": -0.10,
                "default": -0.35
            },
            "volatility_multiplier": 2.5,
            "correlation_increase": 0.6,
            "metadata": {
                "nasdaq_drawdown": -78.0,
                "duration_days": 929
            }
        },
        {
            "name": "Black Monday 1987",
            "start_date": "1987-10-19",
            "end_date": "1987-10-19",
            "description": "Single-day market crash",
            "probability": 0.001,
            "severity": "extreme",
            "fallback_shocks": {
                "equity": -0.22,
                "bond": 0.02,
                "commodity": -0.10,
                "fx": -0.05,
                "default": -0.20
            },
            "volatility_multiplier": 5.0,
            "correlation_increase": 0.9,
            "metadata": {
                "dow_single_day_loss": -22.6,
                "duration_days": 1
            }
        },
        {
            "name": "European Debt Crisis",
            "start_date": "2011-05-01",
            "end_date": "2011-10-01",
            "description": "Sovereign debt crisis in European periphery",
            "probability": 0.05,
            "severity": "moderate",
            "fallback_shocks": {
                "equity": -0.20,
                "bond": -0.12,
                "commodity": -0.15,
                "fx": -0.25,
                "default": -0.18
            },
            "volatility_multiplier": 2.0,
            "correlation_increase": 0.5,
            "metadata": {
                "affected_regions": ["Europe", "Emerging Markets"],
                "duration_days": 153
            }
        },
        {
            "name": "Flash Crash 2010",
            "start_date": "2010-05-06",
            "end_date": "2010-05-06",
            "description": "High-frequency trading induced crash",
            "probability": 0.01,
            "severity": "severe",
            "fallback_shocks": {
                "equity": -0.09,
                "bond": 0.01,
                "commodity": -0.05,
                "fx": -0.03,
                "default": -0.08
            },
            "volatility_multiplier": 4.0,
            "correlation_increase": 0.8,
            "metadata": {
                "intraday_swing": -9.2,
                "recovery_time_minutes": 36,
                "duration_days": 1
            }
        },
        {
            "name": "Taper Tantrum 2013",
            "start_date": "2013-05-22",
            "end_date": "2013-06-24",
            "description": "Fed tapering announcement market reaction",
            "probability": 0.08,
            "severity": "moderate",
            "fallback_shocks": {
                "equity": -0.06,
                "bond": -0.15,
                "commodity": -0.08,
                "fx": -0.10,
                "emerging_markets": -0.20,
                "default": -0.10
            },
            "volatility_multiplier": 1.8,
            "correlation_increase": 0.4,
            "metadata": {
                "10y_yield_increase": 1.0,
                "duration_days": 33
            }
        },
        {
            "name": "China Stock Market Crash 2015",
            "start_date": "2015-06-12",
            "end_date": "2015-08-26",
            "description": "Chinese equity bubble burst",
            "probability": 0.04,
            "severity": "severe",
            "fallback_shocks": {
                "equity": -0.25,
                "china_equity": -0.45,
                "commodity": -0.30,
                "fx": -0.12,
                "emerging_markets": -0.28,
                "default": -0.20
            },
            "volatility_multiplier": 2.5,
            "correlation_increase": 0.6,
            "metadata": {
                "shanghai_composite_loss": -45.0,
                "global_contagion": True,
                "duration_days": 75
            }
        }
    ],
    
    "hypothetical": [
        {
            "name": "Severe Interest Rate Shock",
            "description": "Central banks raise rates aggressively to combat inflation",
            "probability": 0.10,
            "severity": "severe",
            "shocks": [
                {
                    "asset_class": "bond",
                    "type": "price",
                    "magnitude": -0.20,
                    "duration_days": 30
                },
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.15,
                    "duration_days": 30
                },
                {
                    "asset_class": "real_estate",
                    "type": "price",
                    "magnitude": -0.25,
                    "duration_days": 60
                },
                {
                    "asset_class": "all",
                    "type": "volatility",
                    "magnitude": 2.0,
                    "duration_days": 30
                }
            ],
            "metadata": {
                "rate_increase_bps": 300,
                "inflation_trigger": True
            }
        },
        {
            "name": "Geopolitical Crisis",
            "description": "Major geopolitical event disrupts global markets",
            "probability": 0.05,
            "severity": "severe",
            "shocks": [
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.25,
                    "duration_days": 20
                },
                {
                    "asset_class": "commodity",
                    "type": "price",
                    "magnitude": 0.30,  # Oil spike
                    "duration_days": 45
                },
                {
                    "asset_class": "fx",
                    "type": "price",
                    "magnitude": -0.15,
                    "duration_days": 20
                },
                {
                    "asset_class": "safe_haven",
                    "type": "price",
                    "magnitude": 0.10,  # Gold, USD, CHF rise
                    "duration_days": 30
                }
            ],
            "metadata": {
                "trigger": "military_conflict",
                "oil_supply_disruption": True
            }
        },
        {
            "name": "Credit Crisis",
            "description": "Corporate credit market freezes",
            "probability": 0.04,
            "severity": "extreme",
            "shocks": [
                {
                    "asset_class": "corporate_bond",
                    "type": "price",
                    "magnitude": -0.30,
                    "duration_days": 60
                },
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.35,
                    "duration_days": 45
                },
                {
                    "asset_class": "credit_spread",
                    "type": "spread",
                    "magnitude": 5.0,  # 500bps widening
                    "duration_days": 90
                }
            ],
            "metadata": {
                "default_rate_increase": 5.0,
                "liquidity_freeze": True
            }
        },
        {
            "name": "Currency Crisis",
            "description": "Major currency devaluation",
            "probability": 0.06,
            "severity": "severe",
            "shocks": [
                {
                    "asset_class": "fx",
                    "type": "price",
                    "magnitude": -0.40,
                    "duration_days": 10
                },
                {
                    "asset_class": "emerging_market_equity",
                    "type": "price",
                    "magnitude": -0.35,
                    "duration_days": 30
                },
                {
                    "asset_class": "emerging_market_bond",
                    "type": "price",
                    "magnitude": -0.25,
                    "duration_days": 30
                }
            ],
            "metadata": {
                "affected_currencies": ["TRY", "ARS", "ZAR"],
                "capital_flight": True
            }
        },
        {
            "name": "Technology Sector Crash",
            "description": "Tech bubble bursts due to regulation or valuation concerns",
            "probability": 0.07,
            "severity": "severe",
            "shocks": [
                {
                    "asset_class": "tech_equity",
                    "type": "price",
                    "magnitude": -0.40,
                    "duration_days": 60
                },
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.20,
                    "duration_days": 45
                },
                {
                    "asset_class": "crypto",
                    "type": "price",
                    "magnitude": -0.60,
                    "duration_days": 30
                }
            ],
            "metadata": {
                "trigger": "regulatory_crackdown",
                "valuation_reset": True
            }
        },
        {
            "name": "Stagflation Scenario",
            "description": "High inflation with economic stagnation",
            "probability": 0.08,
            "severity": "moderate",
            "shocks": [
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.15,
                    "duration_days": 180
                },
                {
                    "asset_class": "bond",
                    "type": "price",
                    "magnitude": -0.18,
                    "duration_days": 180
                },
                {
                    "asset_class": "commodity",
                    "type": "price",
                    "magnitude": 0.25,
                    "duration_days": 180
                },
                {
                    "asset_class": "real_asset",
                    "type": "price",
                    "magnitude": 0.15,
                    "duration_days": 180
                }
            ],
            "metadata": {
                "inflation_rate": 8.0,
                "gdp_growth": -1.0
            }
        },
        {
            "name": "Liquidity Crisis",
            "description": "Market-wide liquidity evaporation",
            "probability": 0.03,
            "severity": "extreme",
            "shocks": [
                {
                    "asset_class": "all",
                    "type": "liquidity",
                    "magnitude": -0.80,  # 80% reduction in liquidity
                    "duration_days": 10
                },
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.20,
                    "duration_days": 10
                },
                {
                    "asset_class": "all",
                    "type": "volatility",
                    "magnitude": 3.0,
                    "duration_days": 10
                }
            ],
            "metadata": {
                "bid_ask_spread_multiplier": 10,
                "market_depth_reduction": 0.9
            }
        },
        {
            "name": "Deflation Spiral",
            "description": "Deflationary pressures and economic contraction",
            "probability": 0.04,
            "severity": "severe",
            "shocks": [
                {
                    "asset_class": "equity",
                    "type": "price",
                    "magnitude": -0.30,
                    "duration_days": 365
                },
                {
                    "asset_class": "commodity",
                    "type": "price",
                    "magnitude": -0.40,
                    "duration_days": 365
                },
                {
                    "asset_class": "bond",
                    "type": "price",
                    "magnitude": 0.20,  # Flight to quality
                    "duration_days": 365
                }
            ],
            "metadata": {
                "deflation_rate": -2.0,
                "gdp_contraction": -3.0
            }
        }
    ],
    
    "regime_based": {
        "high_volatility": {
            "vol_multiplier": 2.5,
            "correlation_multiplier": 1.5,
            "typical_duration_days": 60
        },
        "risk_off": {
            "equity_beta": -1.5,
            "safe_haven_beta": 2.0,
            "typical_duration_days": 30
        },
        "correlation_breakdown": {
            "correlation_target": 0.9,
            "diversification_reduction": 0.8,
            "typical_duration_days": 20
        }
    }
}