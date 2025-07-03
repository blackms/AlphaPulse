"""
Validation rules configuration for AlphaPulse API endpoints.

Defines validation rules for each API endpoint including:
- Query parameter validation
- Path parameter validation
- Request body validation
- File upload validation
"""

from typing import Dict, Any, Optional
from decimal import Decimal


# Validation rules for API endpoints
ENDPOINT_VALIDATION_RULES = {
    # Authentication endpoints
    '/api/v1/auth/login': {
        'POST': {
            'body': {
                'required_fields': ['username', 'password'],
                'fields': {
                    'username': {
                        'type': 'string',
                        'min_length': 3,
                        'max_length': 50,
                        'pattern': 'alpha_numeric'
                    },
                    'password': {
                        'type': 'string',
                        'min_length': 8,
                        'max_length': 128
                    }
                }
            }
        }
    },
    
    '/api/v1/auth/register': {
        'POST': {
            'body': {
                'required_fields': ['username', 'email', 'password', 'full_name'],
                'fields': {
                    'username': {
                        'type': 'string',
                        'min_length': 3,
                        'max_length': 50,
                        'pattern': 'alpha_numeric'
                    },
                    'email': {
                        'type': 'email'
                    },
                    'password': {
                        'type': 'password'
                    },
                    'full_name': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 100,
                        'pattern': 'alpha_numeric_spaces'
                    },
                    'phone': {
                        'type': 'phone',
                        'required': False
                    }
                }
            }
        }
    },
    
    # User management endpoints
    '/api/v1/users/{user_id}': {
        'GET': {
            'path_params': {
                'user_id': {
                    'type': 'string',
                    'pattern': 'uuid'
                }
            }
        },
        'PUT': {
            'path_params': {
                'user_id': {
                    'type': 'string',
                    'pattern': 'uuid'
                }
            },
            'body': {
                'fields': {
                    'email': {
                        'type': 'email',
                        'required': False
                    },
                    'full_name': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 100,
                        'pattern': 'alpha_numeric_spaces',
                        'required': False
                    },
                    'phone': {
                        'type': 'phone',
                        'required': False
                    }
                }
            }
        }
    },
    
    # Trading endpoints
    '/api/v1/trades/orders': {
        'GET': {
            'query_params': {
                'page': {
                    'type': 'integer',
                    'min_value': 1,
                    'required': False
                },
                'page_size': {
                    'type': 'integer',
                    'min_value': 1,
                    'max_value': 100,
                    'required': False
                },
                'symbol': {
                    'type': 'stock_symbol',
                    'required': False
                },
                'status': {
                    'type': 'string',
                    'pattern': None,  # Will be validated against enum
                    'required': False
                },
                'start_date': {
                    'type': 'datetime',
                    'required': False
                },
                'end_date': {
                    'type': 'datetime',
                    'required': False
                }
            }
        },
        'POST': {
            'body': {
                'required_fields': ['symbol', 'side', 'order_type', 'quantity'],
                'fields': {
                    'symbol': {
                        'type': 'stock_symbol'
                    },
                    'side': {
                        'type': 'string',
                        'allowed_values': ['buy', 'sell']
                    },
                    'order_type': {
                        'type': 'string',
                        'allowed_values': ['market', 'limit', 'stop', 'stop_limit']
                    },
                    'quantity': {
                        'type': 'integer',
                        'min_value': 1,
                        'max_value': 1000000
                    },
                    'price': {
                        'type': 'decimal',
                        'min_value': Decimal('0.01'),
                        'max_value': Decimal('1000000'),
                        'max_decimal_places': 4,
                        'required': False
                    },
                    'stop_price': {
                        'type': 'decimal',
                        'min_value': Decimal('0.01'),
                        'max_value': Decimal('1000000'),
                        'max_decimal_places': 4,
                        'required': False
                    },
                    'time_in_force': {
                        'type': 'string',
                        'allowed_values': ['day', 'gtc', 'ioc', 'fok'],
                        'required': False
                    }
                }
            }
        }
    },
    
    '/api/v1/trades/orders/{order_id}': {
        'GET': {
            'path_params': {
                'order_id': {
                    'type': 'string',
                    'pattern': 'uuid'
                }
            }
        },
        'PUT': {
            'path_params': {
                'order_id': {
                    'type': 'string',
                    'pattern': 'uuid'
                }
            },
            'body': {
                'fields': {
                    'quantity': {
                        'type': 'integer',
                        'min_value': 1,
                        'max_value': 1000000,
                        'required': False
                    },
                    'price': {
                        'type': 'decimal',
                        'min_value': Decimal('0.01'),
                        'max_value': Decimal('1000000'),
                        'max_decimal_places': 4,
                        'required': False
                    }
                }
            }
        },
        'DELETE': {
            'path_params': {
                'order_id': {
                    'type': 'string',
                    'pattern': 'uuid'
                }
            }
        }
    },
    
    # Portfolio endpoints
    '/api/v1/portfolio': {
        'GET': {
            'query_params': {
                'include_positions': {
                    'type': 'string',
                    'allowed_values': ['true', 'false'],
                    'required': False
                },
                'include_performance': {
                    'type': 'string',
                    'allowed_values': ['true', 'false'],
                    'required': False
                }
            }
        }
    },
    
    '/api/v1/portfolio/allocations': {
        'GET': {
            'query_params': {
                'asset_type': {
                    'type': 'string',
                    'allowed_values': ['stock', 'crypto', 'forex', 'commodity'],
                    'required': False
                }
            }
        },
        'POST': {
            'body': {
                'required_fields': ['allocations'],
                'fields': {
                    'allocations': {
                        'type': 'object',
                        'validation': 'custom'  # Custom validation for allocation percentages
                    },
                    'rebalance_threshold': {
                        'type': 'percentage',
                        'required': False
                    }
                }
            }
        }
    },
    
    '/api/v1/portfolio/risk': {
        'GET': {},
        'POST': {
            'body': {
                'required_fields': ['max_position_size', 'max_daily_loss'],
                'fields': {
                    'max_position_size': {
                        'type': 'percentage',
                        'min_value': Decimal('1'),
                        'max_value': Decimal('50')
                    },
                    'max_daily_loss': {
                        'type': 'percentage',
                        'min_value': Decimal('1'),
                        'max_value': Decimal('20')
                    },
                    'max_leverage': {
                        'type': 'decimal',
                        'min_value': Decimal('1.0'),
                        'max_value': Decimal('10.0'),
                        'required': False
                    },
                    'stop_loss_percentage': {
                        'type': 'percentage',
                        'required': False
                    }
                }
            }
        }
    },
    
    # Market data endpoints
    '/api/v1/market/quote/{symbol}': {
        'GET': {
            'path_params': {
                'symbol': {
                    'type': 'stock_symbol'
                }
            }
        }
    },
    
    '/api/v1/market/history/{symbol}': {
        'GET': {
            'path_params': {
                'symbol': {
                    'type': 'stock_symbol'
                }
            },
            'query_params': {
                'start_date': {
                    'type': 'datetime',
                    'required': False
                },
                'end_date': {
                    'type': 'datetime',
                    'required': False
                },
                'interval': {
                    'type': 'string',
                    'allowed_values': ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1M'],
                    'required': False
                },
                'limit': {
                    'type': 'integer',
                    'min_value': 1,
                    'max_value': 5000,
                    'required': False
                }
            }
        }
    },
    
    # Analytics endpoints
    '/api/v1/analytics/performance': {
        'GET': {
            'query_params': {
                'start_date': {
                    'type': 'datetime',
                    'required': True
                },
                'end_date': {
                    'type': 'datetime',
                    'required': True
                },
                'benchmark': {
                    'type': 'stock_symbol',
                    'required': False
                },
                'include_costs': {
                    'type': 'string',
                    'allowed_values': ['true', 'false'],
                    'required': False
                }
            }
        }
    },
    
    '/api/v1/analytics/risk': {
        'GET': {
            'query_params': {
                'metric': {
                    'type': 'string',
                    'allowed_values': ['var', 'cvar', 'sharpe', 'sortino', 'max_drawdown'],
                    'required': False
                },
                'confidence_level': {
                    'type': 'decimal',
                    'min_value': Decimal('0.90'),
                    'max_value': Decimal('0.99'),
                    'required': False
                }
            }
        }
    },
    
    # System administration endpoints
    '/api/v1/admin/users': {
        'GET': {
            'query_params': {
                'page': {
                    'type': 'integer',
                    'min_value': 1,
                    'required': False
                },
                'page_size': {
                    'type': 'integer',
                    'min_value': 1,
                    'max_value': 100,
                    'required': False
                },
                'user_tier': {
                    'type': 'string',
                    'allowed_values': ['basic', 'premium', 'professional', 'institutional'],
                    'required': False
                },
                'status': {
                    'type': 'string',
                    'allowed_values': ['active', 'inactive', 'suspended'],
                    'required': False
                }
            }
        }
    },
    
    '/api/v1/admin/system/config': {
        'GET': {},
        'POST': {
            'body': {
                'fields': {
                    'trading_enabled': {
                        'type': 'string',
                        'allowed_values': ['true', 'false'],
                        'required': False
                    },
                    'market_data_enabled': {
                        'type': 'string',
                        'allowed_values': ['true', 'false'],
                        'required': False
                    },
                    'max_concurrent_orders': {
                        'type': 'integer',
                        'min_value': 1,
                        'max_value': 1000,
                        'required': False
                    },
                    'rate_limit_requests_per_minute': {
                        'type': 'integer',
                        'min_value': 10,
                        'max_value': 10000,
                        'required': False
                    }
                }
            }
        }
    },
    
    # Audit endpoints
    '/api/v1/audit/logs': {
        'GET': {
            'query_params': {
                'start_date': {
                    'type': 'datetime',
                    'required': False
                },
                'end_date': {
                    'type': 'datetime',
                    'required': False
                },
                'event_type': {
                    'type': 'string',
                    'allowed_values': [
                        'user_login', 'user_logout', 'trade_placed', 'trade_executed',
                        'portfolio_rebalanced', 'risk_limit_changed', 'system_config_changed'
                    ],
                    'required': False
                },
                'severity': {
                    'type': 'string',
                    'allowed_values': ['debug', 'info', 'warning', 'error', 'critical'],
                    'required': False
                },
                'user_id': {
                    'type': 'string',
                    'pattern': 'uuid',
                    'required': False
                },
                'page': {
                    'type': 'integer',
                    'min_value': 1,
                    'required': False
                },
                'page_size': {
                    'type': 'integer',
                    'min_value': 1,
                    'max_value': 1000,
                    'required': False
                }
            }
        }
    },
    
    # Security endpoints
    '/api/v1/security/events': {
        'POST': {
            'body': {
                'required_fields': ['event_type', 'severity', 'description'],
                'fields': {
                    'event_type': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 50
                    },
                    'severity': {
                        'type': 'string',
                        'allowed_values': ['low', 'medium', 'high', 'critical']
                    },
                    'description': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 1000
                    },
                    'ip_address': {
                        'type': 'string',
                        'required': False
                    },
                    'user_agent': {
                        'type': 'string',
                        'max_length': 500,
                        'required': False
                    }
                }
            }
        }
    },
    
    '/api/v1/security/ip-filters': {
        'GET': {
            'query_params': {
                'rule_type': {
                    'type': 'string',
                    'allowed_values': ['whitelist', 'blacklist', 'geo_block', 'asn_block'],
                    'required': False
                },
                'active_only': {
                    'type': 'string',
                    'allowed_values': ['true', 'false'],
                    'required': False
                }
            }
        },
        'POST': {
            'body': {
                'required_fields': ['rule_type', 'pattern', 'reason'],
                'fields': {
                    'rule_type': {
                        'type': 'string',
                        'allowed_values': ['whitelist', 'blacklist', 'geo_block', 'asn_block']
                    },
                    'pattern': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 100
                    },
                    'reason': {
                        'type': 'string',
                        'min_length': 1,
                        'max_length': 500
                    },
                    'expires_at': {
                        'type': 'datetime',
                        'required': False
                    },
                    'is_active': {
                        'type': 'string',
                        'allowed_values': ['true', 'false'],
                        'required': False
                    }
                }
            }
        }
    },
    
    # File upload endpoints
    '/api/v1/files/upload': {
        'POST': {
            'files': {
                'file': {
                    'allowed_extensions': ['csv', 'json', 'xlsx', 'pdf'],
                    'max_size': 10 * 1024 * 1024  # 10MB
                }
            },
            'body': {
                'fields': {
                    'file_type': {
                        'type': 'string',
                        'allowed_values': ['csv', 'json', 'xlsx', 'pdf']
                    },
                    'category': {
                        'type': 'string',
                        'allowed_values': ['portfolio', 'trades', 'research', 'reports']
                    },
                    'description': {
                        'type': 'string',
                        'max_length': 500,
                        'required': False
                    }
                }
            }
        }
    }
}


# Default validation rules for common parameter types
DEFAULT_VALIDATION_RULES = {
    'pagination': {
        'page': {
            'type': 'integer',
            'min_value': 1,
            'required': False
        },
        'page_size': {
            'type': 'integer',
            'min_value': 1,
            'max_value': 100,
            'required': False
        }
    },
    
    'date_range': {
        'start_date': {
            'type': 'datetime',
            'required': False
        },
        'end_date': {
            'type': 'datetime',
            'required': False
        }
    },
    
    'financial_filters': {
        'min_amount': {
            'type': 'decimal',
            'min_value': Decimal('0.01'),
            'required': False
        },
        'max_amount': {
            'type': 'decimal',
            'min_value': Decimal('0.01'),
            'max_value': Decimal('1000000000'),
            'required': False
        }
    }
}


def get_endpoint_validation_rules(path: str, method: str) -> Dict[str, Any]:
    """
    Get validation rules for a specific endpoint and method.
    
    Args:
        path: API endpoint path
        method: HTTP method
        
    Returns:
        Dictionary containing validation rules
    """
    # Try exact path match first
    if path in ENDPOINT_VALIDATION_RULES:
        endpoint_rules = ENDPOINT_VALIDATION_RULES[path]
        if method.upper() in endpoint_rules:
            return endpoint_rules[method.upper()]
    
    # Try path parameter matching
    for pattern, rules in ENDPOINT_VALIDATION_RULES.items():
        if _match_path_pattern(path, pattern):
            if method.upper() in rules:
                return rules[method.upper()]
    
    # Return empty rules if no match found
    return {}


def _match_path_pattern(path: str, pattern: str) -> bool:
    """
    Match path against pattern with parameter placeholders.
    
    Args:
        path: Actual request path
        pattern: Pattern with {param} placeholders
        
    Returns:
        True if path matches pattern
    """
    import re
    
    # Convert pattern to regex
    # Replace {param} with regex group
    regex_pattern = re.sub(r'\{[^}]+\}', r'[^/]+', pattern)
    regex_pattern = f"^{regex_pattern}$"
    
    return bool(re.match(regex_pattern, path))


def add_endpoint_validation_rules(path: str, method: str, rules: Dict[str, Any]):
    """
    Add or update validation rules for an endpoint.
    
    Args:
        path: API endpoint path
        method: HTTP method
        rules: Validation rules dictionary
    """
    if path not in ENDPOINT_VALIDATION_RULES:
        ENDPOINT_VALIDATION_RULES[path] = {}
    
    ENDPOINT_VALIDATION_RULES[path][method.upper()] = rules


def get_default_rules(rule_type: str) -> Dict[str, Any]:
    """
    Get default validation rules for common patterns.
    
    Args:
        rule_type: Type of default rules (pagination, date_range, etc.)
        
    Returns:
        Dictionary containing default rules
    """
    return DEFAULT_VALIDATION_RULES.get(rule_type, {})


# Security-focused validation configurations
SECURITY_VALIDATION_CONFIG = {
    'max_request_size': 10 * 1024 * 1024,  # 10MB
    'max_field_length': 10000,
    'max_array_items': 1000,
    'max_object_depth': 10,
    'blocked_user_agents': [
        'sqlmap',
        'nikto',
        'nessus',
        'openvas',
        'w3af',
        'burp',
        'zap'
    ],
    'suspicious_patterns': [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'expression\s*\(',
        r'vbscript:',
        r'data:text/html'
    ],
    'file_upload': {
        'max_size': 10 * 1024 * 1024,  # 10MB
        'allowed_types': ['text/csv', 'application/json', 'application/pdf'],
        'blocked_extensions': ['exe', 'bat', 'sh', 'ps1', 'jar', 'com'],
        'scan_for_malware': True
    }
}


def get_security_config() -> Dict[str, Any]:
    """Get security validation configuration."""
    return SECURITY_VALIDATION_CONFIG


def is_suspicious_request(user_agent: str, path: str, params: Dict[str, Any]) -> bool:
    """
    Check if request appears suspicious based on security heuristics.
    
    Args:
        user_agent: User agent string
        path: Request path
        params: Request parameters
        
    Returns:
        True if request appears suspicious
    """
    # Check user agent
    if user_agent.lower() in [ua.lower() for ua in SECURITY_VALIDATION_CONFIG['blocked_user_agents']]:
        return True
    
    # Check for suspicious patterns in parameters
    import re
    for pattern in SECURITY_VALIDATION_CONFIG['suspicious_patterns']:
        for value in params.values():
            if isinstance(value, str) and re.search(pattern, value, re.IGNORECASE):
                return True
    
    # Check for path traversal attempts
    if '../' in path or '..\\' in path:
        return True
    
    return False