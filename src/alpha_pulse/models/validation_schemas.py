"""
Pydantic validation schemas for AlphaPulse API.

Provides comprehensive data validation models for all API endpoints:
- Trading operations
- User management
- Portfolio operations
- System administration
- Security operations
"""

from typing import Optional, List, Dict, Any, Union
from decimal import Decimal
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator, EmailStr, constr, conint, confloat
import re

from alpha_pulse.utils.input_validator import validator as input_validator


class UserTierEnum(str, Enum):
    """User tier enumeration."""
    BASIC = "basic"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"


class OrderTypeEnum(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSideEnum(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class TimeInForceEnum(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class AssetTypeEnum(str, Enum):
    """Asset type enumeration."""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"


# Base validation schemas
class StockSymbol(str):
    """Custom stock symbol type."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError('Stock symbol must be a string')
        
        result = input_validator.validate_stock_symbol(v)
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        
        return result.sanitized_value


class MonetaryAmount(Decimal):
    """Custom monetary amount type."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        result = input_validator.validate_decimal(
            v,
            min_value=Decimal('0.01'),
            max_value=Decimal('1000000000'),
            max_decimal_places=8
        )
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        
        return result.sanitized_value


class PercentageValue(Decimal):
    """Custom percentage type."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        result = input_validator.validate_percentage(v)
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        
        return result.sanitized_value


# User management schemas
class UserRegistrationRequest(BaseModel):
    """User registration request schema."""
    
    username: constr(min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$') = Field(
        ..., description="Username (3-50 characters, alphanumeric and underscore only)"
    )
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (8-128 characters)")
    full_name: constr(min_length=1, max_length=100) = Field(..., description="Full name")
    phone: Optional[str] = Field(None, description="Phone number in E.164 format")
    user_tier: UserTierEnum = Field(UserTierEnum.BASIC, description="User tier")
    
    @validator('password')
    def validate_password_strength(cls, v):
        result = input_validator.validate_password(v)
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        return v
    
    @validator('phone')
    def validate_phone_number(cls, v):
        if v is not None:
            result = input_validator.validate_phone(v)
            if not result.is_valid:
                raise ValueError('; '.join(result.errors))
            return result.sanitized_value
        return v
    
    @validator('full_name')
    def validate_full_name(cls, v):
        result = input_validator.validate_string(
            v, 
            field_name="full_name",
            min_length=1,
            max_length=100,
            pattern='alpha_numeric_spaces'
        )
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        return result.sanitized_value


class UserUpdateRequest(BaseModel):
    """User update request schema."""
    
    email: Optional[EmailStr] = Field(None, description="Valid email address")
    full_name: Optional[constr(min_length=1, max_length=100)] = Field(None, description="Full name")
    phone: Optional[str] = Field(None, description="Phone number in E.164 format")
    user_tier: Optional[UserTierEnum] = Field(None, description="User tier")
    
    @validator('phone')
    def validate_phone_number(cls, v):
        if v is not None:
            result = input_validator.validate_phone(v)
            if not result.is_valid:
                raise ValueError('; '.join(result.errors))
            return result.sanitized_value
        return v


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @validator('new_password')
    def validate_new_password_strength(cls, v):
        result = input_validator.validate_password(v)
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        return v


# Trading schemas
class OrderRequest(BaseModel):
    """Order placement request schema."""
    
    symbol: StockSymbol = Field(..., description="Stock symbol")
    side: OrderSideEnum = Field(..., description="Order side (buy/sell)")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    quantity: conint(gt=0, le=1000000) = Field(..., description="Order quantity")
    price: Optional[MonetaryAmount] = Field(None, description="Order price (required for limit orders)")
    stop_price: Optional[MonetaryAmount] = Field(None, description="Stop price (for stop orders)")
    time_in_force: TimeInForceEnum = Field(TimeInForceEnum.DAY, description="Time in force")
    asset_type: AssetTypeEnum = Field(AssetTypeEnum.STOCK, description="Asset type")
    
    @validator('price')
    def validate_price_for_order_type(cls, v, values):
        order_type = values.get('order_type')
        if order_type in ['limit', 'stop_limit'] and v is None:
            raise ValueError(f"Price is required for {order_type} orders")
        return v
    
    @validator('stop_price')
    def validate_stop_price_for_order_type(cls, v, values):
        order_type = values.get('order_type')
        if order_type in ['stop', 'stop_limit', 'trailing_stop'] and v is None:
            raise ValueError(f"Stop price is required for {order_type} orders")
        return v


class OrderUpdateRequest(BaseModel):
    """Order update request schema."""
    
    quantity: Optional[conint(gt=0, le=1000000)] = Field(None, description="New order quantity")
    price: Optional[MonetaryAmount] = Field(None, description="New order price")
    stop_price: Optional[MonetaryAmount] = Field(None, description="New stop price")
    time_in_force: Optional[TimeInForceEnum] = Field(None, description="New time in force")


class PortfolioAllocationRequest(BaseModel):
    """Portfolio allocation request schema."""
    
    allocations: Dict[StockSymbol, PercentageValue] = Field(
        ..., 
        description="Asset allocations as percentages",
        min_items=1,
        max_items=100
    )
    rebalance_threshold: PercentageValue = Field(
        default=Decimal('5.0'),
        description="Rebalancing threshold percentage"
    )
    
    @validator('allocations')
    def validate_allocation_sum(cls, v):
        total = sum(v.values())
        if abs(total - 100) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Allocations must sum to 100%, got {total}%")
        return v


class RiskParametersRequest(BaseModel):
    """Risk parameters configuration schema."""
    
    max_position_size: PercentageValue = Field(
        ..., 
        description="Maximum position size as percentage of portfolio"
    )
    max_daily_loss: PercentageValue = Field(
        ...,
        description="Maximum daily loss as percentage of portfolio"
    )
    max_leverage: confloat(ge=1.0, le=10.0) = Field(
        default=1.0,
        description="Maximum leverage ratio"
    )
    stop_loss_percentage: PercentageValue = Field(
        default=Decimal('5.0'),
        description="Default stop loss percentage"
    )
    
    @validator('max_position_size')
    def validate_max_position_size(cls, v):
        if v > 50:  # Maximum 50% of portfolio per position
            raise ValueError("Maximum position size cannot exceed 50% of portfolio")
        return v
    
    @validator('max_daily_loss')
    def validate_max_daily_loss(cls, v):
        if v > 20:  # Maximum 20% daily loss
            raise ValueError("Maximum daily loss cannot exceed 20% of portfolio")
        return v


# System administration schemas
class SystemConfigurationRequest(BaseModel):
    """System configuration request schema."""
    
    trading_enabled: bool = Field(default=True, description="Enable/disable trading")
    market_data_enabled: bool = Field(default=True, description="Enable/disable market data")
    risk_management_enabled: bool = Field(default=True, description="Enable/disable risk management")
    max_concurrent_orders: conint(ge=1, le=1000) = Field(
        default=100,
        description="Maximum concurrent orders per user"
    )
    rate_limit_requests_per_minute: conint(ge=10, le=10000) = Field(
        default=1000,
        description="Rate limit requests per minute"
    )


class AuditLogQueryRequest(BaseModel):
    """Audit log query request schema."""
    
    start_date: Optional[datetime] = Field(None, description="Start date for log query")
    end_date: Optional[datetime] = Field(None, description="End date for log query")
    event_type: Optional[str] = Field(None, description="Event type filter")
    user_id: Optional[str] = Field(None, description="User ID filter")
    severity: Optional[str] = Field(None, description="Severity filter")
    page: conint(ge=1) = Field(default=1, description="Page number")
    page_size: conint(ge=1, le=1000) = Field(default=50, description="Page size")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v and v <= start_date:
            raise ValueError("End date must be after start date")
        return v


# Security schemas
class SecurityEventRequest(BaseModel):
    """Security event reporting schema."""
    
    event_type: constr(min_length=1, max_length=50) = Field(..., description="Security event type")
    severity: constr(regex=r'^(low|medium|high|critical)$') = Field(..., description="Event severity")
    description: constr(min_length=1, max_length=1000) = Field(..., description="Event description")
    ip_address: Optional[str] = Field(None, description="Source IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional event data")
    
    @validator('description')
    def validate_description(cls, v):
        result = input_validator.validate_string(
            v,
            field_name="description",
            min_length=1,
            max_length=1000,
            check_xss=True,
            check_sql=True
        )
        if not result.is_valid:
            raise ValueError('; '.join(result.errors))
        return result.sanitized_value


class IPFilterRuleRequest(BaseModel):
    """IP filter rule configuration schema."""
    
    rule_type: constr(regex=r'^(whitelist|blacklist|geo_block|asn_block)$') = Field(
        ..., 
        description="Filter rule type"
    )
    pattern: constr(min_length=1, max_length=100) = Field(..., description="IP pattern or CIDR")
    reason: constr(min_length=1, max_length=500) = Field(..., description="Rule reason")
    expires_at: Optional[datetime] = Field(None, description="Rule expiration date")
    is_active: bool = Field(default=True, description="Rule active status")
    
    @validator('pattern')
    def validate_ip_pattern(cls, v, values):
        rule_type = values.get('rule_type')
        
        if rule_type in ['whitelist', 'blacklist']:
            # Validate IP address or CIDR
            import ipaddress
            try:
                if '/' in v:
                    ipaddress.ip_network(v, strict=False)
                else:
                    ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("Invalid IP address or CIDR notation")
        
        elif rule_type == 'geo_block':
            # Validate country code
            if not re.match(r'^[A-Z]{2}$', v):
                raise ValueError("Invalid country code (must be 2-letter ISO code)")
        
        elif rule_type == 'asn_block':
            # Validate ASN
            if not re.match(r'^AS\d+$', v):
                raise ValueError("Invalid ASN format (must be ASxxxxx)")
        
        return v


# Pagination and filtering schemas
class PaginationRequest(BaseModel):
    """Pagination request schema."""
    
    page: conint(ge=1) = Field(default=1, description="Page number")
    page_size: conint(ge=1, le=1000) = Field(default=20, description="Page size")


class TradeHistoryQueryRequest(PaginationRequest):
    """Trade history query schema."""
    
    symbol: Optional[StockSymbol] = Field(None, description="Filter by symbol")
    start_date: Optional[date] = Field(None, description="Start date filter")
    end_date: Optional[date] = Field(None, description="End date filter")
    order_side: Optional[OrderSideEnum] = Field(None, description="Filter by order side")
    min_amount: Optional[MonetaryAmount] = Field(None, description="Minimum trade amount")
    max_amount: Optional[MonetaryAmount] = Field(None, description="Maximum trade amount")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v and v <= start_date:
            raise ValueError("End date must be after start date")
        return v


class PortfolioAnalysisRequest(BaseModel):
    """Portfolio analysis request schema."""
    
    analysis_type: constr(regex=r'^(performance|risk|allocation|attribution)$') = Field(
        ..., 
        description="Type of analysis to perform"
    )
    start_date: date = Field(..., description="Analysis start date")
    end_date: date = Field(..., description="Analysis end date")
    benchmark_symbol: Optional[StockSymbol] = Field(None, description="Benchmark symbol for comparison")
    include_costs: bool = Field(default=True, description="Include transaction costs in analysis")
    
    @validator('end_date')
    def validate_analysis_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("End date must be after start date")
        
        # Ensure reasonable analysis period
        if start_date and (v - start_date).days > 3650:  # 10 years
            raise ValueError("Analysis period cannot exceed 10 years")
        
        return v


# File upload schemas
class FileUploadRequest(BaseModel):
    """File upload request schema."""
    
    file_type: constr(regex=r'^(csv|json|xlsx|pdf)$') = Field(..., description="File type")
    description: Optional[constr(max_length=500)] = Field(None, description="File description")
    category: constr(regex=r'^(portfolio|trades|research|reports)$') = Field(
        ..., 
        description="File category"
    )
    
    @validator('description')
    def validate_file_description(cls, v):
        if v is not None:
            result = input_validator.validate_string(
                v,
                field_name="description",
                max_length=500,
                check_xss=True,
                check_sql=True
            )
            if not result.is_valid:
                raise ValueError('; '.join(result.errors))
            return result.sanitized_value
        return v


# Response schemas
class ValidationErrorResponse(BaseModel):
    """Validation error response schema."""
    
    error: str = Field(..., description="Error message")
    details: List[str] = Field(..., description="Validation error details")
    timestamp: float = Field(..., description="Error timestamp")
    type: str = Field(default="validation_error", description="Error type")


class SuccessResponse(BaseModel):
    """Generic success response schema."""
    
    success: bool = Field(default=True, description="Success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: float = Field(..., description="Response timestamp")


# Custom validators for common patterns
def validate_trading_symbol(symbol: str) -> str:
    """Validate trading symbol format."""
    result = input_validator.validate_stock_symbol(symbol)
    if not result.is_valid:
        raise ValueError('; '.join(result.errors))
    return result.sanitized_value


def validate_monetary_value(value: Union[str, int, float, Decimal]) -> Decimal:
    """Validate monetary value."""
    result = input_validator.validate_decimal(
        value,
        min_value=Decimal('0.01'),
        max_value=Decimal('1000000000'),
        max_decimal_places=8
    )
    if not result.is_valid:
        raise ValueError('; '.join(result.errors))
    return result.sanitized_value


def validate_percentage_value(value: Union[str, int, float, Decimal]) -> Decimal:
    """Validate percentage value."""
    result = input_validator.validate_percentage(value)
    if not result.is_valid:
        raise ValueError('; '.join(result.errors))
    return result.sanitized_value