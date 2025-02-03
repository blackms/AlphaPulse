"""
Grid-based hedging strategy implementation with advanced risk management.
"""
from decimal import Decimal
from typing import Dict, Optional
from loguru import logger

from alpha_pulse.execution.broker_interface import BrokerInterface
from alpha_pulse.data_pipeline.exchange_data_provider import ExchangeDataProvider

from .grid_calculator import DefaultGridCalculator
from .interfaces import GridCalculator, OrderManager, RiskManager, StateManager
from .models import GridState, MarketState
from .order_manager import GridOrderManager
from .risk_manager import GridRiskManager
from .state_manager import GridStateManager


class GridHedgeBot:
    """
    Implements a grid-based hedging strategy with advanced risk management.
    Coordinates multiple specialized components following SOLID principles.
    """
    
    def __init__(
        self,
        broker: BrokerInterface,
        data_provider: ExchangeDataProvider,
        symbol: str,
        grid_calculator: Optional[GridCalculator] = None,
        order_manager: Optional[OrderManager] = None,
        risk_manager: Optional[RiskManager] = None,
        state_manager: Optional[StateManager] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize grid hedging bot.
        
        Args:
            broker: Trading broker interface
            data_provider: Market data provider
            symbol: Trading symbol
            grid_calculator: Optional custom grid calculator
            order_manager: Optional custom order manager
            risk_manager: Optional custom risk manager
            state_manager: Optional custom state manager
            config: Optional configuration parameters
        """
        self.symbol = symbol
        self.data_provider = data_provider
        
        # Load configuration
        self.config = config or {}
        self._load_config()
        
        # Initialize components
        self.risk_manager = risk_manager or GridRiskManager(
            max_position_size=self.max_position_size,
            max_drawdown=self.max_drawdown,
            base_stop_loss=self.stop_loss_pct,
            var_limit=self.var_limit
        )
        
        self.grid_calculator = grid_calculator or DefaultGridCalculator(
            grid_spacing_pct=self.grid_spacing,
            num_levels=self.num_levels,
            min_price_distance=self.min_price_distance,
            max_position_size=self.max_position_size
        )
        
        self.order_manager = order_manager or GridOrderManager(
            broker=broker,
            risk_manager=self.risk_manager,
            symbol=symbol,
            max_active_orders=self.max_active_orders
        )
        
        self.state_manager = state_manager or GridStateManager(
            symbol=symbol
        )
        
        logger.info(
            f"Initialized GridHedgeBot for {symbol} with "
            f"{self.num_levels} levels"
        )
        
    def _load_config(self) -> None:
        """Load and validate configuration."""
        # Grid parameters
        self.grid_spacing = Decimal(str(
            self.config.get('grid_spacing_pct', '0.01')  # 1% default
        ))
        self.num_levels = int(self.config.get('num_levels', 5))
        self.min_price_distance = Decimal(str(
            self.config.get('min_price_distance', '1.0')
        ))
        
        # Risk parameters
        self.max_position_size = Decimal(str(
            self.config.get('max_position_size', '1.0')
        ))
        self.max_drawdown = Decimal(str(
            self.config.get('max_drawdown', '0.1')  # 10% default
        ))
        self.stop_loss_pct = Decimal(str(
            self.config.get('stop_loss_pct', '0.04')  # 4% default
        ))
        self.var_limit = Decimal(str(
            self.config.get('var_limit', '10000')  # $10k default
        ))
        
        # Execution parameters
        self.max_active_orders = int(self.config.get('max_active_orders', 50))
        self.rebalance_interval = int(
            self.config.get('rebalance_interval_seconds', 60)
        )
        
        logger.info("Loaded configuration parameters")
        
    async def start(self) -> None:
        """Start the grid hedging strategy."""
        try:
            # Initialize market data
            current_price = await self.data_provider.get_current_price(self.symbol)
            if not current_price:
                raise ValueError(f"Could not get price for {self.symbol}")
                
            market_state = MarketState.from_raw(
                price=current_price,
                volatility=0.02  # Default volatility
            )
            
            # Calculate initial grid levels
            levels = self.grid_calculator.calculate_grid_levels(
                market=market_state,
                position=self.state_manager.position
            )
            
            # Place initial orders
            await self.order_manager.place_grid_orders(
                levels=levels,
                market=market_state
            )
            
            # Update state
            self.state_manager.update_state(GridState.ACTIVE)
            logger.info("Grid hedging strategy started")
            
        except Exception as e:
            logger.error(f"Error starting grid strategy: {str(e)}")
            self.state_manager.update_state(GridState.ERROR)
            raise
            
    async def stop(self) -> None:
        """Stop the grid hedging strategy."""
        try:
            # Cancel all active orders
            active_orders = self.order_manager.get_active_orders()
            await self.order_manager.cancel_orders(list(active_orders.keys()))
            
            # Update state
            self.state_manager.update_state(GridState.STOPPED)
            logger.info("Grid hedging strategy stopped")
            
        except Exception as e:
            logger.error(f"Error stopping grid strategy: {str(e)}")
            self.state_manager.update_state(GridState.ERROR)
            raise
            
    async def execute(self, current_price: float) -> None:
        """
        Execute one iteration of the grid strategy.
        
        Args:
            current_price: Current market price
        """
        if self.state_manager.state != GridState.ACTIVE:
            logger.warning(
                f"Grid bot not active. Current state: {self.state_manager.state}"
            )
            return
            
        try:
            # Create market state
            market_state = MarketState.from_raw(
                price=current_price,
                volatility=0.02  # TODO: Calculate from historical data
            )
            
            # Check risk limits
            if not self.risk_manager.validate_risk_limits(
                position=self.state_manager.position,
                market=market_state
            ):
                logger.warning("Risk limits exceeded")
                await self.stop()
                return
                
            # Update risk orders
            await self.order_manager.update_risk_orders(
                position=self.state_manager.position,
                market=market_state
            )
            
            # Calculate and place grid orders
            levels = self.grid_calculator.calculate_grid_levels(
                market=market_state,
                position=self.state_manager.position
            )
            
            await self.order_manager.place_grid_orders(
                levels=levels,
                market=market_state
            )
            
            # Update state
            self.state_manager.record_rebalance()
            
        except Exception as e:
            logger.error(f"Error executing grid strategy: {str(e)}")
            self.state_manager.update_state(GridState.ERROR)
            
    def get_status(self) -> Dict:
        """Get current strategy status."""
        return self.state_manager.get_status()
        
    @classmethod
    async def create(
        cls,
        broker: BrokerInterface,
        symbol: str,
        config: Optional[Dict] = None
    ) -> 'GridHedgeBot':
        """
        Create and initialize a new grid hedging bot.
        
        Args:
            broker: Trading broker interface
            symbol: Trading symbol
            config: Optional configuration parameters
            
        Returns:
            Initialized GridHedgeBot instance
        """
        # Create data provider
        data_provider = ExchangeDataProvider()
        await data_provider.initialize()
        
        # Create bot instance
        bot = cls(
            broker=broker,
            data_provider=data_provider,
            symbol=symbol,
            config=config
        )
        
        # Start strategy
        await bot.start()
        
        return bot
