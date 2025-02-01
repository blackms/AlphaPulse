from typing import Protocol, List, Optional
from decimal import Decimal
from .models import SpotPosition, FuturesPosition, HedgeRecommendation

class IHedgeAnalyzer(Protocol):
    """Interface for hedge analysis implementations."""
    
    async def analyze(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> HedgeRecommendation:
        """
        Analyze current positions and provide hedging recommendations.
        
        This method can be implemented as either sync or async. The HedgeManager
        will handle both cases appropriately.
        
        Args:
            spot_positions: List of current spot positions
            futures_positions: List of current futures positions
            
        Returns:
            HedgeRecommendation containing analysis results and suggested adjustments
        """
        ...

    def calculate_net_exposure(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> Decimal:
        """
        Calculate the net exposure across spot and futures positions.
        
        Args:
            spot_positions: List of current spot positions
            futures_positions: List of current futures positions
            
        Returns:
            Float representing the net exposure in base currency
        """
        ...

    def evaluate_hedge_effectiveness(
        self,
        spot_positions: List[SpotPosition],
        futures_positions: List[FuturesPosition]
    ) -> dict:
        """
        Evaluate how effectively the futures positions are hedging spot exposure.
        
        Args:
            spot_positions: List of current spot positions
            futures_positions: List of current futures positions
            
        Returns:
            Dictionary containing hedge effectiveness metrics
        """
        ...

class IPositionFetcher(Protocol):
    """Interface for fetching position data."""
    
    async def get_spot_positions(self) -> List[SpotPosition]:
        """Fetch current spot positions."""
        ...
    
    async def get_futures_positions(self) -> List[FuturesPosition]:
        """Fetch current futures positions."""
        ...
    
    async def get_current_price(
        self,
        symbol: str,
        is_futures: bool = False
    ) -> Optional[Decimal]:
        """Get current price for a symbol."""
        ...

class IOrderExecutor(Protocol):
    """Interface for executing orders."""
    
    async def place_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET"
    ) -> str:
        """Place a futures order."""
        ...
    
    async def close_position(
        self,
        symbol: str,
        position_side: str,
        quantity: float
    ) -> str:
        """Close an existing position."""
        ...

class IExecutionStrategy(Protocol):
    """Interface for hedge execution strategies."""
    
    async def execute_hedge_adjustments(
        self,
        recommendations: HedgeRecommendation,
        executor: IOrderExecutor
    ) -> bool:
        """Execute hedge adjustments according to strategy."""
        ...