# AlphaPulse: AI-Driven Hedge Fund System
# Copyright (C) 2024 AlphaPulse Trading System
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Base types for exchange operations.
"""
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional


@dataclass
class Balance:
    """Account balance information."""
    total: Decimal
    available: Decimal
    locked: Decimal

    @property
    def free(self) -> Decimal:
        """Free balance for trading."""
        return self.available


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    exchange: str = ""
    symbol: str = ""
    timeframe: str = ""

    @classmethod
    def from_list(cls, data: list) -> 'OHLCV':
        """Create OHLCV from CCXT list format."""
        return cls(
            timestamp=datetime.fromtimestamp(data[0] / 1000),  # Convert from milliseconds
            open=Decimal(str(data[1])),
            high=Decimal(str(data[2])),
            low=Decimal(str(data[3])),
            close=Decimal(str(data[4])),
            volume=Decimal(str(data[5]))
        )