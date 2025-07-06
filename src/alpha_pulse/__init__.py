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
AlphaPulse package initialization.
"""

# Version of the alpha_pulse package
__version__ = "1.9.0.0"

# Import core modules
from . import config
from . import data_pipeline
from . import features
from . import exchanges

# Instead, expose specific components that are needed
__all__ = [
    'config',
    'data_pipeline',
    'features',
    'exchanges',
]