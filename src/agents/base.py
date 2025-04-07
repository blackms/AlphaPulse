#!/usr/bin/env python3
"""
Definizioni base per gli agenti di trading AlphaPulse.
Include dataclass comuni e la classe base astratta per gli agenti.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

# Configurazione logging (può essere centralizzata altrove)
# logger = logging.getLogger("AgentBase")

# --- Dataclass Comuni ---

class SignalDirection(Enum):
    """Direzione del segnale di trading."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    SHORT = "short" # Non usato in questo backtest S&P 500
    COVER = "cover" # Non usato in questo backtest S&P 500

@dataclass
class MarketData:
    """Container per i dati di mercato utilizzati dagli agenti."""
    prices: pd.DataFrame  # Dati storici dei prezzi (OHLCV)
    volumes: Optional[pd.DataFrame] = None  # Dati dei volumi (se separati)
    fundamentals: Optional[Dict[str, Any]] = None  # Dati fondamentali aggregati
    sentiment: Optional[pd.DataFrame] = None  # DataFrame di sentiment giornaliero
    economic: Optional[Dict[str, float]] = None # Ultimi indicatori economici
    technical_indicators: Optional[Dict[str, pd.DataFrame]] = None # Precalcolati se necessario
    timestamp: datetime = field(default_factory=datetime.now) # Timestamp corrente del backtest
    # Rimosso data_by_symbol per semplificare, gli agenti useranno prices/volumes

@dataclass
class TradeSignal:
    """Segnale di trading generato da un agente."""
    agent_id: str  # Identificativo dell'agente
    symbol: str  # Simbolo di trading (es. "^GSPC")
    direction: SignalDirection  # Direzione del segnale
    confidence: float  # Punteggio di confidenza (0-1)
    timestamp: datetime  # Timestamp di generazione
    target_price: Optional[float] = None  # Prezzo target (opzionale)
    stop_loss: Optional[float] = None  # Livello di stop loss (opzionale)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadati aggiuntivi

@dataclass
class AgentMetrics:
    """Metriche di performance per un agente."""
    signal_accuracy: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_signal: float = 0.0
    total_signals: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

# --- Classe Base Agente ---

class BaseTradeAgent:
    """Implementazione base dell'agente di trading."""

    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """
        Inizializza l'agente di trading base.

        Args:
            agent_id: Identificativo univoco per questo agente
            config: Parametri di configurazione opzionali
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.metrics: Optional[AgentMetrics] = None
        self._signal_history: List[TradeSignal] = []
        self.logger = logging.getLogger(f"Agent_{agent_id}") # Logger specifico per agente

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Inizializza l'agente con la configurazione."""
        self.config.update(config)
        self.logger.info(f"Agente {self.agent_id} inizializzato.")

    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Genera segnali di trading basati sui dati di mercato.
        Metodo astratto da implementare nelle sottoclassi.

        Args:
            market_data: Dati di mercato correnti

        Returns:
            Lista di segnali di trading
        """
        raise NotImplementedError("Le sottoclassi devono implementare generate_signals")

    async def get_confidence_level(self) -> float:
        """
        Ottiene il livello di confidenza corrente dell'agente basato sull'accuratezza.

        Returns:
            Punteggio di confidenza tra 0 e 1
        """
        if not self.metrics:
            return 0.5  # Confidenza di default
        # Usa win_rate invece di signal_accuracy se più rappresentativo
        return min(max(self.metrics.win_rate, 0.0), 1.0)

    async def validate_signal(self, signal: TradeSignal) -> bool:
        """
        Valida un segnale di trading generato.

        Args:
            signal: Segnale di trading da validare

        Returns:
            True se il segnale è valido, False altrimenti
        """
        if not signal.symbol or not signal.direction:
            self.logger.warning(f"Segnale invalido: simbolo o direzione mancante: {signal}")
            return False
        if not 0 <= signal.confidence <= 1:
            self.logger.warning(f"Segnale invalido: confidenza fuori range [0, 1]: {signal.confidence}")
            return False
        # Aggiungere altri controlli se necessario (es. target/stop loss sensati)
        return True

    async def update_metrics(self, performance_data: pd.DataFrame) -> AgentMetrics:
        """
        Aggiorna le metriche dell'agente con i dati di performance dei segnali.

        Args:
            performance_data: DataFrame con colonne ['profit', ...] indicizzate per timestamp

        Returns:
            Metriche dell'agente aggiornate
        """
        if performance_data.empty or 'profit' not in performance_data.columns:
             self.logger.warning("Dati di performance vuoti o mancanti della colonna 'profit'. Metriche non aggiornate.")
             # Restituisce metriche precedenti o default se non ci sono
             return self.metrics or AgentMetrics()

        try:
            # Calcola metriche di base
            total_signals = len(performance_data)
            profitable_trades = performance_data[performance_data['profit'] > 0]
            losing_trades = performance_data[performance_data['profit'] < 0]

            win_rate = len(profitable_trades) / total_signals if total_signals > 0 else 0.0

            gross_profit = abs(profitable_trades['profit'].sum())
            gross_loss = abs(losing_trades['profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

            # Calcola Sharpe ratio (assumendo rendimenti giornalieri)
            returns = performance_data['profit'] / self.config.get('initial_capital_per_trade', 10000) # Stima rendimenti
            sharpe_ratio = 0.0
            if len(returns) > 1 and returns.std() != 0:
                 # Assumendo risk-free rate = 0
                 sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) # Annualizzato

            # Calcola drawdown
            equity_curve = (1 + returns).cumprod()
            rolling_max = equity_curve.expanding().max()
            drawdowns = (equity_curve - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0.0

            self.metrics = AgentMetrics(
                signal_accuracy=win_rate, # Usiamo win_rate come proxy
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_profit_per_signal=performance_data['profit'].mean(),
                total_signals=total_signals,
                timestamp=datetime.now()
            )
            self.logger.info(f"Metriche aggiornate per {self.agent_id}: WinRate={win_rate:.2%}, PF={profit_factor:.2f}, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2%}")

        except Exception as e:
            self.logger.error(f"Errore durante l'aggiornamento delle metriche: {e}")
            # Non aggiornare le metriche in caso di errore, mantieni le precedenti
            return self.metrics or AgentMetrics()

        return self.metrics

    async def adapt_parameters(self, metrics: AgentMetrics) -> None:
        """
        Adatta i parametri dell'agente in base alle metriche di performance.
        Metodo opzionale da implementare nelle sottoclassi se necessario.

        Args:
            metrics: Metriche di performance correnti
        """
        # Esempio base: non fa nulla, ma le sottoclassi possono sovrascriverlo
        self.logger.debug(f"Metodo adapt_parameters chiamato per {self.agent_id}, nessuna azione di default.")
        pass