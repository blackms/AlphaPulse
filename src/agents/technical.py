#!/usr/bin/env python3
"""
Agente di Analisi Tecnica per S&P 500 - Strategia Golden Cross/MA200 Filter.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Importa definizioni base
from .base import BaseTradeAgent, MarketData, TradeSignal, SignalDirection

# Configurazione logging specifica per questo agente
logger = logging.getLogger("EquityTechnicalAgent_MA_Cross")

# Funzione helper per ottenere scalare da Series/DataFrame .iloc[-1]
def _get_scalar(series_or_df_iloc):
    """Estrae un valore scalare dall'output di iloc[-1], gestendo Series/DataFrame e NaN."""
    if isinstance(series_or_df_iloc, (pd.Series, pd.DataFrame)):
        # Usa .item() se ha un solo elemento, altrimenti NaN
        return series_or_df_iloc.item() if series_or_df_iloc.size == 1 else np.nan
    elif isinstance(series_or_df_iloc, (int, float, np.number)):
        # Già uno scalare
        return series_or_df_iloc
    else:
        # Tipo non gestito o vuoto
        return np.nan

class EquityTechnicalAgent(BaseTradeAgent):
    """
    Agente tecnico che implementa una strategia Golden Cross (SMA50/SMA200)
    con filtro SMA200 per il trend principale.
    - Entra Long: Prezzo > SMA200 E SMA50 > SMA200
    - Esce Long: Prezzo < SMA200 O SMA50 < SMA200
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Inizializza l'agente."""
        super().__init__("equity_technical_ma_cross", config) # Nuovo ID agente
        self.sma_short_period = self.config.get("sma_short", 50)
        self.sma_long_period = self.config.get("sma_long", 200)
        self.fixed_confidence = self.config.get("fixed_confidence", 0.9) # Confidenza fissa per i segnali BUY
        self.required_lookback = self.sma_long_period + 1 # Dati necessari
        self.logger.info(f"Strategia MA Cross: SMA Short={self.sma_short_period}, SMA Long={self.sma_long_period}, Confidence={self.fixed_confidence}")

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Inizializza l'agente con la configurazione."""
        await super().initialize(config)
        # Aggiorna parametri se specificati nella config passata
        self.sma_short_period = config.get("sma_short", self.sma_short_period)
        self.sma_long_period = config.get("sma_long", self.sma_long_period)
        self.fixed_confidence = config.get("fixed_confidence", self.fixed_confidence)
        self.required_lookback = self.sma_long_period + 1

    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Genera segnali BUY o SELL basati sulla strategia MA Cross.
        """
        signals = []
        symbol = "^GSPC" # Specifico per questo backtest

        if not isinstance(market_data.prices, pd.DataFrame) or market_data.prices.empty:
            self.logger.warning("Nessun dato di prezzo disponibile")
            return signals

        # Estrai i dati di prezzo (solo 'Close' necessario)
        if 'Close' not in market_data.prices.columns:
             self.logger.error("Colonna 'Close' mancante nei dati di prezzo.")
             return signals
        prices = market_data.prices['Close']

        self.logger.debug(f"Analisi MA Cross per {symbol} con {len(prices)} punti dati per timestamp {market_data.timestamp}")

        if len(prices) < self.required_lookback:
            self.logger.warning(f"Dati insufficienti per {symbol} ({len(prices)} < {self.required_lookback})")
            return signals

        try:
            # Calcola Medie Mobili
            sma_short = prices.rolling(window=self.sma_short_period).mean()
            sma_long = prices.rolling(window=self.sma_long_period).mean()

            # Estrai valori correnti come scalari
            current_price = _get_scalar(prices.iloc[-1])
            current_sma_short = _get_scalar(sma_short.iloc[-1])
            current_sma_long = _get_scalar(sma_long.iloc[-1])

            if np.isnan(current_price) or np.isnan(current_sma_short) or np.isnan(current_sma_long):
                self.logger.warning(f"Valori NaN incontrati per {symbol} a {market_data.timestamp}, impossibile generare segnale.")
                return signals

            # Logica di segnale
            is_above_sma200 = current_price > current_sma_long
            is_golden_cross = current_sma_short > current_sma_long

            signal_direction = SignalDirection.HOLD # Default

            # Condizione di Ingresso Long
            if is_above_sma200 and is_golden_cross:
                signal_direction = SignalDirection.BUY
                self.logger.debug(f"BUY Signal Triggered: Price > SMA200 and SMA50 > SMA200")

            # Condizione di Uscita Long (se prezzo sotto SMA200 o Death Cross)
            elif (not is_above_sma200) or (not is_golden_cross):
                 signal_direction = SignalDirection.SELL
                 self.logger.debug(f"SELL Signal Triggered: Price < SMA200 or SMA50 < SMA200")


            # Genera il segnale se non è HOLD
            if signal_direction != SignalDirection.HOLD:
                 signal = TradeSignal(
                     agent_id=self.agent_id,
                     symbol=symbol,
                     direction=signal_direction,
                     confidence=self.fixed_confidence if signal_direction == SignalDirection.BUY else 1.0, # Usa confidenza fissa per BUY, massima per SELL (chiudi tutto)
                     timestamp=market_data.timestamp,
                     metadata={
                         "strategy": "MA_Cross_Filter",
                         "price": current_price,
                         f"sma_{self.sma_short_period}": current_sma_short,
                         f"sma_{self.sma_long_period}": current_sma_long,
                     }
                 )
                 if await self.validate_signal(signal):
                     signals.append(signal)
                     self.logger.info(f"Generato segnale {signal.direction.value} per {symbol} @ {current_price:.2f}")

        except Exception as e:
            self.logger.error(f"Errore nella generazione dei segnali MA Cross per {symbol}: {str(e)}", exc_info=True)

        return signals