#!/usr/bin/env python3
"""
Agente di Analisi del Sentiment per S&P 500.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Importa definizioni base
from .base import BaseTradeAgent, MarketData, TradeSignal, SignalDirection

# Configurazione logging specifica per questo agente
logger = logging.getLogger("EquitySentimentAgent")

class EquitySentimentAgent(BaseTradeAgent):
    """
    Agente di analisi del sentiment adattato per il mercato azionario.
    Analizza il sentiment aggregato da notizie o altre fonti.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Inizializza l'agente di analisi del sentiment."""
        super().__init__("equity_sentiment_agent", config)
        # Pesi per diverse fonti (attualmente solo news implementato)
        self.weights = {
            'news': self.config.get("news_weight", 1.0), # Peso 1.0 se unica fonte
            # 'social': self.config.get("social_weight", 0.0),
            # 'analyst': self.config.get("analyst_weight", 0.0)
        }
        self.lookback_days = self.config.get("lookback_days", 7)
        self.signal_threshold = self.config.get("signal_threshold", 0.15)
        self.confidence_multiplier = self.config.get("confidence_multiplier", 1.5)
        self.target_multiplier = self.config.get("target_multiplier", 0.3)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.03) # 3% stop loss

        self.logger.info(f"Lookback days: {self.lookback_days}, Signal threshold: {self.signal_threshold}")
        self.logger.info(f"Confidence multiplier: {self.confidence_multiplier}, Target multiplier: {self.target_multiplier}, Stop loss %: {self.stop_loss_pct}")


    async def initialize(self, config: Dict[str, Any]) -> None:
        """Inizializza l'agente con la configurazione."""
        await super().initialize(config)
        # Potrebbe caricare modelli NLP o dati storici qui se necessario

    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Genera segnali di trading basati sull'analisi del sentiment.

        Args:
            market_data: Dati di mercato inclusi dati di sentiment (DataFrame giornaliero)

        Returns:
            Lista di segnali di trading
        """
        signals = []
        symbol = "^GSPC" # Specifico per questo backtest

        # Verifica se sono disponibili dati di sentiment
        # market_data.sentiment qui è un DataFrame aggregato giornaliero
        # Modificato per accettare anche dizionari (come da implementazione precedente)
        sentiment_input = market_data.sentiment
        if sentiment_input is None or (isinstance(sentiment_input, (pd.DataFrame, pd.Series)) and sentiment_input.empty) or (isinstance(sentiment_input, dict) and not sentiment_input):
            self.logger.warning("Nessun dato di sentiment disponibile per l'analisi.")
            return signals

        try:
            # Analizza i dati di sentiment
            sentiment_score = await self._analyze_sentiment(sentiment_input)

            if abs(sentiment_score) > self.signal_threshold:
                direction = SignalDirection.BUY if sentiment_score > 0 else SignalDirection.SELL
                confidence = min(abs(sentiment_score) * self.confidence_multiplier, 0.9) # Limita confidenza

                # Ottieni prezzo corrente
                if market_data.prices.empty:
                     self.logger.warning("Prezzi mancanti, impossibile generare segnale.")
                     return signals
                current_price = market_data.prices['Close'].iloc[-1]

                # Calcola prezzo target e stop loss
                if direction == SignalDirection.BUY:
                    target_price = current_price * (1 + abs(sentiment_score) * self.target_multiplier)
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                else: # SELL
                    target_price = current_price * (1 - abs(sentiment_score) * self.target_multiplier)
                    stop_loss = current_price * (1 + self.stop_loss_pct)

                # Crea segnale
                signal = TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    timestamp=market_data.timestamp, # Usa il timestamp dei dati
                    target_price=target_price,
                    stop_loss=stop_loss,
                    metadata={
                        "strategy": "sentiment",
                        "sentiment_score": sentiment_score,
                    }
                )

                # Valida il segnale prima di aggiungerlo
                if await self.validate_signal(signal):
                    signals.append(signal)
                    self.logger.info(f"Generato segnale di sentiment per {symbol}: {direction.value} con confidenza {confidence:.2f}")
                else:
                    self.logger.warning(f"Segnale di sentiment generato ma non valido: {signal}")

            else:
                self.logger.debug(f"Nessun segnale di sentiment generato per {symbol} (punteggio: {sentiment_score:.2f})")

        except Exception as e:
            self.logger.error(f"Errore nella generazione dei segnali di sentiment: {str(e)}", exc_info=True)

        return signals

    async def _analyze_sentiment(self, sentiment_data: Union[pd.DataFrame, pd.Series, Dict[str, float]]) -> float:
        """
        Analizza i dati di sentiment (DataFrame, Series o Dict).

        Args:
            sentiment_data: Dati di sentiment. Può essere:
                             - DataFrame con indice DatetimeIndex e colonna 'sentiment_score'
                             - Series con indice DatetimeIndex
                             - Dict {data_str: punteggio}

        Returns:
            Punteggio di sentiment aggregato (-1 a 1)
        """
        if sentiment_data is None: return 0.0
        if isinstance(sentiment_data, dict) and not sentiment_data: return 0.0
        if isinstance(sentiment_data, (pd.DataFrame, pd.Series)) and sentiment_data.empty: return 0.0

        try:
            sentiment_series = None
            # Converti diversi formati in Series con DatetimeIndex
            if isinstance(sentiment_data, pd.DataFrame):
                 if 'sentiment_score' not in sentiment_data.columns:
                     self.logger.error("Colonna 'sentiment_score' mancante nel DataFrame di sentiment.")
                     return 0.0
                 sentiment_series = sentiment_data['sentiment_score']
                 if not isinstance(sentiment_series.index, pd.DatetimeIndex):
                      sentiment_series.index = pd.to_datetime(sentiment_series.index)

            elif isinstance(sentiment_data, pd.Series):
                 sentiment_series = sentiment_data
                 if not isinstance(sentiment_series.index, pd.DatetimeIndex):
                      sentiment_series.index = pd.to_datetime(sentiment_series.index)

            elif isinstance(sentiment_data, dict):
                 sentiment_series = pd.Series(sentiment_data)
                 sentiment_series.index = pd.to_datetime(sentiment_series.index)

            else:
                 self.logger.warning(f"Formato dati sentiment non riconosciuto: {type(sentiment_data)}")
                 return 0.0

            sentiment_series = sentiment_series.sort_index().dropna() # Ordina e rimuovi NaN

            if sentiment_series.empty:
                self.logger.debug("Nessun dato di sentiment valido dopo la pulizia.")
                return 0.0

            # Prendi gli ultimi N giorni
            last_ts = sentiment_series.index.max()
            start_lookback = last_ts - pd.Timedelta(days=self.lookback_days)
            recent_sentiment = sentiment_series[sentiment_series.index >= start_lookback]

            if recent_sentiment.empty:
                self.logger.debug("Nessun dato di sentiment nel periodo di lookback.")
                return 0.0

            # Calcola media ponderata temporalmente
            # Pesi lineari crescenti (dati più recenti pesano di più)
            weights = np.linspace(0.5, 1.5, len(recent_sentiment))
            avg_sentiment = np.average(recent_sentiment.values, weights=weights)

            # Normalizza o limita il punteggio se necessario
            avg_sentiment = np.clip(avg_sentiment, -1.0, 1.0)
            self.logger.debug(f"Punteggio sentiment calcolato: {avg_sentiment:.4f}")

        except Exception as e:
            self.logger.error(f"Errore nell'elaborazione dei dati di sentiment: {e}. Restituisco 0.", exc_info=True)
            avg_sentiment = 0.0

        # Assicura che non ritorni NaN
        return avg_sentiment if not np.isnan(avg_sentiment) else 0.0