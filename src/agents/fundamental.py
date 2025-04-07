#!/usr/bin/env python3
"""
Agente di Analisi Fondamentale per S&P 500.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Importa definizioni base
from .base import BaseTradeAgent, MarketData, TradeSignal, SignalDirection

# Configurazione logging specifica per questo agente
logger = logging.getLogger("EquityFundamentalAgent")

class EquityFundamentalAgent(BaseTradeAgent):
    """
    Agente di analisi fondamentale adattato per il mercato azionario.
    Analizza metriche fondamentali aggregate per l'indice S&P 500.
    Nota: L'analisi fondamentale su un indice è meno diretta che su singole azioni.
    Questo agente userà metriche aggregate o medie/mediane dei componenti.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Inizializza l'agente di analisi fondamentale."""
        super().__init__("equity_fundamental_agent", config)
        # Pesi per le metriche aggregate dell'indice
        self.indicators = {
            'pe_ratio': self.config.get("pe_ratio_weight", 0.25),
            'pb_ratio': self.config.get("pb_ratio_weight", 0.20),
            'dividend_yield': self.config.get("dividend_yield_weight", 0.15),
            'earnings_growth': self.config.get("earnings_growth_weight", 0.20), # Esempio: crescita utili
            'revenue_growth': self.config.get("revenue_growth_weight", 0.20)  # Esempio: crescita ricavi
        }
        self.logger.info(f"Indicator weights: {self.indicators}")

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Inizializza l'agente con la configurazione."""
        await super().initialize(config)
        # Potrebbe caricare dati storici delle metriche fondamentali aggregate qui

    async def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Genera segnali di trading basati sull'analisi fondamentale aggregata.

        Args:
            market_data: Dati di mercato inclusi dati fondamentali aggregati

        Returns:
            Lista di segnali di trading
        """
        signals = []
        symbol = "^GSPC" # Specifico per questo backtest

        # Verifica se sono disponibili dati fondamentali aggregati
        # market_data.fundamentals dovrebbe contenere metriche come P/E medio, Yield medio, etc.
        if not market_data.fundamentals:
            self.logger.warning("Nessun dato fondamentale aggregato disponibile per l'analisi.")
            return signals

        try:
            # Analizza i dati fondamentali aggregati
            fundamental_score = await self._analyze_fundamentals(market_data.fundamentals)

            # Soglia minima per generare un segnale
            signal_threshold = self.config.get("signal_threshold", 0.2)
            if abs(fundamental_score) > signal_threshold:
                direction = SignalDirection.BUY if fundamental_score > 0 else SignalDirection.SELL
                confidence = min(abs(fundamental_score), 0.9) # Limita confidenza a 0.9

                # Ottieni prezzo corrente (ultimo prezzo disponibile)
                if market_data.prices.empty:
                     self.logger.warning("Prezzi mancanti, impossibile generare segnale.")
                     return signals
                current_price = market_data.prices['Close'].iloc[-1]

                # Calcola prezzo target e stop loss (semplificato)
                target_multiplier = self.config.get("target_multiplier", 0.5)
                stop_loss_pct = self.config.get("stop_loss_pct", 0.05) # 5% stop loss

                if direction == SignalDirection.BUY:
                    target_price = current_price * (1 + abs(fundamental_score) * target_multiplier)
                    stop_loss = current_price * (1 - stop_loss_pct)
                else: # SELL
                    target_price = current_price * (1 - abs(fundamental_score) * target_multiplier)
                    stop_loss = current_price * (1 + stop_loss_pct)

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
                        "strategy": "fundamental",
                        "fundamental_score": fundamental_score,
                        "indicators": market_data.fundamentals # Includi i dati usati
                    }
                )

                # Valida il segnale prima di aggiungerlo
                if await self.validate_signal(signal):
                    signals.append(signal)
                    self.logger.info(f"Generato segnale fondamentale per {symbol}: {direction.value} con confidenza {confidence:.2f}")
                else:
                    self.logger.warning(f"Segnale fondamentale generato ma non valido: {signal}")

            else:
                self.logger.debug(f"Nessun segnale fondamentale generato per {symbol} (punteggio: {fundamental_score:.2f})")

        except Exception as e:
            self.logger.error(f"Errore nella generazione dei segnali fondamentali: {str(e)}", exc_info=True)

        return signals

    async def _analyze_fundamentals(self, fundamentals: Dict[str, Any]) -> float:
        """
        Analizza i dati fondamentali aggregati dell'indice.

        Args:
            fundamentals: Dizionario con metriche fondamentali aggregate (es. P/E, Yield)

        Returns:
            Punteggio fondamentale aggregato (-1 a 1)
        """
        scores = {}

        # Analizza P/E Ratio (rispetto alla media storica o benchmark)
        if 'pe_ratio' in fundamentals:
            pe = fundamentals['pe_ratio']
            historical_avg_pe = self.config.get('historical_avg_pe', 18.0) # Esempio
            # P/E sotto la media è positivo
            if pe is not None and not np.isnan(pe): # Aggiunto controllo None e NaN
                 scores['pe_ratio'] = np.clip((historical_avg_pe - pe) / historical_avg_pe, -1, 1) # Normalizzato
            else: scores['pe_ratio'] = 0.0

        # Analizza P/B Ratio
        if 'pb_ratio' in fundamentals:
            pb = fundamentals['pb_ratio']
            historical_avg_pb = self.config.get('historical_avg_pb', 3.0) # Esempio
            if pb is not None and not np.isnan(pb): # Aggiunto controllo None e NaN
                 scores['pb_ratio'] = np.clip((historical_avg_pb - pb) / historical_avg_pb, -1, 1)
            else: scores['pb_ratio'] = 0.0

        # Analizza Dividend Yield
        if 'dividend_yield' in fundamentals:
            div_yield = fundamentals['dividend_yield']
            historical_avg_yield = self.config.get('historical_avg_yield', 2.0) # Esempio
            if div_yield is not None and not np.isnan(div_yield): # Aggiunto controllo None e NaN
                 # Yield sopra la media è positivo
                 scores['dividend_yield'] = np.clip((div_yield - historical_avg_yield) / historical_avg_yield, -1, 1)
            else: scores['dividend_yield'] = 0.0

        # Analizza Crescita Utili (Earnings Growth)
        if 'earnings_growth' in fundamentals:
            eg = fundamentals['earnings_growth'] # Assumendo % di crescita
            min_expected_growth = self.config.get('min_expected_growth', 5.0) # Esempio 5%
            if eg is not None and not np.isnan(eg): # Aggiunto controllo None e NaN
                 # Crescita sopra le aspettative è positiva
                 scores['earnings_growth'] = np.clip((eg - min_expected_growth) / min_expected_growth, -1, 1)
            else: scores['earnings_growth'] = 0.0

        # Analizza Crescita Ricavi (Revenue Growth)
        if 'revenue_growth' in fundamentals:
            rg = fundamentals['revenue_growth'] # Assumendo % di crescita
            min_expected_rev_growth = self.config.get('min_expected_rev_growth', 3.0) # Esempio 3%
            if rg is not None and not np.isnan(rg): # Aggiunto controllo None e NaN
                 scores['revenue_growth'] = np.clip((rg - min_expected_rev_growth) / min_expected_rev_growth, -1, 1)
            else: scores['revenue_growth'] = 0.0

        # Calcola punteggio medio ponderato
        total_score = 0.0
        total_weight = 0.0

        for indicator, weight in self.indicators.items():
            score = scores.get(indicator)
            # Assicurati che score sia un float valido prima di usarlo
            if score is not None and isinstance(score, (int, float)) and not np.isnan(score):
                total_score += score * weight
                total_weight += weight

        if total_weight <= 1e-9: # Usa tolleranza per divisione per zero
            self.logger.warning("Nessun indicatore fondamentale valido o peso totale nullo.")
            return 0.0

        final_score = total_score / total_weight
        self.logger.debug(f"Punteggio fondamentale calcolato: {final_score:.4f} (basato su {scores})")
        return final_score