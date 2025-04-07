#!/usr/bin/env python3
"""
Framework di Backtesting per S&P 500 con AlphaPulse.

Questo modulo implementa il motore di backtesting che simula le operazioni
di trading basate sui segnali generati dagli agenti AlphaPulse adattati.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg') # Imposta backend non interattivo PRIMA di importare pyplot o usare quantstats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import quantstats as qs
import warnings # Importa modulo warnings

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

# Importa componenti necessari
from src.data_manager import DataManager # Usa import assoluto da src
# Importa agenti dalla nuova posizione
from src.agents.technical import EquityTechnicalAgent
from src.agents.fundamental import EquityFundamentalAgent
from src.agents.sentiment import EquitySentimentAgent
from src.agents.base import ( # Importa definizioni base
    MarketData,
    TradeSignal,
    SignalDirection
)
# Importa altri agenti se necessario (es. Value, Activist se li implementi)
# Importa la nuova strategia
from src.strategies.long_short.long_short_strategy import LongShortStrategyAgent


# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtester.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Backtester")

@dataclass
class Trade:
    """Rappresenta una singola operazione di trading."""
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    quantity: float
    price: float
    commission: float
    slippage: float
    signal_confidence: float
    agent_id: str

@dataclass
class PortfolioSnapshot:
    """Snapshot del portafoglio in un dato momento."""
    timestamp: datetime
    cash: float
    equity_value: float
    total_value: float
    positions: Dict[str, float]  # Simbolo -> Quantità
    pnl: float
    returns: float

class Backtester:
    """
    Motore di backtesting per simulare strategie di trading sull'S&P 500.
    """

    def __init__(self, config: Dict[str, Any], combined_daily_data: Optional[pd.DataFrame] = None):
        """
        Inizializza il Backtester.

        Args:
            config: Configurazione del backtest.
            combined_daily_data: DataFrame pre-fetched containing combined daily data
                                 (e.g., SP500 OHLCV and VIX Close). Optional for backward compatibility.
        """
        self.config = config
        self.all_daily_data = combined_daily_data # Store the pre-fetched data
        # self.data_manager = data_manager # Remove old data_manager dependency

        # Parametri di backtest
        self.start_date = pd.to_datetime(config['backtest']['start_date'])
        self.end_date = pd.to_datetime(config['backtest']['end_date'])
        self.initial_capital = float(config['backtest']['initial_capital'])
        self.benchmark_symbol = config['backtest']['benchmark']

        # Parametri di trading
        self.commission_pct = float(config['backtest']['transaction_costs']['commission_pct'])
        self.slippage_pct = float(config['backtest']['transaction_costs']['slippage_pct'])

        # Agenti verranno inizializzati nel metodo async initialize
        self.agents: Dict[str, BaseTradeAgent] = {}

        # Stato del backtest
        self.current_date: Optional[datetime] = None
        self.cash: float = self.initial_capital
        self.positions: Dict[str, float] = {}  # Simbolo -> Quantità
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.position_stop_loss: Dict[str, Optional[float]] = {} # Simbolo -> Prezzo Stop Loss

        self.trades: List[Trade] = []
        # Calcola max_agent_lookback DOPO l'inizializzazione degli agenti
        self.max_agent_lookback = 200 # Default iniziale

        logger.info("Backtester inizializzato")

    async def _init_agents(self):
        """Inizializza asincronamente gli agenti di trading."""
        self.agents = {}
        agent_configs = self.config.get('agents', {})
        max_lookback = 0

        if agent_configs.get('technical', {}).get('enabled', False):
            agent = EquityTechnicalAgent(agent_configs['technical'])
            await agent.initialize(agent_configs['technical']) # Await initialize
            self.agents['technical'] = agent
            if hasattr(agent, 'required_lookback'):
                 max_lookback = max(max_lookback, agent.required_lookback)
            elif hasattr(agent, 'timeframes') and 'long' in agent.timeframes: # Fallback per vecchie versioni
                 max_lookback = max(max_lookback, agent.timeframes['long'])


        if agent_configs.get('fundamental', {}).get('enabled', False):
            agent = EquityFundamentalAgent(agent_configs['fundamental'])
            await agent.initialize(agent_configs['fundamental']) # Await initialize
            self.agents['fundamental'] = agent
            if hasattr(agent, 'required_lookback'):
                 max_lookback = max(max_lookback, agent.required_lookback)

        if agent_configs.get('sentiment', {}).get('enabled', False):
            agent = EquitySentimentAgent(agent_configs['sentiment'])
            await agent.initialize(agent_configs['sentiment']) # Await initialize
            self.agents['sentiment'] = agent
            if hasattr(agent, 'required_lookback'):
                 max_lookback = max(max_lookback, agent.required_lookback)


        # --- Initialize Long/Short Strategy Agent ---
        if agent_configs.get('long_short_strategy', {}).get('enabled', False):
            agent_config = agent_configs['long_short_strategy']
            # Pass the cache_dir from the main data config if needed by the agent's handler
            agent_config['cache_dir'] = self.config.get('data', {}).get('cache_dir', './data/cache/long_short')
            agent = LongShortStrategyAgent(agent_config)
            await agent.initialize(agent_config) # Await initialize
            self.agents[agent.agent_id] = agent # Use agent_id from the instance
            if hasattr(agent, 'required_lookback'):
                 max_lookback = max(max_lookback, agent.required_lookback)
            logger.info(f"Initialized LongShortStrategyAgent: {agent.agent_id}")

        # Aggiungere qui altri agenti se necessario

        if not self.agents:
            raise ValueError("Nessun agente abilitato nella configurazione")

        # Imposta il lookback massimo basato sugli agenti inizializzati
        self.max_agent_lookback = max(max_lookback, 20) # Assicura almeno 20
        logger.info(f"Max agent lookback calcolato: {self.max_agent_lookback}")


    def _get_market_data_for_date(self, date: datetime) -> Optional[MarketData]:
        """
        Recupera i dati di mercato per una data specifica.

        Args:
            date: Data per cui recuperare i dati

        Returns:
            Oggetto MarketData o None se i dati non sono disponibili
        """
        if self.all_daily_data is None or self.all_daily_data.empty:
            logger.error("DataFrame 'all_daily_data' pre-fetched non disponibile nel Backtester.")
            return None

        # Verifica se la data corrente esiste nell'indice dei dati pre-fetched
        if date not in self.all_daily_data.index:
            # Attempt to find the closest previous date if exact match is missing (e.g., weekend/holiday)
            try:
                loc = self.all_daily_data.index.get_loc(date, method='ffill')
                closest_date = self.all_daily_data.index[loc]
                logger.warning(f"Data {date} non trovata nell'indice. Using previous date: {closest_date}")
                date = closest_date # Use the previous date for slicing
            except KeyError:
                 logger.error(f"Data {date} non trovata nell'indice e nessuna data precedente disponibile.")
                 return None

        # Estrai dati storici fino alla data corrente (finestra mobile) dalla fonte pre-fetched
        # Calculate the actual start date needed based on lookback
        # Ensure start_slice_date is within the bounds of the available data
        min_data_date = self.all_daily_data.index.min()
        required_start_date = date - pd.Timedelta(days=self.max_agent_lookback * 2) # Ideal start
        start_slice_date = max(min_data_date, required_start_date) # Actual start based on available data

        # Slice the pre-fetched data up to the current date
        historical_data_slice = self.all_daily_data.loc[start_slice_date:date]

        # Verifica se abbiamo abbastanza dati storici (considerando l'effettivo start_slice_date)
        # The agent itself should handle insufficient data if the slice is too short
        if len(historical_data_slice) < 2: # Need at least 2 points for some calcs
             logger.warning(f"Slice dati storici troppo corta per {date}: {len(historical_data_slice)} punti.")
             # Return the slice anyway, let the agent decide
             # return None # Or decide to skip the day here

        # --- Populate MarketData object ---
        # The structure depends on what the agent expects.
        # LongShortStrategyAgent expects combined data with prefixed columns.
        # Other agents might expect separate prices/volumes/economic/sentiment.
        # For now, prioritize LongShortStrategyAgent.

        # Pass the *entire* historical slice to the agent.
        # The agent's internal logic (resampling, indicator calc) will use this slice.
        # We don't need to separate prices/volumes here if the agent handles the combined data.

        # TODO: Adapt this if supporting multiple agent types simultaneously with different data needs.
        # If supporting old agents, we might need to extract SP500 OHLCV, economic, sentiment separately here.

        return MarketData(
            prices=historical_data_slice, # Pass the combined slice
            volumes=None, # Volume is within the slice if needed
            fundamentals=None, # Not used by LS strategy
            sentiment=None, # Not used by LS strategy
            economic=None, # Not used by LS strategy
            technical_indicators={}, # Agent calculates its own
            timestamp=date # The current simulation date
        )

    async def _generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Genera segnali da tutti gli agenti abilitati.

        Args:
            market_data: Dati di mercato correnti

        Returns:
            Lista di segnali di trading
        """
        all_signals = []
        for agent_id, agent in self.agents.items():
            try:
                signals = await agent.generate_signals(market_data)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Errore nella generazione segnali per agente {agent_id}: {str(e)}")

        # Qui si potrebbe implementare una logica di aggregazione dei segnali
        # Per ora, li utilizziamo tutti
        return all_signals

    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calcola il valore totale corrente del portafoglio.

        Args:
            current_price: Prezzo corrente scalare dell'S&P 500

        Returns:
            Valore totale scalare del portafoglio
        """
        equity_value = 0.0
        position_qty = self.positions.get("^GSPC", 0)

        # Assicurati che current_price sia scalare
        current_price_scalar = _get_scalar(current_price)
        if np.isnan(current_price_scalar):
             logger.error("Prezzo corrente non valido nel calcolo del valore del portafoglio.")
             # Potrebbe restituire l'ultimo valore valido o sollevare un errore
             return self.portfolio_history[-1].total_value if self.portfolio_history else self.initial_capital

        if position_qty != 0:
            equity_value = position_qty * current_price_scalar

        # Assicurati che il risultato sia float
        total_value = float(self.cash + equity_value)
        return total_value

    def _execute_trade(self, signal: TradeSignal, current_price: float):
        """
        Simula l'esecuzione di un'operazione di trading.

        Args:
            signal: Segnale di trading da eseguire
            current_price: Prezzo corrente dell'S&P 500
        """
        if signal.symbol != "^GSPC":
            logger.warning(f"Simbolo non supportato: {signal.symbol}")
            return

        # Assicurati che current_price sia scalare
        current_price_scalar = _get_scalar(current_price)
        if np.isnan(current_price_scalar) or current_price_scalar <= 0:
             logger.error(f"Prezzo corrente non valido ({current_price_scalar}) per il trade del segnale: {signal}")
             return

        portfolio_value = self._calculate_portfolio_value(current_price_scalar) # Passa scalare
        current_position_qty = self.positions.get("^GSPC", 0)
        current_position_value = current_position_qty * current_price_scalar

        quantity_to_trade = 0.0
        target_allocation = None
        stop_loss_price = self.position_stop_loss.get(signal.symbol, None) # Keep existing SL unless overwritten

        # --- Logic for Long/Short Strategy Agent ---
        if signal.agent_id == "long_short_sp500_strategy":
            target_allocation = signal.metadata.get('target_allocation')
            new_stop_loss = signal.metadata.get('stop_loss_price') # SL calculated by agent

            if target_allocation is None:
                logger.warning(f"LongShortStrategy signal missing 'target_allocation' in metadata: {signal}")
                return

            # Calculate desired position based on allocation
            desired_value = portfolio_value * target_allocation
            if abs(current_price_scalar) > 1e-9: # Avoid division by zero
                 desired_qty = desired_value / current_price_scalar
            else:
                 desired_qty = 0.0
                 logger.warning(f"Current price is near zero ({current_price_scalar}), cannot calculate desired quantity.")


            # Determine quantity to trade to reach desired state
            quantity_to_trade = desired_qty - current_position_qty
            stop_loss_price = new_stop_loss # Use the SL from the signal for this trade

            logger.debug(f"Target Alloc Signal: Target={target_allocation:.2f}, DesiredQty={desired_qty:.4f}, CurrentQty={current_position_qty:.4f}, TradeQty={quantity_to_trade:.4f}, SL={stop_loss_price}")

        # --- Logic for existing BUY/SELL Agents ---
        else:
            if signal.direction == SignalDirection.BUY:
                # Original logic: Target value based on confidence
                target_value = portfolio_value * signal.confidence
                value_to_buy = max(0, target_value - current_position_value)
                if abs(current_price_scalar) > 1e-9:
                     quantity_to_trade = value_to_buy / current_price_scalar
                else:
                     quantity_to_trade = 0.0
                     logger.warning(f"Current price is near zero ({current_price_scalar}), cannot calculate BUY quantity.")

            elif signal.direction == SignalDirection.SELL:
                 # Original logic: Sell entire position if held
                 if current_position_qty > 1e-9:
                     quantity_to_trade = -current_position_qty
                 else:
                     quantity_to_trade = 0.0 # Cannot sell if not held

            # No stop loss info from these agents by default
            stop_loss_price = None # Clear any previous stop loss if using old agent logic

        # --- Common Execution Logic ---
        if abs(quantity_to_trade * current_price_scalar) < 1: # Ignore trades with value < $1
            # Log only if target allocation was non-zero, otherwise it's just holding
            if target_allocation is not None and abs(target_allocation) > 1e-4:
                 logger.debug(f"Trade ignored because quantity or value is too small: Qty={quantity_to_trade:.4f}, Value={quantity_to_trade * current_price_scalar:.2f}")
            # Also ignore if quantity is effectively zero
            if abs(quantity_to_trade) < 1e-9:
                 return # No trade needed
            # else: # If target_allocation is None, it means old agent logic, proceed with original check
            #      if abs(quantity_to_trade * current_price_scalar) < 1:
            #           logger.debug(f"Trade ignored because quantity or value is too small: Qty={quantity_to_trade:.4f}, Value={quantity_to_trade * current_price_scalar:.2f}")
            #           return


        # Apply slippage (using quantity_to_trade)
        execution_price = current_price_scalar
        if quantity_to_trade > 0: # Buying or Covering Short
            execution_price *= (1 + self.slippage_pct)
        elif quantity_to_trade < 0: # Selling or Shorting
            execution_price *= (1 - self.slippage_pct)

        # Calcola costo/ricavo dell'operazione
        trade_cost_abs = abs(quantity_to_trade) * execution_price

        # Calcola commissione
        commission = trade_cost_abs * self.commission_pct

        # Verifica fondi disponibili per BUY/COVER operations (cost > 0)
        # Note: This simple check doesn't account for margin requirements for shorting.
        # Assumes we can short freely if cash allows covering potential losses (not realistic).
        cost_of_trade = (quantity_to_trade * execution_price) + commission # Negative for sell/short, positive for buy/cover
        if quantity_to_trade > 0 and (trade_cost_abs + commission) > self.cash: # Check for BUY cost > cash
            logger.warning(f"Fondi insufficienti per eseguire BUY/COVER: richiesti {trade_cost_abs + commission:.2f}, disponibili {self.cash:.2f}. Riduco la quantità.")
            # Riduci la quantità proporzionalmente
            available_for_trade = self.cash * 0.99 # Lascia un piccolo margine
            # Check if division is safe
            if abs(trade_cost_abs + commission) < 1e-9:
                 logger.warning("Costo del trade vicino a zero, impossibile calcolare il rapporto.")
                 return
            ratio = available_for_trade / (trade_cost_abs + commission)
            if ratio < 0: ratio = 0 # Evita quantità negative
            quantity_to_trade *= ratio
            if abs(quantity_to_trade * current_price_scalar) < 1: # Ricontrolla se il trade è ancora significativo
                 logger.debug("Trade annullato dopo aggiustamento fondi insufficienti.")
                 return
            # Recalculate costs
            trade_cost_abs = abs(quantity_to_trade) * execution_price
            commission = trade_cost_abs * self.commission_pct

        # Aggiorna stato del portafoglio
        self.cash -= (quantity_to_trade * execution_price + commission) # Subtract cost (positive for buy, negative for sell)
        self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + quantity_to_trade

        # Store or clear stop loss for the position
        new_position_qty = self.positions[signal.symbol]
        if abs(new_position_qty) > 1e-9: # If position is non-zero after trade
             self.position_stop_loss[signal.symbol] = stop_loss_price # Store the SL from the signal
             logger.debug(f"Stop loss for {signal.symbol} set/updated to: {stop_loss_price}")
        else: # Position is closed
             self.positions[signal.symbol] = 0.0 # Ensure it's exactly zero
             if signal.symbol in self.position_stop_loss:
                  del self.position_stop_loss[signal.symbol] # Remove SL for closed position
                  logger.debug(f"Stop loss for {signal.symbol} cleared.")


        # Registra l'operazione
        # Determine actual direction based on quantity_to_trade
        actual_direction = SignalDirection.HOLD
        if quantity_to_trade > 1e-9: actual_direction = SignalDirection.BUY
        elif quantity_to_trade < -1e-9: actual_direction = SignalDirection.SELL

        trade = Trade(
            timestamp=self.current_date,
            symbol=signal.symbol,
            direction=actual_direction, # Record actual executed direction
            quantity=quantity_to_trade,
            price=execution_price,
            commission=commission,
            slippage=abs(execution_price - current_price_scalar),
            # Store target allocation if available, otherwise original confidence
            signal_confidence=target_allocation if target_allocation is not None else signal.confidence,
            agent_id=signal.agent_id
        )
        self.trades.append(trade)
        logger.info(f"Trade Eseguito: {trade.direction.name} {abs(trade.quantity):.4f} {trade.symbol} @ {trade.price:.2f} (Target Alloc: {target_allocation if target_allocation is not None else 'N/A'})")

    def _record_portfolio_snapshot(self, date: datetime, current_price: float):
        """Registra uno snapshot dello stato del portafoglio."""
        # Assicurati che current_price sia scalare
        current_price_scalar = _get_scalar(current_price)
        if np.isnan(current_price_scalar):
             logger.error(f"Prezzo non valido ({current_price}) per snapshot alla data {date}. Uso ultimo valore valido.")
             if self.portfolio_history:
                  current_price_scalar = self.portfolio_history[-1].total_value / self.portfolio_history[-1].positions.get("^GSPC", 1) if self.portfolio_history[-1].positions.get("^GSPC", 0) != 0 else 0
             else:
                  current_price_scalar = 0 # Non possiamo fare molto altro all'inizio


        equity_value = self.positions.get("^GSPC", 0) * current_price_scalar
        total_value = self.cash + equity_value

        # Calcola PnL e rendimenti giornalieri
        pnl = 0.0
        returns = 0.0
        if len(self.portfolio_history) > 0:
            prev_snapshot = self.portfolio_history[-1]
            prev_value = prev_snapshot.total_value
            if prev_value != 0:
                pnl = total_value - prev_value
                returns = pnl / prev_value

        snapshot = PortfolioSnapshot(
            timestamp=date,
            cash=self.cash,
            equity_value=equity_value,
            total_value=total_value,
            positions=self.positions.copy(),
            pnl=pnl,
            returns=returns
        )
        self.portfolio_history.append(snapshot)

    async def initialize_backtester(self):
        """Metodo asincrono per inizializzare gli agenti e calcolare il lookback."""
        await self._init_agents() # Inizializza agenti e calcola max_agent_lookback
        for agent_id in self.agents:
             logger.info(f"Agente {agent_id} inizializzato correttamente.")
        logger.info("Inizializzazione Backtester completata.")

    async def run(self):
        """Esegue il backtest."""
        logger.info(f"Avvio backtest da {self.start_date} a {self.end_date}")

        # Use the pre-fetched combined daily data
        if self.all_daily_data is None or self.all_daily_data.empty:
            logger.error("Dati giornalieri combinati pre-fetched non disponibili nel Backtester.")
            return

        # Filtra i dati pre-fetched per il periodo di backtest effettivo
        backtest_data = self.all_daily_data[(self.all_daily_data.index >= self.start_date) & (self.all_daily_data.index <= self.end_date)]

        if backtest_data.empty:
            logger.error("Nessun dato disponibile per il periodo di backtest specificato.")
            return

        logger.info(f"Periodo di backtest: {backtest_data.index.min()} - {backtest_data.index.max()}")
        logger.info(f"Numero di giorni di trading: {len(backtest_data)}")

        # Ciclo principale del backtest
        for date, row in backtest_data.iterrows():
            self.current_date = date
            # Assicurati che current_price sia uno scalare qui
            # Use the correct prefixed column name
            sp500_close_col = 'SP500_Close' # Assuming this is the correct prefix and column
            if sp500_close_col not in row.index:
                 logger.error(f"Colonna '{sp500_close_col}' non trovata nei dati per la data {date}. Colonne disponibili: {row.index.tolist()}")
                 continue # Skip day if essential column is missing

            current_price = _get_scalar(row[sp500_close_col])
            if np.isnan(current_price):
                 logger.error(f"Prezzo '{sp500_close_col}' non valido per la data {date}. Salto il giorno.")
                 continue # Salta al giorno successivo se il prezzo non è valido

            # --- Check Stop Loss BEFORE generating new signals ---
            stop_loss_triggered = False
            symbol_to_check = "^GSPC" # Assuming single asset for now
            current_position_qty = self.positions.get(symbol_to_check, 0)
            stop_price = self.position_stop_loss.get(symbol_to_check, None)
            # Get low/high for the current day from the row data
            current_low = _get_scalar(row.get('Low'))
            current_high = _get_scalar(row.get('High'))

            if current_position_qty != 0 and stop_price is not None and not pd.isna(current_low) and not pd.isna(current_high):
                stop_hit = False
                exit_direction = SignalDirection.HOLD
                reason = ""

                if current_position_qty > 0 and current_low <= stop_price: # Long position stopped out
                    logger.info(f"[{date.date()}] STOP LOSS triggered for LONG position at {stop_price:.2f} (Low: {current_low:.2f})")
                    stop_hit = True
                    exit_direction = SignalDirection.SELL
                    reason = 'stop_loss_long'
                elif current_position_qty < 0 and current_high >= stop_price: # Short position stopped out
                    logger.info(f"[{date.date()}] STOP LOSS triggered for SHORT position at {stop_price:.2f} (High: {current_high:.2f})")
                    stop_hit = True
                    exit_direction = SignalDirection.BUY # Buy to cover
                    reason = 'stop_loss_short'

                if stop_hit:
                    # Create a signal to close/cover the position at stop price
                    exit_signal = TradeSignal(
                        agent_id='stop_loss_manager',
                        symbol=symbol_to_check,
                        direction=exit_direction,
                        confidence=1.0, # Close/Cover full position
                        timestamp=self.current_date,
                        metadata={'reason': reason, 'stop_price': stop_price}
                    )
                    # Execute the stop-loss trade immediately using the stop price
                    self._execute_trade(exit_signal, stop_price)
                    stop_loss_triggered = True # Mark that SL was hit

            # If stop loss triggered, record snapshot and skip normal signal processing for this day
            if stop_loss_triggered:
                # Record snapshot at stop price after the SL trade
                self._record_portfolio_snapshot(date, stop_price)
                continue # Move to the next day

            # --- Process normal signals if stop loss was NOT triggered ---
            # 1. Recupera dati di mercato per la data corrente
            market_data = self._get_market_data_for_date(date)
            if market_data is None:
                logger.warning(f"Dati di mercato non trovati per {date}, salto il giorno.")
                # Registra snapshot con valori precedenti se possibile
                if self.portfolio_history:
                    last_snapshot = self.portfolio_history[-1]
                    snapshot = PortfolioSnapshot(
                        timestamp=date,
                        cash=last_snapshot.cash,
                        equity_value=last_snapshot.equity_value,
                        total_value=last_snapshot.total_value,
                        positions=last_snapshot.positions.copy(),
                        pnl=0.0,
                        returns=0.0
                    )
                    self.portfolio_history.append(snapshot)
                else: # First day, no history, record initial state
                     self._record_portfolio_snapshot(date, current_price)
                continue

            # 2. Genera segnali dagli agenti
            signals = await self._generate_signals(market_data)

            # 3. Esegui operazioni basate sui segnali
            if signals:
                # Execute all signals from the LongShortStrategy (should be only one target signal)
                # Or handle multiple signals if other agents were enabled
                for signal in signals:
                     # Use current_price (Close) for execution decision, _execute_trade handles slippage
                     self._execute_trade(signal, current_price)
                     # If multiple agents, might break after first trade or aggregate differently
                     if signal.agent_id == "long_short_sp500_strategy":
                          break # Assume only one target signal per day from this strategy

            # 4. Registra lo snapshot del portafoglio alla fine della giornata
            # Use the day's closing price for the end-of-day snapshot
            self._record_portfolio_snapshot(date, current_price)

            # Log progresso ogni 100 giorni
            if len(self.portfolio_history) % 100 == 0:
               logger.info(f"Progresso: {date.date()} - Valore Portafoglio: {self.portfolio_history[-1].total_value:.2f}")

        logger.info("Backtest completato")

    def get_results(self) -> pd.DataFrame:
        """
        Restituisce i risultati del backtest come DataFrame.

        Returns:
            DataFrame con la storia del portafoglio.
        """
        if not self.portfolio_history:
            return pd.DataFrame()

        # Converti lista di dataclass in DataFrame
        history_df = pd.DataFrame([vars(s) for s in self.portfolio_history])
        history_df.set_index('timestamp', inplace=True)
        return history_df

    def get_trades(self) -> pd.DataFrame:
        """
        Restituisce le operazioni eseguite come DataFrame.

        Returns:
            DataFrame con la storia delle operazioni.
        """
        if not self.trades:
            return pd.DataFrame()

        trades_df = pd.DataFrame([vars(t) for t in self.trades])
        trades_df.set_index('timestamp', inplace=True)
        return trades_df

    def analyze_performance(self, output_dir: str = "./results"):
        """
        Analizza le performance del backtest e salva i risultati.

        Args:
            output_dir: Directory dove salvare i risultati dell'analisi
        """
        logger.info("Analisi performance del backtest")

        results_df = self.get_results()
        if results_df.empty:
            logger.error("Nessun risultato da analizzare")
            return

        # Crea directory di output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepara i rendimenti
        returns = results_df['returns']

        # Esegui analisi quantstats
        try:
            # Assicurati che matplotlib usi il backend Agg
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt # Importa pyplot DOPO aver impostato il backend

            # Assicurati che l'indice sia datetime
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)

            # Ottieni rendimenti benchmark
            benchmark_returns = self.get_benchmark_returns()

            logger.info(f"Generazione grafici e metriche QuantStats in {output_dir}")

            # --- Genera e salva grafici individuali come PNG ---
            plot_functions = {
                "returns": qs.plots.returns,
                "drawdown": qs.plots.drawdown,
                "monthly_returns": qs.plots.monthly_returns,
                "distribution": qs.plots.distribution,
                "rolling_sharpe": qs.plots.rolling_sharpe,
                "rolling_volatility": qs.plots.rolling_volatility,
                # Aggiungi altri plot se necessario, es: qs.plots.snapshot
            }

            # Sopprimi FutureWarning specifici da quantstats durante il plotting
            with warnings.catch_warnings():
                 warnings.filterwarnings(
                     "ignore",
                     category=FutureWarning,
                     message=".*A value is trying to be set on a copy of a DataFrame or Series.*"
                 )

                 for name, plot_func in plot_functions.items():
                     fig = None # Inizializza fig a None
                     try:
                         # Rimuovi 'benchmark' dalle chiamate che non lo supportano
                         kwargs = {'show': False, 'figsize': (10, 6)}
                         if name not in ['drawdown', 'monthly_returns', 'distribution']:
                             kwargs['benchmark'] = benchmark_returns

                         fig = plot_func(returns, **kwargs)

                         if fig:
                             plot_path = Path(output_dir) / f"performance_{name}.png"
                             fig.savefig(plot_path, format='png', bbox_inches='tight')
                             plt.close(fig) # Chiudi la figura per liberare memoria
                             logger.info(f"Grafico '{name}' salvato in {plot_path}")
                         else:
                             logger.warning(f"La funzione qs.plots.{name} non ha restituito una figura.")
                     except Exception as plot_err:
                          logger.error(f"Errore durante la generazione/salvataggio del grafico '{name}': {plot_err}")
                          if fig: plt.close(fig) # Assicura chiusura figura in caso di errore
            # --- Fine soppressione warning ---
            # -----------------------------------------------------

            # Calcola e salva metriche
            metrics_output_path = Path(output_dir) / "performance_metrics.txt"
            metrics_df = qs.reports.metrics(returns, benchmark=benchmark_returns, display=False)
            with open(metrics_output_path, 'w') as f:
                 f.write(metrics_df.to_string())
            logger.info(f"Metriche QuantStats salvate in {metrics_output_path}")

            # Rimosso il tentativo di generare il report HTML completo

        except Exception as e: # Questo except cattura errori generali nell'analisi
            logger.error(f"Errore generale durante l'analisi QuantStats: {str(e)}")
            # Salva comunque i rendimenti se possibile
            if not returns.empty:
                try:
                    returns.to_csv(output_path / "portfolio_returns.csv")
                    logger.info(f"Rendimenti del portafoglio salvati in {output_path / 'portfolio_returns.csv'}")
                except Exception as save_err:
                    logger.error(f"Impossibile salvare i rendimenti dopo errore analisi: {save_err}")

    def get_benchmark_returns(self) -> Optional[pd.Series]:
        """
        Recupera i rendimenti del benchmark (S&P 500).

        Returns:
            Series con i rendimenti giornalieri del benchmark
        """
        # Use pre-fetched data, assuming it contains benchmark columns (e.g., SP500_Close)
        if self.all_daily_data is None or self.all_daily_data.empty:
            logger.warning("Dati giornalieri pre-fetched non disponibili per il benchmark.")
            return None

        # Identify benchmark columns (assuming SP500 prefix)
        benchmark_cols = [col for col in self.all_daily_data.columns if isinstance(col, str) and col.startswith('SP500_')]
        if not benchmark_cols:
             # Handle potential MultiIndex
             benchmark_cols = [col for col in self.all_daily_data.columns if isinstance(col, tuple) and col[0].startswith('SP500_')]

        if not benchmark_cols:
            logger.warning("Nessuna colonna SP500 trovata nei dati pre-fetched per il benchmark.")
            return None

        # Select only the benchmark data
        benchmark_data = self.all_daily_data[benchmark_cols]
        # Rename columns to remove prefix for consistency if needed, or use prefixed names
        # Example: benchmark_data.columns = [col[0].replace('SP500_', '') if isinstance(col, tuple) else col.replace('SP500_', '') for col in benchmark_data.columns]

        # Ensure the benchmark data has the correct index type
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
             benchmark_data.index = pd.to_datetime(benchmark_data.index)

        # Filtra per il periodo di backtest
        benchmark_data_filtered = benchmark_data[(benchmark_data.index >= self.start_date) & (benchmark_data.index <= self.end_date)]

        if benchmark_data_filtered.empty:
             logger.warning("Nessun dato benchmark disponibile per il periodo di backtest.")
             return None

        # Use the correct prefixed 'Close' column
        price_column_id = 'SP500_Close'
        # Handle potential MultiIndex
        if isinstance(benchmark_data_filtered.columns, pd.MultiIndex):
             potential_ids = [('SP500_Close', ''), ('SP500_Close', 'SP500_^GSPC'), 'SP500_Close']
             for pid in potential_ids:
                  if pid in benchmark_data_filtered.columns:
                       price_column_id = pid
                       break
             else:
                  logger.warning(f"Colonna 'SP500_Close' non trovata per il benchmark nei dati filtrati: {benchmark_data_filtered.columns}")
                  return None
        elif price_column_id not in benchmark_data_filtered.columns:
             logger.warning(f"Colonna 'SP500_Close' non trovata per il benchmark nei dati filtrati: {benchmark_data_filtered.columns}")
             return None

        if price_column_id:
            benchmark_returns = benchmark_data_filtered[price_column_id].pct_change().fillna(0)
            # Assicura che l'indice sia allineato con i rendimenti del portafoglio
            portfolio_dates = self.get_results().index
            if not portfolio_dates.empty:
                 return benchmark_returns.reindex(portfolio_dates).fillna(0)
            else:
                 return benchmark_returns
        else:
            logger.warning("Colonna 'Adj Close' non trovata nei dati benchmark.")
            return None

    def plot_results(self, output_dir: str = "./results"):
        """
        Genera grafici dei risultati del backtest.
        (Nota: la maggior parte dei plot è ora gestita da analyze_performance con QuantStats)
        """
        results_df = self.get_results()
        if results_df.empty:
            logger.warning("Nessun risultato da plottare.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Grafico Valore Portafoglio
        plt.figure(figsize=(12, 6))
        results_df['total_value'].plot(title='Valore Totale Portafoglio')
        plt.ylabel('Valore ($)')
        plt.xlabel('Data')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path / "portfolio_value.png")
        plt.close()
        logger.info(f"Grafico valore portafoglio salvato in {output_path / 'portfolio_value.png'}")

        # Potresti aggiungere altri grafici specifici qui se necessario