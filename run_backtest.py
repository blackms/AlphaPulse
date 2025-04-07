#!/usr/bin/env python3
"""
Script principale per eseguire il backtest S&P 500 con AlphaPulse.

Questo script orchestra l'intero processo:
1. Carica la configurazione
2. Carica i dati dal database utilizzando DataLoader
3. Esegue l'orchestratore della strategia per generare segnali e target
4. Inizializza ed esegue il Backtester con i target generati
5. Analizza e salva i risultati
"""

import os
import argparse
import logging
import yaml
import asyncio
from typing import Dict, List
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Importa componenti necessari dalla nuova struttura
from src.alpha_pulse.backtesting.backtester import Backtester, BacktestResult
from src.alpha_pulse.backtesting.data_loader import load_ohlcv_data
from src.alpha_pulse.strategies.long_short.orchestrator import LongShortOrchestrator
# Importa pyfolio se vuoi usarlo per l'analisi (assicurati sia installato)
try:
    import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except ImportError:
    PYFOLIO_AVAILABLE = False
    logging.warning("Pyfolio non trovato. L'analisi dettagliata delle performance sarà limitata.")


# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RunBacktest")

# Carica le variabili d'ambiente (per connessione DB)
load_dotenv()

def parse_arguments():
    """Parse gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Esegui backtest S&P 500 con AlphaPulse')
    parser.add_argument('--config', type=str, default='config/backtest_sp500_config.yaml',
                        help='Percorso del file di configurazione YAML')
    parser.add_argument('--output-dir', type=str, default='./results/sp500_backtest',
                        help='Directory di output per i risultati')
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Carica la configurazione da un file YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configurazione caricata da {config_path}")
        return config
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Errore nel caricamento della configurazione: {e}")
        raise ValueError(f"Impossibile caricare la configurazione da {config_path}: {str(e)}")

def save_results(output_dir: Path, result: BacktestResult):
    """Salva i risultati del backtest."""
    try:
        # Salva curva equity
        result.equity_curve.to_csv(output_dir / "equity_curve.csv")
        logger.info(f"Curva equity salvata in {output_dir / 'equity_curve.csv'}")

        # Salva storico posizioni
        if result.positions:
            positions_df = pd.DataFrame([p.__dict__ for p in result.positions])
            positions_df.to_csv(output_dir / "trades_history.csv", index=False)
            logger.info(f"Storia operazioni salvata in {output_dir / 'trades_history.csv'}")
        else:
            logger.info("Nessuna operazione eseguita, file trades_history.csv non creato.")

        # Salva riepilogo metriche
        summary = {
            "Total Return": f"{result.total_return:.2%}",
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Max Drawdown": f"{result.max_drawdown:.2%}",
            "Total Trades": result.total_trades,
            "Winning Trades": result.winning_trades,
            "Losing Trades": result.losing_trades,
            "Win Rate": f"{result.win_rate:.2%}",
            "Avg Win Pct": f"{result.avg_win:.2%}",
            "Avg Loss Pct": f"{result.avg_loss:.2%}",
            "Profit Factor": f"{result.profit_factor:.2f}",
        }
        with open(output_dir / "summary.yaml", 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        logger.info(f"Riepilogo metriche salvato in {output_dir / 'summary.yaml'}")

        # Genera report Pyfolio se disponibile
        if PYFOLIO_AVAILABLE:
            logger.info("Generazione report Pyfolio...")
            # Pyfolio richiede i rendimenti giornalieri semplici
            returns = result.equity_curve.pct_change().fillna(0)
            # Assicurati che l'indice sia timezone-naive per Pyfolio
            if isinstance(returns.index, pd.DatetimeIndex) and returns.index.tz is not None:
                 returns.index = returns.index.tz_localize(None)

            pf.create_full_tear_sheet(
                returns,
                positions=None, # Pyfolio positions format is complex, skip for now
                transactions=None, # Pyfolio transactions format is complex, skip for now
                round_trips=False, # Requires transaction data
                live_start_date=None,
                output=str(output_dir / 'pyfolio_report.pdf')
            )
            logger.info(f"Report Pyfolio salvato in {output_dir / 'pyfolio_report.pdf'}")

    except Exception as e:
        logger.error(f"Errore durante il salvataggio o l'analisi dei risultati: {e}")


async def main():
    """Funzione principale asincrona."""
    args = parse_arguments()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory di output: {output_dir}")

    # --- Imposta variabili ambiente DB (se non già presenti) ---
    db_config = config.get('database', {})
    os.environ.setdefault('DB_HOST', db_config.get('host', 'localhost'))
    os.environ.setdefault('DB_PORT', str(db_config.get('port', 5432)))
    os.environ.setdefault('DB_NAME', db_config.get('database', 'backtesting')) # Usa DB corretto
    os.environ.setdefault('DB_USER', db_config.get('user', 'devuser'))       # Usa utente corretto
    os.environ.setdefault('DB_PASS', db_config.get('password', 'devpassword')) # Usa password corretta
    logger.info(f"Variabili DB impostate per: {os.environ['DB_USER']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}")

    # --- 1. Caricamento Dati ---
    logger.info("=== Fase 1: Caricamento Dati ===")
    backtest_config = config.get('backtest', {})
    data_loader_config = config.get('data_loader', {})
    strategy_config = config.get('strategy', {}).get('long_short', {}) # Config specifica strategia

    start_date_naive = datetime.fromisoformat(backtest_config['start_date'])
    end_date_naive = datetime.fromisoformat(backtest_config['end_date'])
    # Make dates timezone-aware (UTC) for comparison with DataFrame index
    start_date = pd.Timestamp(start_date_naive, tz='UTC')
    end_date = pd.Timestamp(end_date_naive, tz='UTC')
    symbols_needed = strategy_config.get('symbols', ['^GSPC', '^VIX']) # Simboli richiesti dalla strategia
    timeframe = strategy_config.get('timeframe', '1d') # Timeframe della strategia
    exchange = data_loader_config.get('exchange', 'yfinance') # Exchange da cui caricare

    # Calcola una data di inizio leggermente precedente per permettere il calcolo degli indicatori iniziali
    # Es: se MA è 40 periodi, servono almeno 40 periodi prima di start_date
    indicator_config = strategy_config.get('indicators', {})
    lookback_periods = max(
        indicator_config.get('ma_window', 40),
        indicator_config.get('rsi_window', 14),
        indicator_config.get('atr_window', 14)
    )
    # Stima un buffer (es. 1.5x il lookback in giorni di calendario)
    buffer_days = int(lookback_periods * 1.5 * (7/5 if timeframe == '1d' else 1)) # Approssima giorni lavorativi
    # Use the naive start date for calculating the buffer start
    load_start_date_naive = start_date_naive - pd.Timedelta(days=buffer_days)
    # load_ohlcv_data expects naive datetimes based on its type hints
    load_end_date_naive = end_date_naive

    loaded_data = await load_ohlcv_data(
        symbols=symbols_needed,
        timeframe=timeframe,
        start_dt=load_start_date_naive, # Pass naive datetime
        end_dt=load_end_date_naive,     # Pass naive datetime
        exchange=exchange
    )

    # Check if data loading failed or if the primary symbol's DataFrame is missing or empty
    primary_symbol = '^GSPC' # Define primary symbol for check
    if loaded_data is None or loaded_data.get(primary_symbol) is None or loaded_data[primary_symbol].empty:
        logger.error(f"Caricamento dati fallito o dati per il simbolo primario '{primary_symbol}' mancanti/vuoti. Impossibile continuare.")
        return
    logger.info("Caricamento dati completato.")

    # --- 2. Generazione Segnali e Target ---
    logger.info("=== Fase 2: Generazione Segnali e Target ===")
    orchestrator = LongShortOrchestrator(strategy_config)
    orchestrator_results = orchestrator.calculate_signals_and_targets(loaded_data)

    if orchestrator_results is None:
        logger.error("Generazione segnali e target fallita. Impossibile continuare.")
        return

    data_with_signals, target_allocations, stop_losses = orchestrator_results
    logger.info("Generazione segnali e target completata.")

    # --- 3. Esecuzione Backtest ---
    logger.info("=== Fase 3: Esecuzione Backtest ===")
    # Estrai la serie di prezzi necessaria per il backtester (es. Close del simbolo primario)
    # Assicurati che le colonne preparate dall'orchestrator siano usate qui
    price_series = data_with_signals['SP500_Close'] # Usa la colonna corretta

    # Filtra i dati per il range di backtest effettivo (escludendo il buffer iniziale)
    price_series = price_series[start_date:end_date]
    target_allocations = target_allocations[start_date:end_date]
    stop_losses = stop_losses[start_date:end_date]

    # Inizializza il Backtester
    backtester_params = backtest_config.get('parameters', {})
    backtester = Backtester(
        initial_capital=backtester_params.get('initial_capital', 100000.0),
        commission=backtester_params.get('commission', 0.001)
    )

    # Esegui il backtest
    backtest_result = backtester.backtest(
        prices=price_series,
        signals=target_allocations, # Passa le allocazioni target
        stop_losses=stop_losses     # Passa gli stop loss calcolati
    )
    logger.info("Esecuzione backtest completata.")

    # --- 4. Analisi e Salvataggio Risultati ---
    logger.info("=== Fase 4: Analisi e Salvataggio Risultati ===")
    print(backtest_result) # Stampa il riepilogo base
    save_results(output_dir, backtest_result)
    logger.info("Analisi e salvataggio risultati completati.")


if __name__ == "__main__":
    asyncio.run(main())