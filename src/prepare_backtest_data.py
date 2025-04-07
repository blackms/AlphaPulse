#!/usr/bin/env python3
"""
Script per preparare i dati per il backtest S&P 500 con AlphaPulse.

Questo script utilizza il DataManager per recuperare e preparare tutti i dati
necessari per eseguire un backtest completo dell'S&P 500.
"""

import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Importa il DataManager
from data_manager import DataManager

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BacktestPreparation")

# Carica le variabili d'ambiente
load_dotenv()

def parse_arguments():
    """Parse gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Preparazione dati per backtest S&P 500')
    
    parser.add_argument('--start-date', type=str, default='2010-01-01',
                        help='Data di inizio (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='Data di fine (YYYY-MM-DD)')
    
    parser.add_argument('--output-dir', type=str, default='./data/backtest',
                        help='Directory di output per i dati preparati')
    
    parser.add_argument('--cache-dir', type=str, default='./data/cache',
                        help='Directory di cache per i dati scaricati')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Genera visualizzazioni dei dati')
    
    return parser.parse_args()

def create_directories(dirs):
    """Crea le directory necessarie."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory creata: {dir_path}")

def visualize_data(data, output_dir):
    """
    Genera visualizzazioni dei dati.
    
    Args:
        data: Dati da visualizzare
        output_dir: Directory di output per le visualizzazioni
    """
    logger.info("Generazione visualizzazioni dei dati")
    
    # Crea directory per le visualizzazioni
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Imposta stile
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Grafico dei prezzi S&P 500
    if 'prices' in data and not data['prices'].empty:
        plt.figure()
        plt.title('S&P 500 (2010-2023)')
        plt.plot(data['prices']['Close'], label='Close')
        if 'Adj Close' in data['prices'].columns:
            plt.plot(data['prices']['Adj Close'], label='Adj Close', alpha=0.7)
        plt.xlabel('Data')
        plt.ylabel('Prezzo')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "sp500_prices.png")
        plt.close()
        logger.info(f"Grafico prezzi S&P 500 salvato in {viz_dir / 'sp500_prices.png'}")
    
    # 2. Grafico dei volumi
    if 'volumes' in data and data['volumes'] is not None:
        plt.figure()
        plt.title('Volume di Trading S&P 500')
        plt.bar(data['volumes'].index, data['volumes'], alpha=0.7)
        plt.xlabel('Data')
        plt.ylabel('Volume')
        plt.tight_layout()
        plt.savefig(viz_dir / "sp500_volumes.png")
        plt.close()
        logger.info(f"Grafico volumi S&P 500 salvato in {viz_dir / 'sp500_volumes.png'}")
    
    # 3. Grafico degli indicatori economici
    if 'economic' in data and data['economic']:
        # Crea un DataFrame combinato per gli indicatori
        econ_data = {}
        for indicator, series in data['economic'].items():
            econ_data[indicator] = series
        
        econ_df = pd.DataFrame(econ_data)
        
        # Normalizza i dati per la visualizzazione
        econ_df_norm = (econ_df - econ_df.mean()) / econ_df.std()
        
        plt.figure()
        plt.title('Indicatori Economici (Normalizzati)')
        for col in econ_df_norm.columns:
            plt.plot(econ_df_norm.index, econ_df_norm[col], label=col)
        plt.xlabel('Data')
        plt.ylabel('Valore (Normalizzato)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "economic_indicators.png")
        plt.close()
        logger.info(f"Grafico indicatori economici salvato in {viz_dir / 'economic_indicators.png'}")
    
    # 4. Grafico del sentiment
    if 'sentiment' in data and not data['sentiment'].empty:
        plt.figure()
        plt.title('Sentiment di Mercato')
        plt.plot(data['sentiment'].index, data['sentiment']['sentiment_score'], label='Sentiment Score')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.fill_between(data['sentiment'].index, data['sentiment']['sentiment_score'], 0, 
                         where=data['sentiment']['sentiment_score'] > 0, color='green', alpha=0.3)
        plt.fill_between(data['sentiment'].index, data['sentiment']['sentiment_score'], 0, 
                         where=data['sentiment']['sentiment_score'] < 0, color='red', alpha=0.3)
        plt.xlabel('Data')
        plt.ylabel('Sentiment Score')
        plt.tight_layout()
        plt.savefig(viz_dir / "market_sentiment.png")
        plt.close()
        logger.info(f"Grafico sentiment salvato in {viz_dir / 'market_sentiment.png'}")
    
    # 5. Grafico di correlazione
    if 'prices' in data and not data['prices'].empty and 'economic' in data and data['economic']:
        # Prepara DataFrame per la correlazione
        corr_data = {'SP500': data['prices']['Close']}
        
        for indicator, series in data['economic'].items():
            # Riesampiona alla stessa frequenza dei prezzi
            resampled = series.reindex(data['prices'].index, method='ffill')
            corr_data[indicator] = resampled
        
        corr_df = pd.DataFrame(corr_data).dropna()
        
        if not corr_df.empty:
            # Calcola matrice di correlazione
            corr_matrix = corr_df.corr()
            
            plt.figure(figsize=(10, 8))
            plt.title('Matrice di Correlazione')
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.tight_layout()
            plt.savefig(viz_dir / "correlation_matrix.png")
            plt.close()
            logger.info(f"Matrice di correlazione salvata in {viz_dir / 'correlation_matrix.png'}")
    
    logger.info("Visualizzazioni generate con successo")

def main():
    """Funzione principale."""
    # Parse degli argomenti
    args = parse_arguments()
    
    # Crea le directory necessarie
    create_directories([args.output_dir, args.cache_dir])
    
    # Configurazione per il DataManager
    config = {
        "cache_dir": args.cache_dir
    }
    
    # Inizializza il DataManager
    data_manager = DataManager(config)
    
    # Recupera e prepara i dati
    logger.info(f"Preparazione dati da {args.start_date} a {args.end_date}")
    market_data = data_manager.prepare_market_data(args.start_date, args.end_date)
    
    if not market_data:
        logger.error("Errore nella preparazione dei dati di mercato")
        return
    
    # Formatta i dati per gli agenti
    agent_data = data_manager.format_data_for_agents(market_data)
    
    if not agent_data:
        logger.error("Errore nella formattazione dei dati per gli agenti")
        return
    
    # Salva i dati preparati
    output_file = f"sp500_data_{args.start_date}_{args.end_date}.json"
    output_path = Path(args.output_dir) / output_file
    
    # Salva i dati in formato JSON
    with open(output_path, 'w') as f:
        # Converti DataFrame in dizionari
        serializable_data = {}
        for key, value in market_data.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = value.to_dict(orient='records')
            elif isinstance(value, pd.Series):
                serializable_data[key] = value.to_dict()
            else:
                serializable_data[key] = value
        
        json.dump(serializable_data, f, indent=2, default=str)
    
    logger.info(f"Dati salvati in {output_path}")
    
    # Genera visualizzazioni se richiesto
    if args.visualize:
        visualize_data(market_data, args.output_dir)
    
    logger.info("Preparazione dati completata con successo")
    
    # Stampa riepilogo
    print("\n=== Riepilogo Preparazione Dati ===")
    print(f"Periodo: {args.start_date} - {args.end_date}")
    print(f"Dati S&P 500: {len(market_data['prices'])} giorni")
    
    if 'economic' in market_data:
        print("\nIndicatori Economici:")
        for indicator, data in market_data['economic'].items():
            print(f"  - {indicator}: {len(data)} punti")
    
    if 'sentiment' in market_data and not market_data['sentiment'].empty:
        print(f"\nDati Sentiment: {len(market_data['sentiment'])} giorni")
    
    print(f"\nDati salvati in: {output_path}")
    
    if args.visualize:
        print(f"Visualizzazioni salvate in: {Path(args.output_dir) / 'visualizations'}")
    
    print("\nI dati sono pronti per il backtest!")

if __name__ == "__main__":
    main()