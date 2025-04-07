#!/usr/bin/env python3
"""
Data Manager per il backtest S&P 500 con AlphaPulse.

Questo modulo gestisce il recupero, la pulizia e la preparazione dei dati
da diverse fonti (FRED, Yahoo Finance, NewsAPI) per il backtest.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import requests
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import json
from pathlib import Path

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataManager")

# Carica le variabili d'ambiente
load_dotenv()

class DataManager:
    """
    Gestisce il recupero e la preparazione dei dati per il backtest S&P 500.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il Data Manager.
        
        Args:
            config: Configurazione per il data manager
        """
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "./data/cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza connessioni API
        self._init_api_connections()
        
        # Dizionario per memorizzare i dati
        self.data = {
            "market": {},
            "fundamental": {},
            "economic": {},
            "sentiment": {}
        }
        
        logger.info("DataManager inizializzato")
    
    def _init_api_connections(self):
        """Inizializza le connessioni alle API."""
        # FRED API
        fred_api_key = os.getenv("FRED_API_KEY")
        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
            logger.info("Connessione FRED API inizializzata")
        else:
            self.fred = None
            logger.warning("FRED API key non trovata, funzionalità FRED disabilitate")
        
        # NewsAPI
        self.news_api_key = os.getenv("NEWS_API_KEY")
        if not self.news_api_key:
            logger.warning("NewsAPI key non trovata, funzionalità NewsAPI disabilitate")
    
    def fetch_sp500_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Recupera i dati storici dell'indice S&P 500.
        
        Args:
            start_date: Data di inizio (formato YYYY-MM-DD)
            end_date: Data di fine (formato YYYY-MM-DD)
            
        Returns:
            DataFrame con i dati storici
        """
        cache_file = self.cache_dir / f"sp500_{start_date}_{end_date}.parquet"
        
        # Controlla se i dati sono già in cache
        if cache_file.exists():
            logger.info(f"Caricamento dati S&P 500 dalla cache: {cache_file}")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Recupero dati S&P 500 da {start_date} a {end_date}")
        
        # Prova prima con Yahoo Finance
        try:
            sp500 = yf.download("^GSPC", start=start_date, end=end_date)
            if not sp500.empty:
                logger.info(f"Dati S&P 500 recuperati da Yahoo Finance: {len(sp500)} righe")
                # Salva in cache
                sp500.to_parquet(cache_file)
                return sp500
        except Exception as e:
            logger.warning(f"Errore nel recupero dati da Yahoo Finance: {str(e)}")
        
        # Se Yahoo Finance fallisce, prova con FRED
        if self.fred:
            try:
                sp500 = self.fred.get_series('SP500', start_date, end_date)
                if not sp500.empty:
                    # Converti in DataFrame
                    sp500_df = pd.DataFrame(sp500).reset_index()
                    sp500_df.columns = ['Date', 'Close']
                    sp500_df.set_index('Date', inplace=True)
                    
                    logger.info(f"Dati S&P 500 recuperati da FRED: {len(sp500_df)} righe")
                    # Salva in cache
                    sp500_df.to_parquet(cache_file)
                    return sp500_df
            except Exception as e:
                logger.error(f"Errore nel recupero dati da FRED: {str(e)}")
        
        logger.error("Impossibile recuperare dati S&P 500 da nessuna fonte")
        return pd.DataFrame()
    
    def fetch_economic_indicators(self, indicators: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """
        Recupera indicatori economici da FRED.
        
        Args:
            indicators: Lista di codici indicatori FRED
            start_date: Data di inizio (formato YYYY-MM-DD)
            end_date: Data di fine (formato YYYY-MM-DD)
            
        Returns:
            Dizionario di Series con i dati degli indicatori
        """
        if not self.fred:
            logger.error("FRED API non inizializzata")
            return {}
        
        results = {}
        for indicator in indicators:
            cache_file = self.cache_dir / f"fred_{indicator}_{start_date}_{end_date}.parquet"
            
            # Controlla se i dati sono già in cache
            if cache_file.exists():
                logger.info(f"Caricamento indicatore {indicator} dalla cache")
                results[indicator] = pd.read_parquet(cache_file).squeeze()
                continue
            
            try:
                logger.info(f"Recupero indicatore {indicator} da FRED")
                data = self.fred.get_series(indicator, start_date, end_date)
                if not data.empty:
                    # Salva in cache
                    pd.DataFrame(data).to_parquet(cache_file)
                    results[indicator] = data
                    logger.info(f"Indicatore {indicator} recuperato: {len(data)} punti")
                else:
                    logger.warning(f"Nessun dato trovato per l'indicatore {indicator}")
            except Exception as e:
                logger.error(f"Errore nel recupero dell'indicatore {indicator}: {str(e)}")
        
        return results
    
    def fetch_news_sentiment(self, query: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Recupera e analizza il sentiment dalle notizie.
        
        Args:
            query: Query di ricerca (es. "S&P 500")
            start_date: Data di inizio (formato YYYY-MM-DD)
            end_date: Data di fine (formato YYYY-MM-DD)
            
        Returns:
            DataFrame con dati di sentiment
        """
        if not self.news_api_key:
            logger.error("NewsAPI key non trovata")
            return pd.DataFrame()
        
        cache_file = self.cache_dir / f"news_{query.replace(' ', '_')}_{start_date}_{end_date}.parquet"
        
        # Controlla se i dati sono già in cache
        if cache_file.exists():
            logger.info(f"Caricamento dati di sentiment dalla cache")
            return pd.read_parquet(cache_file)
        
        # NewsAPI ha limitazioni sulla data di inizio per account gratuiti
        # Potrebbe essere necessario suddividere la richiesta in intervalli più piccoli
        
        logger.info(f"Recupero notizie per '{query}' da {start_date} a {end_date}")
        
        base_url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.news_api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                if not articles:
                    logger.warning(f"Nessun articolo trovato per '{query}'")
                    return pd.DataFrame()
                
                # Crea DataFrame con i dati degli articoli
                articles_df = pd.DataFrame(articles)
                
                # Estrai la data di pubblicazione
                articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'])
                
                # Implementa qui un semplice sentiment analyzer
                # Per ora, utilizziamo un sentiment casuale per dimostrazione
                # In un'implementazione reale, utilizzeremmo un modello NLP
                articles_df['sentiment_score'] = np.random.uniform(-1, 1, size=len(articles_df))
                
                # Aggrega per data
                daily_sentiment = articles_df.groupby(articles_df['publishedAt'].dt.date).agg({
                    'sentiment_score': 'mean',
                    'title': 'count'
                }).rename(columns={'title': 'article_count'})
                
                logger.info(f"Dati di sentiment elaborati: {len(daily_sentiment)} giorni")
                
                # Salva in cache
                daily_sentiment.to_parquet(cache_file)
                
                return daily_sentiment
            else:
                logger.error(f"Errore nella richiesta NewsAPI: {response.status_code}")
                logger.error(response.text)
        except Exception as e:
            logger.error(f"Errore nel recupero dati di sentiment: {str(e)}")
        
        return pd.DataFrame()
    
    def prepare_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Prepara tutti i dati di mercato necessari per il backtest.
        
        Args:
            start_date: Data di inizio (formato YYYY-MM-DD)
            end_date: Data di fine (formato YYYY-MM-DD)
            
        Returns:
            Dizionario con i dati di mercato
        """
        logger.info(f"Preparazione dati di mercato da {start_date} a {end_date}")
        
        # Recupera dati S&P 500
        sp500_data = self.fetch_sp500_data(start_date, end_date)
        if sp500_data.empty:
            logger.error("Impossibile procedere senza dati S&P 500")
            return {}
        
        # Recupera indicatori economici
        economic_indicators = [
            'GDP',           # PIL
            'UNRATE',        # Tasso di disoccupazione
            'CPIAUCSL',      # Inflazione (CPI)
            'DFF',           # Federal Funds Rate
            'DGS10'          # 10-Year Treasury Yield
        ]
        
        economic_data = self.fetch_economic_indicators(economic_indicators, start_date, end_date)
        
        # Recupera dati di sentiment
        sentiment_data = self.fetch_news_sentiment("S&P 500 OR S&P500 OR 'Standard and Poor'", start_date, end_date)
        
        # Memorizza i dati
        self.data["market"]["sp500"] = sp500_data
        self.data["economic"] = economic_data
        self.data["sentiment"]["news"] = sentiment_data
        
        logger.info("Preparazione dati di mercato completata")
        
        return {
            "prices": sp500_data,
            "volumes": sp500_data['Volume'] if 'Volume' in sp500_data.columns else None,
            "economic": economic_data,
            "sentiment": sentiment_data
        }
    
    def format_data_for_agents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formatta i dati nel formato richiesto dagli agenti AlphaPulse.
        
        Args:
            data: Dati grezzi
            
        Returns:
            Dati formattati per gli agenti
        """
        logger.info("Formattazione dati per agenti AlphaPulse")
        
        # Crea DataFrame di prezzi
        prices_df = data.get("prices", pd.DataFrame())
        if prices_df.empty:
            logger.error("Nessun dato di prezzo disponibile")
            return {}
        
        # Crea DataFrame di volumi
        volumes_df = None
        if data.get("volumes") is not None:
            volumes_df = pd.DataFrame(data["volumes"])
        
        # Prepara dati fondamentali
        fundamentals = {}
        
        # Prepara dati di sentiment
        sentiment = {}
        sentiment_df = data.get("sentiment")
        if sentiment_df is not None and not sentiment_df.empty:
            # Converti in dizionario di sentiment per data
            for date, row in sentiment_df.iterrows():
                sentiment[str(date)] = float(row['sentiment_score'])
        
        # Prepara indicatori tecnici
        technical_indicators = {}
        
        # Crea struttura dati compatibile con gli agenti
        agent_data = {
            'prices': prices_df,
            'volumes': volumes_df,
            'fundamentals': fundamentals,
            'sentiment': sentiment,
            'technical_indicators': technical_indicators,
            'timestamp': datetime.now(),
            'data_by_symbol': {"^GSPC": prices_df.to_dict('records')}
        }
        
        logger.info("Dati formattati per agenti AlphaPulse")
        
        return agent_data
    
    def save_data(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Salva i dati in un file.
        
        Args:
            data: Dati da salvare
            filename: Nome del file
            
        Returns:
            True se il salvataggio è riuscito, False altrimenti
        """
        try:
            file_path = self.cache_dir / filename
            
            # Determina il formato in base all'estensione
            if filename.endswith('.parquet'):
                pd.DataFrame(data).to_parquet(file_path)
            elif filename.endswith('.csv'):
                pd.DataFrame(data).to_csv(file_path)
            elif filename.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                logger.error(f"Formato file non supportato: {filename}")
                return False
            
            logger.info(f"Dati salvati in {file_path}")
            return True
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei dati: {str(e)}")
            return False
    
    def load_data(self, filename: str) -> Dict[str, Any]:
        """
        Carica i dati da un file.
        
        Args:
            filename: Nome del file
            
        Returns:
            Dati caricati
        """
        try:
            file_path = self.cache_dir / filename
            
            if not file_path.exists():
                logger.error(f"File non trovato: {file_path}")
                return {}
            
            # Determina il formato in base all'estensione
            if filename.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            elif filename.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                logger.error(f"Formato file non supportato: {filename}")
                return {}
            
            logger.info(f"Dati caricati da {file_path}")
            return data
        except Exception as e:
            logger.error(f"Errore nel caricamento dei dati: {str(e)}")
            return {}


if __name__ == "__main__":
    # Esempio di utilizzo
    config = {
        "cache_dir": "./data/cache"
    }
    
    data_manager = DataManager(config)
    
    # Recupera dati per un periodo di test
    market_data = data_manager.prepare_market_data("2020-01-01", "2020-12-31")
    
    if market_data:
        # Formatta i dati per gli agenti
        agent_data = data_manager.format_data_for_agents(market_data)
        
        # Salva i dati formattati
        data_manager.save_data(agent_data, "agent_data_2020.json")
        
        print("Dati preparati e salvati con successo")
    else:
        print("Errore nella preparazione dei dati")