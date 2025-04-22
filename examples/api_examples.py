#!/usr/bin/env python3
"""
Esempi di utilizzo delle API FRED e NewsAPI per il progetto di backtest S&P 500.
Questo script dimostra come caricare le API keys dal file .env e utilizzarle
per effettuare richieste alle API.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni le API keys dalle variabili d'ambiente
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def test_fred_api():
    """Test della connessione all'API FRED."""
    print("=== Test FRED API ===")
    
    # URL di base per l'API FRED
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Parametri per la richiesta (S&P 500 Index)
    params = {
        "series_id": "SP500",  # Serie S&P 500
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2020-01-01",
        "observation_end": "2020-12-31"
    }
    
    # Effettua la richiesta
    response = requests.get(base_url, params=params)
    
    # Verifica la risposta
    if response.status_code == 200:
        data = response.json()
        observations = data.get("observations", [])
        
        # Converti in DataFrame
        df = pd.DataFrame(observations)
        
        print(f"Connessione a FRED API riuscita!")
        print(f"Dati ricevuti: {len(observations)} osservazioni")
        print(f"Prime 5 righe:")
        print(df.head())
        
        # Salva i dati in un file CSV
        df.to_csv("fred_sp500_data_sample.csv", index=False)
        print("Dati salvati in 'fred_sp500_data_sample.csv'")
    else:
        print(f"Errore nella richiesta: {response.status_code}")
        print(response.text)

def test_news_api():
    """Test della connessione all'API NewsAPI."""
    print("\n=== Test NewsAPI ===")
    
    # URL di base per NewsAPI
    base_url = "https://newsapi.org/v2/everything"
    
    # Data di una settimana fa
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Parametri per la richiesta (notizie su S&P 500)
    params = {
        "q": "S&P 500 OR S&P500 OR 'Standard and Poor'",
        "from": one_week_ago,
        "language": "en",
        "sortBy": "popularity",
        "apiKey": NEWS_API_KEY
    }
    
    # Effettua la richiesta
    response = requests.get(base_url, params=params)
    
    # Verifica la risposta
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        
        print(f"Connessione a NewsAPI riuscita!")
        print(f"Articoli trovati: {len(articles)}")
        
        if articles:
            print("\nPrimi 3 articoli:")
            for i, article in enumerate(articles[:3]):
                print(f"{i+1}. {article['title']}")
                print(f"   Fonte: {article['source']['name']}")
                print(f"   Data: {article['publishedAt']}")
                print(f"   URL: {article['url']}")
                print()
        
        # Salva i dati in un file JSON
        with open("newsapi_sample.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Dati salvati in 'newsapi_sample.json'")
    else:
        print(f"Errore nella richiesta: {response.status_code}")
        print(response.text)

def main():
    """Funzione principale."""
    # Verifica che le API keys siano state caricate
    if not FRED_API_KEY:
        print("Errore: FRED API Key non trovata nel file .env")
        return
    
    if not NEWS_API_KEY:
        print("Errore: NewsAPI Key non trovata nel file .env")
        return
    
    # Testa le API
    test_fred_api()
    test_news_api()

if __name__ == "__main__":
    main()