# Backtest S&P 500 con AlphaPulse

Questo progetto implementa un backtest completo dell'indice S&P 500 utilizzando il sistema multi-agente AlphaPulse, con focus sulla gestione del rischio e l'ottimizzazione del portafoglio.

## Configurazione Iniziale

### Prerequisiti
- Python 3.9+
- PostgreSQL con TimescaleDB
- Redis (opzionale)

### Installazione Dipendenze

```bash
# Crea un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

## API Keys

Le API keys sono memorizzate nel file `.env` che non deve essere condiviso o committato nel repository. Il file è già configurato con le seguenti chiavi:

- **FRED API**: Per dati economici e dell'indice S&P 500
- **NewsAPI**: Per dati di sentiment basati su notizie finanziarie

### Test delle API

Per verificare che le API keys funzionino correttamente, esegui lo script di esempio:

```bash
python api_examples.py
```

Questo script:
1. Testa la connessione all'API FRED recuperando dati storici dell'S&P 500
2. Testa la connessione a NewsAPI recuperando notizie recenti sull'S&P 500
3. Salva esempi di dati in file locali per riferimento

## Struttura del Progetto

- `SP500_Backtest_Requisiti.md`: Documento dettagliato dei requisiti
- `.env`: File con le API keys (non committare)
- `api_examples.py`: Script di esempio per l'utilizzo delle API
- `requirements.txt`: Dipendenze Python

## Implementazione

Per implementare il backtest completo, seguire le istruzioni dettagliate nel documento `SP500_Backtest_Requisiti.md`, che include:

1. Configurazione dell'infrastruttura
2. Preparazione dei dati
3. Adattamento degli agenti AlphaPulse
4. Esecuzione del backtest
5. Validazione e reporting

## Utilizzo delle API

### FRED API

```python
import os
import requests
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Esempio di richiesta per dati S&P 500
base_url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "SP500",  # Serie S&P 500
    "api_key": FRED_API_KEY,
    "file_type": "json",
    "observation_start": "2020-01-01",
    "observation_end": "2020-12-31"
}

response = requests.get(base_url, params=params)
data = response.json()
```

### NewsAPI

```python
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Data di una settimana fa
one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

# Esempio di richiesta per notizie sull'S&P 500
base_url = "https://newsapi.org/v2/everything"
params = {
    "q": "S&P 500 OR S&P500",
    "from": one_week_ago,
    "language": "en",
    "sortBy": "popularity",
    "apiKey": NEWS_API_KEY
}

response = requests.get(base_url, params=params)
data = response.json()
```

## Sicurezza

- Non committare mai il file `.env` nel repository
- Ruotare periodicamente le API keys
- Limitare l'accesso alle API keys solo a chi ne ha bisogno

## Supporto

Per domande o problemi, contattare il team di sviluppo.