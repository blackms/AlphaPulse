# Real Data Implementation Plan

## Data Sources Integration

### 1. Market Data (Prices & Volumes)
- **Primary: Financial Exchange APIs**
  - Binance API (already integrated)
  - Bybit API (already integrated)
  - Add Interactive Brokers API for stocks
  - Add Alpaca API for US stocks

- **Implementation Steps:**
  ```python
  from alpha_pulse.data_pipeline.providers import ExchangeProvider
  from alpha_pulse.exchanges import BinanceExchange, IBKRExchange, AlpacaExchange
  
  # Configure providers
  exchange_providers = {
      'crypto': BinanceExchange(credentials),
      'stocks': IBKRExchange(credentials),
      'us_stocks': AlpacaExchange(credentials)
  }
  
  # Fetch market data
  async def get_market_data(symbols, timeframe='1d', lookback_days=365):
      data = {}
      for symbol in symbols:
          provider = determine_provider(symbol)
          data[symbol] = await provider.fetch_ohlcv(
              symbol=symbol,
              timeframe=timeframe,
              limit=lookback_days
          )
      return data
  ```

### 2. Fundamental Data
- **Primary Sources:**
  - Financial Modeling Prep API (comprehensive financial data)
  - Alpha Vantage (fundamental data)
  - SEC EDGAR API (filings and reports)
  - Yahoo Finance API (basic fundamentals)

- **Implementation:**
  ```python
  from alpha_pulse.data_pipeline.providers import (
      FMPProvider,
      AlphaVantageProvider,
      EDGARProvider
  )
  
  class FundamentalDataManager:
      def __init__(self):
          self.fmp = FMPProvider(api_key)
          self.alpha_vantage = AlphaVantageProvider(api_key)
          self.edgar = EDGARProvider()
          
      async def get_fundamentals(self, symbol):
          # Financial statements
          financials = await self.fmp.get_financial_statements(symbol)
          
          # Key metrics
          metrics = await self.alpha_vantage.get_company_overview(symbol)
          
          # SEC filings
          filings = await self.edgar.get_recent_filings(symbol)
          
          return self._process_fundamental_data(
              financials, metrics, filings
          )
  ```

### 3. News & Sentiment Data
- **Primary Sources:**
  - NewsAPI (general news)
  - Twitter API v2 (social sentiment)
  - Reddit API (social sentiment)
  - Finnhub (financial news)
  - Bloomberg API (premium news)

- **Implementation:**
  ```python
  from alpha_pulse.data_pipeline.providers import (
      NewsAPIProvider,
      TwitterProvider,
      RedditProvider,
      FinnhubProvider
  )
  
  class SentimentDataManager:
      def __init__(self):
          self.news_api = NewsAPIProvider(api_key)
          self.twitter = TwitterProvider(credentials)
          self.reddit = RedditProvider(credentials)
          self.finnhub = FinnhubProvider(api_key)
          
      async def get_sentiment_data(self, symbol):
          # Gather news
          news = await self.news_api.get_company_news(symbol)
          
          # Social media sentiment
          tweets = await self.twitter.get_cashtag_sentiment(symbol)
          reddit_posts = await self.reddit.get_stock_sentiment(symbol)
          
          # Financial news
          financial_news = await self.finnhub.get_company_news(symbol)
          
          return self._analyze_sentiment(
              news, tweets, reddit_posts, financial_news
          )
  ```

### 4. Technical Indicators
- **Implementation using TA-Lib:**
  ```python
  import talib
  import numpy as np
  
  class TechnicalAnalysisManager:
      def __init__(self):
          self.indicators = {
              'trend': ['SMA', 'EMA', 'MACD'],
              'momentum': ['RSI', 'STOCH', 'ADX'],
              'volatility': ['ATR', 'BBANDS'],
              'volume': ['OBV', 'AD', 'ADOSC']
          }
          
      def calculate_indicators(self, ohlcv_data):
          results = {}
          
          # Convert data to numpy arrays
          opens = np.array(ohlcv_data['open'])
          highs = np.array(ohlcv_data['high'])
          lows = np.array(ohlcv_data['low'])
          closes = np.array(ohlcv_data['close'])
          volumes = np.array(ohlcv_data['volume'])
          
          # Calculate indicators
          results['sma'] = talib.SMA(closes)
          results['ema'] = talib.EMA(closes)
          results['macd'], results['macd_signal'], _ = talib.MACD(closes)
          results['rsi'] = talib.RSI(closes)
          results['upper'], results['middle'], results['lower'] = talib.BBANDS(closes)
          
          return results
  ```

## Integration Steps

1. **Create Data Pipeline Manager:**
```python
class RealDataManager:
    def __init__(self):
        self.market_data = MarketDataManager()
        self.fundamentals = FundamentalDataManager()
        self.sentiment = SentimentDataManager()
        self.technicals = TechnicalAnalysisManager()
        
    async def get_market_data(self, symbols: List[str]) -> MarketData:
        # Fetch all data concurrently
        prices_task = self.market_data.get_prices(symbols)
        fundamentals_task = self.fundamentals.get_fundamentals(symbols)
        sentiment_task = self.sentiment.get_sentiment_data(symbols)
        
        # Gather results
        prices, fundamentals, sentiment = await asyncio.gather(
            prices_task, fundamentals_task, sentiment_task
        )
        
        # Calculate technical indicators
        technicals = self.technicals.calculate_indicators(prices)
        
        return MarketData(
            prices=prices,
            volumes=prices['volume'],
            fundamentals=fundamentals,
            sentiment=sentiment,
            technical_indicators=technicals,
            timestamp=datetime.now()
        )
```

2. **API Key Management:**
```python
class APIKeyManager:
    def __init__(self):
        self.keys = self._load_api_keys()
        
    def _load_api_keys(self):
        # Load from environment or secure storage
        return {
            'fmp': os.getenv('FMP_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'news_api': os.getenv('NEWS_API_KEY'),
            'twitter': {
                'api_key': os.getenv('TWITTER_API_KEY'),
                'api_secret': os.getenv('TWITTER_API_SECRET')
            },
            # ... other API keys
        }
```

3. **Error Handling and Retries:**
```python
class DataFetchRetry:
    def __init__(self, max_retries=3, delay=1):
        self.max_retries = max_retries
        self.delay = delay
        
    async def fetch_with_retry(self, fetch_func):
        for attempt in range(self.max_retries):
            try:
                return await fetch_func()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.delay * (attempt + 1))
```

4. **Data Validation:**
```python
class DataValidator:
    def validate_market_data(self, data: MarketData) -> bool:
        if data.prices.empty:
            return False
        if data.prices.isnull().any().any():
            # Handle missing values
            data.prices = data.prices.fillna(method='ffill')
        return True
```

## Configuration

Create `config/data_sources.yaml`:
```yaml
market_data:
  providers:
    - name: binance
      type: crypto
      priority: 1
    - name: interactive_brokers
      type: stocks
      priority: 1
    - name: alpaca
      type: us_stocks
      priority: 2

fundamental_data:
  providers:
    - name: financial_modeling_prep
      priority: 1
    - name: alpha_vantage
      priority: 2
    - name: yahoo_finance
      priority: 3

sentiment_data:
  providers:
    - name: news_api
      weight: 0.3
    - name: twitter
      weight: 0.3
    - name: reddit
      weight: 0.2
    - name: finnhub
      weight: 0.2

technical_analysis:
  indicators:
    trend:
      - SMA
      - EMA
      - MACD
    momentum:
      - RSI
      - STOCH
      - ADX
    volatility:
      - ATR
      - BBANDS
    volume:
      - OBV
      - AD
      - ADOSC
```

## Implementation Timeline

1. **Phase 1: Market Data (Week 1-2)**
   - Implement exchange integrations
   - Set up data fetching and caching
   - Add error handling and validation

2. **Phase 2: Fundamental Data (Week 3-4)**
   - Implement FMP and Alpha Vantage integration
   - Add SEC EDGAR integration
   - Set up data processing and storage

3. **Phase 3: Sentiment Data (Week 5-6)**
   - Implement news and social media APIs
   - Set up sentiment analysis pipeline
   - Add data aggregation and scoring

4. **Phase 4: Technical Analysis (Week 7)**
   - Implement TA-Lib indicators
   - Add custom indicators
   - Set up real-time calculation

5. **Phase 5: Integration & Testing (Week 8)**
   - Integrate all data sources
   - Add monitoring and logging
   - Comprehensive testing
   - Performance optimization

## Required Dependencies

Add to `setup.py`:
```python
install_requires=[
    # Existing dependencies...
    "financial-modeling-prep-api",
    "alpha_vantage",
    "newsapi-python",
    "tweepy",
    "praw",
    "finnhub-python",
    "sec-edgar-downloader",
    "yfinance",
    "ibkr-api",
    "alpaca-trade-api",
]