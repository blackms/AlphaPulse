# AlphaPulse Architecture Diagrams

This document provides comprehensive architectural diagrams for the AlphaPulse AI-powered algorithmic trading system using C4 model, data flow diagrams, sequence diagrams, and other visualization methods.

## Table of Contents
1. [C4 Model Diagrams](#c4-model-diagrams)
   - [Level 1: System Context](#level-1-system-context)
   - [Level 2: Container Diagram](#level-2-container-diagram)
   - [Level 3: Component Diagrams](#level-3-component-diagrams)
2. [Data Flow Diagrams](#data-flow-diagrams)
3. [Sequence Diagrams](#sequence-diagrams)
4. [Deployment Diagram](#deployment-diagram)
5. [State Diagrams](#state-diagrams)
6. [Entity Relationship Diagram](#entity-relationship-diagram)

## C4 Model Diagrams

### Level 1: System Context

```mermaid
C4Context
    title System Context diagram for AlphaPulse Trading System

    Person(trader, "Trader", "Human trader or investor using the system")
    Person(admin, "System Admin", "Manages and monitors the system")
    
    System(alphapulse, "AlphaPulse", "AI-powered algorithmic trading system that operates as an automated hedge fund")
    
    System_Ext(exchanges, "Cryptocurrency Exchanges", "Trading venues (Binance, Coinbase, etc.)")
    System_Ext(dataProviders, "Market Data Providers", "Real-time and historical market data")
    System_Ext(newsAPIs, "News & Sentiment APIs", "Financial news and social sentiment data")
    System_Ext(fundamentalData, "Fundamental Data Sources", "Company financials, economic indicators")
    System_Ext(monitoring, "Monitoring Systems", "Prometheus, Grafana, AlertManager")
    
    Rel(trader, alphapulse, "Configures strategies, monitors performance", "HTTPS/WebSocket")
    Rel(admin, alphapulse, "Manages system, handles alerts", "HTTPS/SSH")
    Rel(alphapulse, exchanges, "Executes trades, fetches prices", "REST/WebSocket")
    Rel(dataProviders, alphapulse, "Provides market data", "REST/WebSocket")
    Rel(newsAPIs, alphapulse, "Provides news and sentiment", "REST")
    Rel(fundamentalData, alphapulse, "Provides fundamental data", "REST")
    Rel(alphapulse, monitoring, "Exports metrics and logs", "Prometheus/Loki")
```

### Level 2: Container Diagram

```mermaid
C4Container
    title Container diagram for AlphaPulse Trading System

    Person(trader, "Trader", "Human trader or investor")
    Person(admin, "System Admin", "System administrator")
    
    Container_Boundary(alphapulse, "AlphaPulse System") {
        Container(web, "Web Dashboard", "React, TypeScript", "Trading interface and monitoring dashboard")
        Container(api, "API Gateway", "FastAPI, Python", "REST API and WebSocket server")
        Container(tradingEngine, "Trading Engine", "Python", "Core trading logic and orchestration")
        Container(agents, "AI Trading Agents", "Python, PyTorch", "6 specialized trading agents")
        Container(riskMgmt, "Risk Management", "Python", "Position sizing, stop-loss, risk controls")
        Container(portfolio, "Portfolio Manager", "Python", "Portfolio optimization and allocation")
        Container(execution, "Execution Engine", "Python, CCXT", "Order management and execution")
        Container(dataPipeline, "Data Pipeline", "Python", "Real-time data processing")
        Container(ml, "ML Services", "Python, PyTorch", "RL agents and ML models")
        Container(monitoring, "Monitoring Service", "Python", "Metrics collection and alerting")
        
        ContainerDb(timeseries, "TimescaleDB", "PostgreSQL", "Time-series market data")
        ContainerDb(operational, "PostgreSQL", "PostgreSQL", "System state and configuration")
        ContainerDb(redis, "Redis Cache", "Redis", "Real-time data cache")
    }
    
    System_Ext(exchanges, "Exchanges", "Trading venues")
    System_Ext(dataFeeds, "Data Feeds", "Market data providers")
    System_Ext(prometheus, "Prometheus", "Metrics storage")
    
    Rel(trader, web, "Uses", "HTTPS")
    Rel(web, api, "Makes API calls", "HTTPS/WSS")
    Rel(api, tradingEngine, "Forwards requests", "gRPC")
    Rel(tradingEngine, agents, "Orchestrates", "Internal")
    Rel(tradingEngine, riskMgmt, "Validates trades", "Internal")
    Rel(tradingEngine, portfolio, "Optimizes allocation", "Internal")
    Rel(tradingEngine, execution, "Executes trades", "Internal")
    Rel(agents, dataPipeline, "Consumes data", "Internal")
    Rel(dataPipeline, redis, "Caches data", "TCP")
    Rel(dataPipeline, timeseries, "Stores history", "TCP")
    Rel(execution, exchanges, "Sends orders", "HTTPS/WSS")
    Rel(dataFeeds, dataPipeline, "Streams data", "WSS")
    Rel(monitoring, prometheus, "Exports metrics", "HTTP")
    Rel(ml, agents, "Provides models", "Internal")
```

### Level 3: Component Diagrams

#### Trading Agents Component Diagram

```mermaid
C4Component
    title Component diagram for AI Trading Agents

    Container_Boundary(agents, "AI Trading Agents Container") {
        Component(orchestrator, "Agent Orchestrator", "Python", "Coordinates agent execution")
        Component(technical, "Technical Agent", "Python", "Technical analysis signals")
        Component(fundamental, "Fundamental Agent", "Python", "Value and financial analysis")
        Component(sentiment, "Sentiment Agent", "Python, NLP", "News and social sentiment")
        Component(value, "Value Agent", "Python", "Long-term value investing")
        Component(activist, "Activist Agent", "Python", "Event-driven strategies")
        Component(buffett, "Buffett Agent", "Python", "Warren Buffett style analysis")
        Component(signalAgg, "Signal Aggregator", "Python", "Combines and weights signals")
        Component(confidence, "Confidence Scorer", "Python", "Calculates signal confidence")
    }
    
    Container(dataPipeline, "Data Pipeline", "Python", "Market data provider")
    Container(ml, "ML Services", "Python", "ML model inference")
    Container(riskMgmt, "Risk Management", "Python", "Risk validation")
    
    Rel(orchestrator, technical, "Executes", "Internal call")
    Rel(orchestrator, fundamental, "Executes", "Internal call")
    Rel(orchestrator, sentiment, "Executes", "Internal call")
    Rel(orchestrator, value, "Executes", "Internal call")
    Rel(orchestrator, activist, "Executes", "Internal call")
    Rel(orchestrator, buffett, "Executes", "Internal call")
    
    Rel(technical, dataPipeline, "Fetches OHLCV data", "API call")
    Rel(fundamental, dataPipeline, "Fetches financials", "API call")
    Rel(sentiment, dataPipeline, "Fetches news", "API call")
    Rel(sentiment, ml, "Uses NLP models", "API call")
    
    Rel(technical, signalAgg, "Provides signals", "Internal")
    Rel(fundamental, signalAgg, "Provides signals", "Internal")
    Rel(sentiment, signalAgg, "Provides signals", "Internal")
    Rel(value, signalAgg, "Provides signals", "Internal")
    Rel(activist, signalAgg, "Provides signals", "Internal")
    Rel(buffett, signalAgg, "Provides signals", "Internal")
    
    Rel(signalAgg, confidence, "Calculates confidence", "Internal")
    Rel(confidence, riskMgmt, "Sends weighted signals", "API call")
```

#### Risk Management Component Diagram

```mermaid
C4Component
    title Component diagram for Risk Management System

    Container_Boundary(riskMgmt, "Risk Management Container") {
        Component(riskEngine, "Risk Engine", "Python", "Core risk calculation engine")
        Component(positionSizer, "Position Sizer", "Python", "Kelly Criterion & fixed fractional sizing")
        Component(stopLoss, "Stop Loss Manager", "Python", "Dynamic stop-loss calculation")
        Component(leverage, "Leverage Controller", "Python", "Manages position leverage")
        Component(drawdown, "Drawdown Monitor", "Python", "Tracks and limits drawdowns")
        Component(correlation, "Correlation Analyzer", "Python", "Portfolio correlation analysis")
        Component(var, "VaR Calculator", "Python", "Value at Risk calculations")
        Component(stress, "Stress Tester", "Python", "Scenario analysis")
        Component(limits, "Limit Enforcer", "Python", "Enforces risk limits")
    }
    
    Container(portfolio, "Portfolio Manager", "Python", "Portfolio state")
    Container(execution, "Execution Engine", "Python", "Trade executor")
    Container(monitoring, "Monitoring", "Python", "Alerting system")
    
    Rel(riskEngine, positionSizer, "Calculates size", "Internal")
    Rel(riskEngine, stopLoss, "Sets stops", "Internal")
    Rel(riskEngine, leverage, "Checks leverage", "Internal")
    Rel(riskEngine, drawdown, "Monitors DD", "Internal")
    Rel(riskEngine, correlation, "Analyzes", "Internal")
    Rel(riskEngine, var, "Calculates VaR", "Internal")
    Rel(riskEngine, stress, "Runs scenarios", "Internal")
    Rel(limits, riskEngine, "Provides limits", "Internal")
    
    Rel(portfolio, riskEngine, "Provides positions", "API call")
    Rel(riskEngine, execution, "Approves trades", "API call")
    Rel(riskEngine, monitoring, "Sends alerts", "Event")
```

## Data Flow Diagrams

### Real-time Data Processing Flow

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        EX[Exchange APIs]
        NEWS[News APIs]
        FUND[Fundamental Data]
        SOCIAL[Social Media]
    end
    
    subgraph DataPipeline["Data Pipeline Layer"]
        FETCH[Data Fetchers]
        NORM[Normalizers]
        VALID[Validators]
        CACHE[Redis Cache]
        QUEUE[Message Queue]
    end
    
    subgraph Processing["Processing Layer"]
        TECH[Technical Indicators]
        SENT[Sentiment Analysis]
        FUND_PROC[Fundamental Processing]
        ML_FEAT[Feature Engineering]
    end
    
    subgraph Storage["Storage Layer"]
        TS[(TimescaleDB)]
        PG[(PostgreSQL)]
        S3[S3 Compatible Storage]
    end
    
    subgraph Consumers["Data Consumers"]
        AGENTS[Trading Agents]
        RISK[Risk Management]
        DASH[Dashboard]
        BACK[Backtesting]
    end
    
    EX -->|WebSocket/REST| FETCH
    NEWS -->|REST| FETCH
    FUND -->|REST| FETCH
    SOCIAL -->|REST| FETCH
    
    FETCH --> NORM
    NORM --> VALID
    VALID --> CACHE
    VALID --> QUEUE
    
    QUEUE --> TECH
    QUEUE --> SENT
    QUEUE --> FUND_PROC
    QUEUE --> ML_FEAT
    
    TECH --> TS
    SENT --> PG
    FUND_PROC --> PG
    ML_FEAT --> S3
    
    CACHE --> AGENTS
    CACHE --> RISK
    CACHE --> DASH
    
    TS --> BACK
    PG --> AGENTS
    S3 --> AGENTS
```

### Trading Signal Flow

```mermaid
flowchart LR
    subgraph Agents["AI Trading Agents"]
        TA[Technical Agent]
        FA[Fundamental Agent]
        SA[Sentiment Agent]
        VA[Value Agent]
        AA[Activist Agent]
        BA[Buffett Agent]
    end
    
    subgraph Aggregation["Signal Processing"]
        AGG[Signal Aggregator]
        CONF[Confidence Scorer]
        WEIGHT[Weight Calculator]
    end
    
    subgraph Risk["Risk Layer"]
        RCHECK[Risk Checker]
        PSIZE[Position Sizer]
        STOP[Stop Loss Calc]
    end
    
    subgraph Portfolio["Portfolio Layer"]
        OPT[Portfolio Optimizer]
        ALLOC[Asset Allocator]
        REBAL[Rebalancer]
    end
    
    subgraph Execution["Execution Layer"]
        OBOOK[Order Book Analyzer]
        OMAN[Order Manager]
        EXEC[Trade Executor]
    end
    
    TA -->|Technical Signals| AGG
    FA -->|Value Signals| AGG
    SA -->|Sentiment Signals| AGG
    VA -->|Value Signals| AGG
    AA -->|Event Signals| AGG
    BA -->|Quality Signals| AGG
    
    AGG --> CONF
    CONF --> WEIGHT
    
    WEIGHT --> RCHECK
    RCHECK -->|Approved| PSIZE
    RCHECK -->|Rejected| AGG
    PSIZE --> STOP
    
    STOP --> OPT
    OPT --> ALLOC
    ALLOC --> REBAL
    
    REBAL --> OBOOK
    OBOOK --> OMAN
    OMAN --> EXEC
    
    EXEC -->|Feedback| AGG
```

## Sequence Diagrams

### Trade Execution Sequence

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant API
    participant TradingEngine
    participant Agents
    participant RiskMgmt
    participant Portfolio
    participant Execution
    participant Exchange
    
    User->>Dashboard: Configure strategy
    Dashboard->>API: POST /strategies
    API->>TradingEngine: Create strategy
    
    loop Trading Loop
        TradingEngine->>Agents: Request signals
        Agents->>Agents: Analyze market
        Agents-->>TradingEngine: Return signals
        
        TradingEngine->>RiskMgmt: Validate signals
        RiskMgmt->>RiskMgmt: Check limits
        RiskMgmt->>RiskMgmt: Calculate position size
        RiskMgmt-->>TradingEngine: Approved trades
        
        TradingEngine->>Portfolio: Optimize allocation
        Portfolio->>Portfolio: Run optimization
        Portfolio-->>TradingEngine: Optimal weights
        
        TradingEngine->>Execution: Execute trades
        Execution->>Exchange: Place orders
        Exchange-->>Execution: Order confirmation
        Execution-->>TradingEngine: Execution report
        
        TradingEngine->>Dashboard: Update status
        Dashboard->>User: Show results
    end
```

### Real-time Data Processing Sequence

```mermaid
sequenceDiagram
    participant Exchange
    participant DataPipeline
    participant Cache
    participant TimescaleDB
    participant Agents
    participant Dashboard
    
    Exchange->>DataPipeline: WebSocket stream
    
    loop Real-time Processing
        DataPipeline->>DataPipeline: Normalize data
        DataPipeline->>DataPipeline: Validate data
        
        par Parallel Processing
            DataPipeline->>Cache: Store in Redis
        and
            DataPipeline->>TimescaleDB: Store history
        end
        
        Cache-->>Agents: Broadcast update
        Cache-->>Dashboard: Push update
        
        Agents->>Agents: Process new data
        Dashboard->>Dashboard: Update UI
    end
```

### Agent Decision Making Sequence

```mermaid
sequenceDiagram
    participant Orchestrator
    participant TechnicalAgent
    participant FundamentalAgent
    participant SentimentAgent
    participant MLService
    participant Aggregator
    
    Orchestrator->>TechnicalAgent: Request analysis
    Orchestrator->>FundamentalAgent: Request analysis
    Orchestrator->>SentimentAgent: Request analysis
    
    par Parallel Analysis
        TechnicalAgent->>TechnicalAgent: Calculate indicators
        TechnicalAgent->>TechnicalAgent: Identify patterns
    and
        FundamentalAgent->>FundamentalAgent: Analyze financials
        FundamentalAgent->>FundamentalAgent: Calculate ratios
    and
        SentimentAgent->>MLService: Get NLP analysis
        MLService-->>SentimentAgent: Sentiment scores
        SentimentAgent->>SentimentAgent: Process sentiment
    end
    
    TechnicalAgent-->>Aggregator: Technical signals
    FundamentalAgent-->>Aggregator: Fundamental signals
    SentimentAgent-->>Aggregator: Sentiment signals
    
    Aggregator->>Aggregator: Weight signals
    Aggregator->>Aggregator: Calculate confidence
    Aggregator-->>Orchestrator: Final signal
```

## Deployment Diagram

```mermaid
flowchart TB
    subgraph Cloud["Cloud Infrastructure"]
        subgraph K8s["Kubernetes Cluster"]
            subgraph Services["Service Pods"]
                API[API Gateway<br/>3 replicas]
                TE[Trading Engine<br/>2 replicas]
                DP[Data Pipeline<br/>3 replicas]
                MON[Monitoring<br/>2 replicas]
            end
            
            subgraph Agents["Agent Pods"]
                AG1[Agent Set 1<br/>6 agents]
                AG2[Agent Set 2<br/>6 agents]
            end
            
            subgraph ML["ML Pods"]
                MLS[ML Service<br/>GPU enabled]
                RL[RL Training<br/>GPU enabled]
            end
        end
        
        subgraph Data["Data Layer"]
            subgraph DBCluster["Database Cluster"]
                TS1[(TimescaleDB Primary)]
                TS2[(TimescaleDB Replica)]
                PG1[(PostgreSQL Primary)]
                PG2[(PostgreSQL Replica)]
            end
            
            subgraph Cache["Cache Layer"]
                R1[Redis Primary]
                R2[Redis Replica]
            end
            
            S3[S3 Storage<br/>Model artifacts]
        end
        
        LB[Load Balancer]
        CDN[CDN<br/>Static assets]
    end
    
    subgraph External["External Services"]
        EX[Exchange APIs]
        DATA[Data Providers]
        PROM[Prometheus]
        GRAF[Grafana]
    end
    
    subgraph Users["Users"]
        TRADER[Traders]
        ADMIN[Admins]
    end
    
    TRADER --> CDN
    ADMIN --> LB
    CDN --> LB
    LB --> API
    
    API --> TE
    TE --> AG1
    TE --> AG2
    
    AG1 --> MLS
    AG2 --> MLS
    
    DP --> R1
    DP --> TS1
    
    TE --> PG1
    MLS --> S3
    
    MON --> PROM
    PROM --> GRAF
    
    EX --> DP
    DATA --> DP
```

## State Diagrams

### Order Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> SignalGenerated
    
    SignalGenerated --> RiskValidation
    
    RiskValidation --> Rejected: Risk check failed
    RiskValidation --> Approved: Risk check passed
    
    Rejected --> [*]
    
    Approved --> PortfolioOptimization
    
    PortfolioOptimization --> OrderCreation
    
    OrderCreation --> OrderPlaced
    
    OrderPlaced --> PartiallyFilled: Partial execution
    OrderPlaced --> Filled: Full execution
    OrderPlaced --> Cancelled: User cancelled
    OrderPlaced --> Failed: Exchange rejected
    
    PartiallyFilled --> Filled: Remaining filled
    PartiallyFilled --> Cancelled: Timeout/User action
    
    Filled --> PositionOpen
    
    PositionOpen --> PositionClosed: Exit signal/Stop loss
    
    PositionClosed --> [*]
    
    Failed --> [*]
    Cancelled --> [*]
```

### System Health State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> HealthCheck
    
    HealthCheck --> Healthy: All checks pass
    HealthCheck --> Degraded: Some checks fail
    HealthCheck --> Critical: Critical checks fail
    
    Healthy --> Monitoring: Normal operation
    
    Monitoring --> Healthy: Periodic check OK
    Monitoring --> Warning: Minor issues
    Monitoring --> Degraded: Service issues
    
    Warning --> Monitoring: Issues resolved
    Warning --> Degraded: Issues escalate
    
    Degraded --> Recovery: Auto-healing
    Degraded --> Critical: Further degradation
    
    Recovery --> HealthCheck: Recovery complete
    
    Critical --> EmergencyStop: Safety shutdown
    
    EmergencyStop --> [*]
```

## Entity Relationship Diagram

```mermaid
erDiagram
    TRADER ||--o{ STRATEGY : creates
    STRATEGY ||--o{ POSITION : generates
    POSITION ||--o{ ORDER : contains
    ORDER ||--o{ EXECUTION : results_in
    
    STRATEGY ||--o{ AGENT_CONFIG : uses
    AGENT_CONFIG ||--|| AGENT : configures
    
    POSITION ||--|| ASSET : trades
    ASSET ||--o{ MARKET_DATA : has
    ASSET ||--o{ FUNDAMENTAL_DATA : has
    
    ORDER ||--|| EXCHANGE : sent_to
    EXCHANGE ||--o{ EXECUTION : processes
    
    POSITION ||--o{ RISK_METRIC : tracked_by
    RISK_METRIC ||--|| RISK_LIMIT : constrained_by
    
    EXECUTION ||--o{ TRANSACTION : creates
    TRANSACTION ||--|| PORTFOLIO : affects
    
    PORTFOLIO ||--o{ PERFORMANCE_METRIC : measured_by
    
    TRADER {
        string trader_id PK
        string name
        string email
        json preferences
        timestamp created_at
    }
    
    STRATEGY {
        string strategy_id PK
        string trader_id FK
        string name
        json configuration
        string status
        timestamp created_at
    }
    
    POSITION {
        string position_id PK
        string strategy_id FK
        string asset_id FK
        decimal quantity
        decimal entry_price
        decimal current_price
        decimal pnl
        string status
        timestamp opened_at
        timestamp closed_at
    }
    
    ORDER {
        string order_id PK
        string position_id FK
        string exchange_id FK
        string type
        string side
        decimal quantity
        decimal price
        string status
        timestamp created_at
    }
    
    ASSET {
        string asset_id PK
        string symbol
        string name
        string type
        string exchange
        json metadata
    }
    
    MARKET_DATA {
        string asset_id FK
        timestamp time
        decimal open
        decimal high
        decimal low
        decimal close
        decimal volume
    }
```

## Architecture Decision Records (ADR)

### ADR-001: Microservice vs Monolithic Architecture

**Status**: Accepted

**Context**: Need to decide between microservice and monolithic architecture for AlphaPulse.

**Decision**: Hybrid approach - monolithic core with service boundaries that can be easily extracted to microservices.

**Consequences**:
- Easier initial development and deployment
- Clear service boundaries for future scaling
- Can extract services as needed without major refactoring

### ADR-002: Time-Series Database Selection

**Status**: Accepted

**Context**: Need specialized time-series storage for market data.

**Decision**: TimescaleDB (PostgreSQL extension)

**Consequences**:
- Native PostgreSQL compatibility
- Excellent time-series performance
- Built-in compression and retention policies
- SQL interface familiar to developers

### ADR-003: Message Queue vs Direct Communication

**Status**: Accepted

**Context**: Inter-service communication pattern selection.

**Decision**: Direct API calls with Redis pub/sub for real-time events

**Consequences**:
- Lower latency for critical paths
- Simpler debugging and monitoring
- Redis provides event broadcasting when needed
- Can add message queue later if needed

## Performance Architecture

### Latency-Critical Path

```mermaid
flowchart LR
    subgraph Hot Path ["< 10ms latency"]
        MD[Market Data] -->|WebSocket| CACHE[Redis Cache]
        CACHE --> AGENTS[Trading Agents]
        AGENTS --> RISK[Risk Check]
        RISK --> EXEC[Execution]
    end
    
    subgraph Async Path ["Async processing"]
        CACHE -.-> TS[(TimescaleDB)]
        CACHE -.-> ANALYTICS[Analytics]
        EXEC -.-> REPORTING[Reporting]
    end
    
    style Hot Path fill:#ff9999
    style Async Path fill:#99ccff
```

### Caching Strategy

```mermaid
flowchart TB
    subgraph L1["L1 Cache - In-Memory"]
        PROC[Process Cache<br/>~1ms latency]
    end
    
    subgraph L2["L2 Cache - Redis"]
        REDIS[Redis Cluster<br/>~5ms latency]
    end
    
    subgraph L3["L3 Storage - Database"]
        DB[(TimescaleDB<br/>~20ms latency)]
    end
    
    REQUEST[Data Request] --> PROC
    PROC -->|Miss| REDIS
    REDIS -->|Miss| DB
    
    DB --> REDIS
    REDIS --> PROC
    PROC --> RESPONSE[Response]
```

## Security Architecture

```mermaid
flowchart TB
    subgraph External["External Layer"]
        USER[Users]
        API[API Clients]
    end
    
    subgraph Edge["Edge Security"]
        WAF[Web Application Firewall]
        DDOS[DDoS Protection]
        LB[Load Balancer<br/>SSL Termination]
    end
    
    subgraph App["Application Layer"]
        AUTH[Auth Service<br/>JWT/OAuth2]
        GATEWAY[API Gateway<br/>Rate Limiting]
        RBAC[RBAC Engine]
    end
    
    subgraph Internal["Internal Services"]
        SERVICES[Trading Services]
        SECRETS[Secret Manager<br/>Vault]
        AUDIT[Audit Logger]
    end
    
    subgraph Data["Data Layer"]
        ENC[Encryption at Rest]
        DB[(Encrypted Database)]
    end
    
    USER --> WAF
    API --> WAF
    WAF --> DDOS
    DDOS --> LB
    LB --> AUTH
    AUTH --> GATEWAY
    GATEWAY --> RBAC
    RBAC --> SERVICES
    SERVICES --> SECRETS
    SERVICES --> AUDIT
    SERVICES --> ENC
    ENC --> DB
```

## Monitoring and Observability

```mermaid
flowchart LR
    subgraph Apps["Applications"]
        API[API Service]
        TRADING[Trading Engine]
        AGENTS[AI Agents]
        EXEC[Execution Service]
    end
    
    subgraph Telemetry["Telemetry Collection"]
        METRICS[Prometheus Metrics]
        LOGS[Structured Logs]
        TRACES[Distributed Traces]
        EVENTS[Business Events]
    end
    
    subgraph Storage["Storage & Processing"]
        PROM[(Prometheus)]
        LOKI[(Loki)]
        TEMPO[(Tempo)]
        KAFKA[Kafka]
    end
    
    subgraph Visualization["Visualization & Alerting"]
        GRAF[Grafana]
        ALERT[AlertManager]
        SLACK[Slack]
        PAGER[PagerDuty]
    end
    
    Apps --> METRICS
    Apps --> LOGS
    Apps --> TRACES
    Apps --> EVENTS
    
    METRICS --> PROM
    LOGS --> LOKI
    TRACES --> TEMPO
    EVENTS --> KAFKA
    
    PROM --> GRAF
    LOKI --> GRAF
    TEMPO --> GRAF
    KAFKA --> GRAF
    
    GRAF --> ALERT
    ALERT --> SLACK
    ALERT --> PAGER
```

---

This document provides a comprehensive view of the AlphaPulse architecture using various diagramming techniques. The diagrams cover:

1. **C4 Model**: System context, containers, and components
2. **Data Flows**: How data moves through the system
3. **Sequences**: Step-by-step interaction flows
4. **Deployment**: Infrastructure and deployment topology
5. **State Machines**: Order and system state transitions
6. **Entity Relationships**: Database schema overview
7. **Architecture Decisions**: Key design choices
8. **Performance**: Latency-critical paths and caching
9. **Security**: Security layers and controls
10. **Monitoring**: Observability architecture

Each diagram provides a different perspective on the system, helping developers, operators, and stakeholders understand how AlphaPulse works at various levels of abstraction.