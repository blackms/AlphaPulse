# AlphaPulse Agent Instructions

## Core Guidelines

### File Operations
- Check large file size: `grep -c '\n' filename > line_count.txt`
- Break files >500 lines into chunks
- Pipe command output: `command > output.txt`
- Use read_file for content, search_files for patterns

### Module Changes
- Search for dependencies before changes:
  ```
  <search_files>
  <path>src</path>
  <regex>function_name</regex>
  </search_files>
  ```
- Update all affected modules
- Update SPEC.md for significant changes

### External Tools
- Use ask_perplexity for research
- Use chat_perplexity for discussions
- Maintain local context

## Module Operations

### Agents (`src/alpha_pulse/agents/`)
- **Data Safety:**
  ```python
  # Check file size before processing
  grep -c '\n' market_data.csv > line_count.txt
  count = int(read_file('line_count.txt'))
  if count > 500:
      # Process in chunks
  ```
- **Code Changes:**
  ```python
  # Search for dependencies
  <search_files>
  <path>src/alpha_pulse/agents</path>
  <regex>function_name</regex>
  </search_files>
  ```

### Risk Management (`src/alpha_pulse/risk_management/`)
- **Config:**
  ```python
  # Load and validate
  config = yaml.safe_load('config/risk_config.yaml')
  ```
- **Calculations:**
  ```python
  # Use tolerance
  TOLERANCE = 1e-10
  if abs(a - b) > TOLERANCE:
      # Handle difference
  ```

### Portfolio (`src/alpha_pulse/portfolio/`)
- **Caching:**
  ```python
  @lru_cache(maxsize=100)
  def create_price_matrix(data_str: str)
  ```
- **Data Handling:**
  - Forward-fill limit: 5 periods
  - Use Decimal for money
  - Validate weights sum to 1.0

### Execution (`src/alpha_pulse/execution/`)
- **Safe Trading:**
  ```python
  # Always use timeout and retry
  result = await self._retry_with_timeout(
      exchange.execute_trade,
      max_retries=2,
      timeout=20.0
  )
  ```
- **Error Prevention:**
  - Implement rollbacks
  - Track order states
  - Monitor rate limits

## Critical Parameters

### Risk Limits
- Max position size: 20% of portfolio
- Max leverage: 1.5x
- Stop loss: 2% per trade
- Max drawdown: 25%
- Min technical score: 0.15

### Data Handling
- Cache size: 100 entries
- Forward-fill: 5 periods max
- Retry attempts: 2
- Timeout: 20 seconds
- Floating tolerance: 1e-10

### Required Validations
- Config parameters
- Exchange responses
- Balance checks
- Weight sums (1.0)
- Rate limits