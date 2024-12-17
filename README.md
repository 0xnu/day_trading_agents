## Automated Day Trading Agents (ADTA)

ADTA leverages [CrewAI](https://www.crewai.com/) and OpenAI's [LLM Model](https://platform.openai.com/docs/models/) for market analysis, signal generation, and risk management—operating strictly during US market hours.

> [!WARNING]
> Disclaimer: ADTA provides simulated trading signals for educational purposes only. Real-time trading behaviour may differ. Not intended for actual investment decisions. Consult a qualified financial advisor.

### Features

* Real-time market data collection and validation (WIP; might evolve in the future)
* Technical analysis using RSI, MACD, and Bollinger Bands
* Risk management and portfolio protection
* Data processing with Polars DataFrames
* Market schedule awareness (9:30 AM - 4:00 PM ET, weekdays only)
* Multi-agent system using CrewAI framework

### Prerequisites

* Python 3.8+
* OpenAI API key
* Required Python packages:
    - crewai
    - langchain-openai
    - polars
    - pandas
    - numpy
    - tqdm
    - icecream
    - python-dotenv
    - pydantic
    - requests

### Installation

1. Clone the repository:

```bash
git clone git@github.com:0xnu/day_trading_agents.git
cd day_trading_agents
```

2. Install dependencies:

```bash
## Prerequisites
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python3 -m pip install --upgrade pip

## When you finish
deactivate
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env file with your OpenAI and Alpha Vantage API keys
```

### Project Structure

```bash
day_trading_agents/
├── scripts/
│   ├── day_trading_agents.py
│   └── day_trading_agents_advanced.py
├── trading_data/
│   ├── signals/
│   │   ├── csv/    # Polars DataFrame storage
│   │   └── json/   # JSON signal storage
│   └── results/    # Trading session results
├── requirements.txt
└── .env
```

### Debug Mode

Enable debug tracking:

```python
# In scripts/day_trading_agents.py
DEBUG = True  # Enables icecream logging
```

Debug features include:

* Configuration validation
* Agent initialisation tracking
* Signal processing progress
* File storage operations
* Exception details

### Market Hours

Disable and bypass market hours:

```python
# In scripts/day_trading_agents_advanced.py
enforce_market_hours=False
```

### Usage

Run the trading system:

```bash
python3 -m scripts.day_trading_agents ## Simple

python3 -m scripts.day_trading_agents_advanced ## Advanced
```

The system will:

1. Verify market hours (US Eastern Time)
2. Process each stock with progress tracking
3. Generate and store trading signals
4. Track all operations with debug logging when enabled

### Agents (Simple)

#### MarketDataAgent

Collects and validates real-time market data using GPT-4 Turbo, ensuring data quality and completeness.

#### TechnicalAnalysisAgent

Analyses market data using technical indicators (RSI, MACD, Bollinger Bands) to generate trading signals with confidence scores.

#### RiskManagementAgent

Evaluates trading signals, manages risk parameters, and provides portfolio protection strategies.

### Data Models

* `Config`: System configuration and parameters
* `StockInfo`: Basic stock information
* `MarketData`: Market data validation model
* `TechnicalSignal`: Technical analysis signals with confidence scores

### Agents (Advanced)

#### MarketDataAgent

Handles real-time market data collection and validation using [Alpha Vantage API](https://www.alphavantage.co/). The agent:

- Retrieves current and historical price data with rate limiting (5 calls/minute)
- Implements caching to minimise API calls (60-second expiry)
- Validates data quality and completeness
- Processes batch requests for multiple stock symbols
- Stores data in both CSV and JSON formats for analysis

#### TechnicalAnalysisAgent 

Performs technical analysis using multiple indicators:

- RSI (Relative Strength Index)
  - Calculates momentum with 14-period default
  - Generates buy signals below 30
  - Generates sell signals above 70

- MACD (Moving Average Convergence Divergence)
  - Uses 12/26/9 standard periods
  - Identifies trend reversals and momentum
  - Generates signals based on line crossovers

- Bollinger Bands
  - Implements 20-period moving average with 2 standard deviations
  - Generates buy signals at lower band crosses
  - Generates sell signals at upper band crosses

#### RiskManagementAgent

Evaluates trading signals and manages portfolio risk:

- Analyses confidence scores for each technical signal
- Monitors signal metadata and technical indicators
- Provides risk assessments based on market conditions
- Generates trade recommendations with risk parameters
- Stores assessment results for future analysis

#### Data Models

##### `Config`: Advanced configuration model with settings:

- API keys for OpenAI and Alpha Vantage (SecretStr for security)
- Model selection and parameters (temperature, max tokens)
- File paths for data and signal storage
- Market hours enforcement flags
- Stock symbol list with descriptions
- Debugging and logging settings

##### `StockInfo`: Detailed stock information model containing:

- Trading symbol
- Company name
- Market sector
- Market capitalisation
- Average trading volume

##### `MarketData`: Market data validation model with temporal features:

- Trading symbol
- Timestamp with timezone awareness
- OHLC (Open, High, Low, Close) prices
- Trading volume
- Data validation status

##### `TechnicalSignal`: Technical analysis signal model:

- Trading symbol and timestamp
- Indicator type (RSI, MACD, Bollinger Bands)
- Signal value and direction (Buy, Sell, Hold)
- Confidence score (0-1 range)
- Technical metadata dictionary containing:
  - RSI values
  - MACD line and signal line
  - Bollinger Band values (upper, middle, lower)
  - Additional indicator-specific data

##### `SignalStorage`: Storage handler for trading signals:
- Structured storage in CSV and JSON formats
- Timestamp-based file organisation
- Historical signal retrieval
- Data persistence and backup
- Signal aggregation and filtering

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### License

This project is licensed under the [BSD 3-Clause](LICENSE) License.

### Citation

```tex
@misc{afoadta2024,
  author       = {Oketunji, A.F.},
  title        = {Automated Day Trading Agents (ADTA)},
  year         = 2024,
  version      = {2.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14502412},
  url          = {https://doi.org/10.5281/zenodo.14502412}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.