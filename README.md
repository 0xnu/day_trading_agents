## Automated Day Trading Agents (ADTA)

ADTA leverages [CrewAI](https://www.crewai.com/) and OpenAI's [GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4) for market analysis, signal generation, and risk management—operating strictly during US market hours.

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
    - tqdm
    - icecream
    - python-dotenv
    - pydantic

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
# Edit .env file with your OpenAI API key
```

### Configuration

The system uses a `Config` class that supports:

* OpenAI API settings (GPT-4 Turbo model)
* Storage paths for signals and results
* Default stock list:
    - AAPL (Apple Inc.)
    - MSFT (Microsoft Corporation)
    - GOOGL (Alphabet Inc.)
    - AMZN (Amazon.com Inc.)
    - NVDA (NVIDIA Corporation)
* Model parameters (temperature, max tokens)

### Project Structure

```bash
day_trading_agents/
├── trading_data/
│ ├── signals/
│ │ ├── csv/    # Polars DataFrame storage
│ │ └── json/   # JSON signal storage
│ └── results/  # Trading session results
├── day_trading_agents.py
├── requirements.txt
└── .env
```

### Debug Mode

Enable debug tracking:

```python
# In day_trading_agents.py
DEBUG = True  # Enables icecream logging
```

Debug features include:

* Configuration validation
* Agent initialisation tracking
* Signal processing progress
* File storage operations
* Exception details

### Usage

Run the trading system:

```bash
python3 -m day_trading_agents
```

The system will:

1. Verify market hours (US Eastern Time)
2. Process each stock with progress tracking
3. Generate and store trading signals
4. Track all operations with debug logging when enabled

### Agents

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
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14502412},
  url          = {https://doi.org/10.5281/zenodo.14502412}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
