#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: day_trading_agents_advanced.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Monday December 16 18:10:02 2024
@updated: Tuesday December 17 22:04:00 2024
@desc: Automated Day Trading Agents (ADTA) - Advanced.
@run: python3 day_trading_agents_advanced.py
"""

# Standard library imports
from typing import List, Any, Optional, Dict, Tuple
from datetime import datetime, timedelta
import calendar
import json
import logging
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import requests
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from icecream import ic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from tqdm import tqdm

# Configure icecream
ic.configureOutput(prefix='DEBUG -> ')

# Debug mode flag
DEBUG = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
required_env_vars = ['OPENAI_API_KEY', 'ALPHA_VANTAGE_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class Config(BaseModel):
    """Configuration settings for the trading system."""
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    model_name: str = Field("gpt-4o-2024-08-06", description="OpenAI model name")
    storage_path: Path = Field(default=Path("trading_data"), description="Path for data storage")
    signals_path: Path = Field(default=Path("trading_data/signals"), description="Path for signals")
    alpha_vantage_key: SecretStr = Field(..., description="Alpha Vantage API key")
    enforce_market_hours: bool = Field(
        default=True,
        description="Whether to enforce market hours checking"
    )
    stocks: List[str] = Field(
        default=[
            "AAPL",  # Apple Inc.
            "MSFT",  # Microsoft Corporation
            "GOOGL", # Alphabet Inc.
            "AMZN",  # Amazon.com Inc.
            "NVDA"   # NVIDIA Corporation
        ],
        description="List of stock symbols to analyze"
    )
    temperature: float = Field(default=0.7, description="LLM temperature setting")
    max_tokens: int = Field(default=2000, description="Maximum tokens for LLM response")

    def __init__(self, **data):
        super().__init__(**data)
        if DEBUG:
            ic(self.dict(exclude={'openai_api_key', 'alpha_vantage_key'}))

class StockInfo(BaseModel):
    """Model for stock information."""
    symbol: str
    company_name: str
    sector: str
    market_cap: float
    average_volume: int

class MarketData(BaseModel):
    """Model for market data validation."""
    symbol: str = Field(..., description="Trading symbol")
    date: datetime = Field(..., description="Date of the trading data")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")

class TechnicalSignal(BaseModel):
    """Model for technical analysis signals."""
    symbol: str
    date: datetime
    indicator: str
    value: float
    signal: str = Field(..., description="Buy, Sell, or Hold")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence score")
    technical_data: Dict[str, float] = Field(default_factory=dict)

class TechnicalIndicators:
    """Technical analysis indicators calculator."""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, periods: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Array of prices
            periods: RSI period (default 14)
            
        Returns:
            Array of RSI values
        """
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initialize
        avg_gain[periods] = np.mean(gain[:periods])
        avg_loss[periods] = np.mean(loss[:periods])
        
        # Calculate smoothed averages
        for i in range(periods + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (periods-1) + gain[i-1]) / periods
            avg_loss[i] = (avg_loss[i-1] * (periods-1) + loss[i-1]) / periods
        
        rs = avg_gain / np.where(avg_loss == 0, 1e-9, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def calculate_macd(prices: np.ndarray, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Array of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line)
        """
        # Calculate EMAs
        ema_fast = pd.Series(prices).ewm(span=fast_period, adjust=False).mean()
        ema_slow = pd.Series(prices).ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        return np.array(macd_line), np.array(signal_line)

    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, 
                                period: int = 20,
                                num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of prices
            period: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = pd.Series(prices).rolling(window=period).mean()
        std_dev = pd.Series(prices).rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return (
            np.array(upper_band),
            np.array(middle_band),
            np.array(lower_band)
        )

class SignalStorage:
    """Class to handle signal storage operations."""
    
    def __init__(self, base_path: Path):
        """Initialize signal storage."""
        self.signals_path = base_path / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.signals_path / "csv"
        self.json_path = self.signals_path / "json"
        self.csv_path.mkdir(parents=True, exist_ok=True)
        self.json_path.mkdir(parents=True, exist_ok=True)
        
        if DEBUG:
            ic(str(self.signals_path), str(self.csv_path), str(self.json_path))

    def save_signals(self, signals: List[TechnicalSignal], symbol: str) -> None:
        """Save technical analysis signals in both CSV and JSON formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{symbol}_signals_{timestamp}"
        
        if DEBUG:
            ic(f"Saving signals for {symbol}")
            ic(f"Signal count: {len(signals)}")
        
        # Save as JSON with progress bar
        json_file = self.json_path / f"{base_filename}.json"
        signals_dict = [signal.dict() for signal in tqdm(signals, desc="Processing signals")]
        with open(json_file, 'w') as f:
            json.dump(signals_dict, f, indent=4, default=str)
        
        # Convert to Polars DataFrame and save as CSV
        signals_data = pl.DataFrame([
            {
                'symbol': s.symbol,
                'date': s.date,
                'indicator': s.indicator,
                'value': s.value,
                'signal': s.signal,
                'confidence': s.confidence,
                'rsi': s.technical_data.get('rsi'),
                'macd': s.technical_data.get('macd'),
                'macd_signal': s.technical_data.get('macd_signal'),
                'bb_upper': s.technical_data.get('bb_upper'),
                'bb_middle': s.technical_data.get('bb_middle'),
                'bb_lower': s.technical_data.get('bb_lower')
            }
            for s in signals
        ])
        csv_file = self.csv_path / f"{base_filename}.csv"
        signals_data.write_csv(csv_file)
        
        if DEBUG:
            ic(f"Files saved: {json_file}, {csv_file}")

    def load_recent_signals(self, symbol: str, days: int = 7) -> List[TechnicalSignal]:
        """Load recent signals for a given symbol."""
        cutoff_date = datetime.now() - timedelta(days=days)
        signals = []
        
        # Load from JSON files
        json_files = list(self.json_path.glob(f"{symbol}_signals_*.json"))
        
        for file_path in json_files:
            file_date = datetime.strptime(
                file_path.stem.split('_')[-1],
                "%Y%m%d_%H%M%S"
            )
            
            if file_date >= cutoff_date:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    signals.extend([TechnicalSignal(**signal) for signal in data])
        
        return signals

class StockPriceFetcher:
    """Handle real-time stock price fetching."""
    
    def __init__(self, api_key: str):
        """Initialize stock price fetcher."""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = 60  # Cache expiry in seconds

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Fetch current stock price from Alpha Vantage API.
        Returns dict with 'price' and 'volume' if successful, None if failed.
        """
        # Check cache first
        if symbol in self.cache:
            cached_data = self.cache[symbol]
            if time.time() - cached_data['timestamp'] < self.cache_expiry:
                return {
                    'price': cached_data['price'],
                    'volume': cached_data['volume']
                }

        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Global Quote' in data:
                quote = data['Global Quote']
                price_data = {
                    'price': float(quote['05. price']),
                    'volume': float(quote['06. volume'])
                }
                
                # Update cache
                self.cache[symbol] = {
                    'price': price_data['price'],
                    'volume': price_data['volume'],
                    'timestamp': time.time()
                }
                
                return price_data
            
            logger.warning(f"No price data found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None

    def get_batch_prices(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetch prices for multiple symbols with rate limiting.
        Returns dict of symbol -> price data.
        """
        results = {}
        
        for symbol in tqdm(symbols, desc="Fetching stock prices"):
            results[symbol] = self.get_current_price(symbol)
            time.sleep(12)  # Rate limit: 5 calls per minute
            
        return results

    def get_historical_prices(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch historical daily prices for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical price data or None if failed
        """
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Time Series (Daily)' in data:
                prices_dict = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(prices_dict, orient='index')
                
                # Convert string columns to numeric
                df = df.apply(pd.to_numeric)
                
                # Rename columns
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Sort by date and limit to requested days
                df = df.sort_index().tail(days)
                
                return df
            
            logger.warning(f"No historical data found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

class MarketSchedule:
    """Handle market schedule and trading day checks."""
    
    @staticmethod
    def is_trading_day(enforce_market_hours: bool = True) -> bool:
        """Check if today is a trading day."""
        if not enforce_market_hours:
            return True
            
        current_time = datetime.now(ZoneInfo("America/New_York"))
        
        # Check if it's a weekend
        if current_time.weekday() in [calendar.SATURDAY, calendar.SUNDAY]:
            logger.warning("Markets are closed (Weekend)")
            return False
            
        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if not (market_open <= current_time <= market_close):
            logger.warning("Outside market hours")
            return False
            
        return True

    @staticmethod
    def get_next_trading_day() -> datetime:
        """Get the next trading day."""
        current_time = datetime.now(ZoneInfo("America/New_York"))
        next_day = current_time + timedelta(days=1)
        
        while next_day.weekday() in [calendar.SATURDAY, calendar.SUNDAY]:
            next_day += timedelta(days=1)
            
        return next_day.replace(hour=9, minute=30, second=0, microsecond=0)

class TechnicalAnalysis:
    """Handle technical analysis calculations and signal generation."""
    
    def __init__(self):
        """Initialize technical analysis."""
        self.indicators = TechnicalIndicators()

    def analyze_price_data(self, 
                         symbol: str,
                         prices: np.ndarray,
                         volumes: np.ndarray) -> List[TechnicalSignal]:
        """
        Perform technical analysis on price data.
        
        Args:
            symbol: Stock symbol
            prices: Array of prices
            volumes: Array of trading volumes
            
        Returns:
            List of technical signals
        """
        signals = []
        
        # Calculate technical indicators
        rsi = self.indicators.calculate_rsi(prices)
        macd_line, signal_line = self.indicators.calculate_macd(prices)
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(prices)
        
        # Generate signals based on indicator values
        current_price = prices[-1]
        current_rsi = rsi[-1]
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        
        # Create technical data dictionary
        technical_data = {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'bb_upper': bb_upper[-1],
            'bb_middle': bb_middle[-1],
            'bb_lower': bb_lower[-1]
        }
        
        # RSI signals
        if current_rsi < 30:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='RSI',
                value=current_rsi,
                signal='BUY',
                confidence=0.8,
                technical_data=technical_data
            ))
        elif current_rsi > 70:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='RSI',
                value=current_rsi,
                signal='SELL',
                confidence=0.8,
                technical_data=technical_data
            ))
        
        # MACD signals
        if current_macd > current_signal:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='MACD',
                value=current_macd,
                signal='BUY',
                confidence=0.7,
                technical_data=technical_data
            ))
        elif current_macd < current_signal:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='MACD',
                value=current_macd,
                signal='SELL',
                confidence=0.7,
                technical_data=technical_data
            ))
        
        # Bollinger Bands signals
        if current_price < bb_lower[-1]:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='BB',
                value=current_price,
                signal='BUY',
                confidence=0.75,
                technical_data=technical_data
            ))
        elif current_price > bb_upper[-1]:
            signals.append(TechnicalSignal(
                symbol=symbol,
                date=datetime.now(),
                indicator='BB',
                value=current_price,
                signal='SELL',
                confidence=0.75,
                technical_data=technical_data
            ))
        
        return signals

class MarketDataAgent(Agent):
    """Agent responsible for market data collection."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize Market Data Agent."""
        super().__init__(
            role='Market Data Collector',
            goal='Collect and validate real-time market data',
            backstory='''Expert in financial data collection with experience in 
            handling real-time market feeds and ensuring data quality.''',
            llm=ChatOpenAI(
                api_key=api_key,
                model=model
            ),
            verbose=True
        )

class TechnicalAnalysisAgent(Agent):
    """Agent responsible for technical analysis."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize Technical Analysis Agent."""
        super().__init__(
            role='Technical Analyst',
            goal='Analyze market data and generate trading signals',
            backstory='''Expert technical analyst with deep knowledge of market 
            patterns and technical indicators.''',
            llm=ChatOpenAI(
                api_key=api_key,
                model=model
            ),
            verbose=True
        )

class RiskManagementAgent(Agent):
    """Agent responsible for risk management."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize Risk Management Agent."""
        super().__init__(
            role='Risk Manager',
            goal='Monitor and manage trading risks',
            backstory='''Expert risk manager specialising in portfolio protection 
            and trade sizing.''',
            llm=ChatOpenAI(
                api_key=api_key,
                model=model
            ),
            verbose=True
        )

class DayTradingAgents:
    """Main trading system integrating all components."""
    
    def __init__(self, config: Config):
        """Initialize trading system."""
        self.config = config
        self.storage = SignalStorage(config.storage_path)
        self.price_fetcher = StockPriceFetcher(
            config.alpha_vantage_key.get_secret_value()
        )
        self.technical_analysis = TechnicalAnalysis()
        
        if DEBUG:
            ic("Initialising DayTradingAgents")
            ic(f"Number of stocks to analyse: {len(config.stocks)}")
        
        # Initialize agents
        self._init_agents()
        
        # Create agent crew
        self.crew = Crew(
            agents=[
                self.market_data_agent,
                self.technical_analysis_agent,
                self.risk_management_agent
            ],
            tasks=self.create_tasks(),
            verbose=True
        )

    def _init_agents(self):
        """Initialize trading agents."""
        if DEBUG:
            ic("Creating trading agents")
        
        common_agent_params = {
            'api_key': self.config.openai_api_key.get_secret_value(),
            'model': self.config.model_name
        }
        
        self.market_data_agent = MarketDataAgent(**common_agent_params)
        self.technical_analysis_agent = TechnicalAnalysisAgent(**common_agent_params)
        self.risk_management_agent = RiskManagementAgent(**common_agent_params)

    def create_tasks(self) -> List[Task]:
        """Create tasks for the trading crew."""
        return [
            Task(
                description='''Collect and validate real-time market data for the 
                configured stocks. Ensure data quality and completeness.''',
                expected_output='''JSON object containing market data and validation metrics''',
                agent=self.market_data_agent
            ),
            Task(
                description='''Analyse market data using technical indicators and 
                generate trading signals with confidence scores.''',
                expected_output='''JSON object containing technical analysis signals and metadata''',
                agent=self.technical_analysis_agent
            ),
            Task(
                description='''Evaluate trading signals considering risk parameters 
                and current market conditions.''',
                expected_output='''JSON object containing risk assessment and recommendations''',
                agent=self.risk_management_agent
            )
        ]

    def run_trading_session(self):
        """Run a complete trading session."""
        if not MarketSchedule.is_trading_day(self.config.enforce_market_hours):
            if self.config.enforce_market_hours:
                next_trading_day = MarketSchedule.get_next_trading_day()
                logger.info(
                    f"Markets are closed. Next trading day: "
                    f"{next_trading_day.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                return
        
        if DEBUG:
            ic("Starting trading session")
        
        try:
            # Fetch current market data
            market_data = self.price_fetcher.get_batch_prices(self.config.stocks)
            if DEBUG:
                ic("Fetched market data:", market_data)
            
            # Process stocks with progress bar
            for symbol in tqdm(self.config.stocks, desc="Processing stocks"):
                if DEBUG:
                    ic(f"Processing stock: {symbol}")
                
                # Get historical data for technical analysis
                historical_data = self.price_fetcher.get_historical_prices(symbol)
                if historical_data is None:
                    logger.warning(f"Skipping {symbol} due to missing historical data")
                    continue
                
                # Perform technical analysis
                signals = self.technical_analysis.analyze_price_data(
                    symbol=symbol,
                    prices=historical_data['close'].values,
                    volumes=historical_data['volume'].values
                )
                
                # Save signals
                self.storage.save_signals(signals, symbol)
                
                # Execute crew tasks
                result = self.crew.kickoff()
                
                # Process and store results
                self._process_trading_results(result, symbol)
                
                if DEBUG:
                    ic(f"Completed processing for: {symbol}")
            
            logger.info("\nTrading Session Complete")
            
        except Exception as e:
            logger.error(f"Error during trading session: {str(e)}")
            if DEBUG:
                ic("Exception details:", str(e))
            raise

    def _process_trading_results(self, results: Any, symbol: str):
        """Process and store trading results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = (self.config.storage_path / "results" / 
                       f"{symbol}_session_{timestamp}.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        if DEBUG:
            ic(f"Results saved to: {results_file}")
        else:
            logger.info(f"Results saved to: {results_file}")

def main():
    """Main function to run the trading system."""
    if DEBUG:
        ic("Starting main function")
    
    try:
        # Initialize configuration with environment variables
        config = Config(
            openai_api_key=SecretStr(os.getenv('OPENAI_API_KEY')),
            alpha_vantage_key=SecretStr(os.getenv('ALPHA_VANTAGE_KEY')),
            storage_path=Path("trading_data"),
            signals_path=Path("trading_data/signals"),
            enforce_market_hours=True  # Can be set to False to bypass market hours check
        )
        
        # Create trading system
        trading_system = DayTradingAgents(config)
        
        # Run trading session
        trading_system.run_trading_session()
        
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {str(e)}")
        raise

if __name__ == "__main__":
    main()
