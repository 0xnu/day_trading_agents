#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: day_trading_agents.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Saturday December 14 04:12:12 2024
@updated: Saturday December 14 11:25:00 2024
@desc: Automated Day Trading Agents (ADTA).
@run: python3 day_trading_agents.py
"""

from typing import List, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, SecretStr
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import polars as pl
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import logging
import calendar
from zoneinfo import ZoneInfo
from tqdm import tqdm
from icecream import ic

# Configure icecream
ic.configureOutput(prefix='DEBUG -> ')

# Debug mode flag (uncomment to enable debugging)
# DEBUG = True
DEBUG = False

# Set up logging with tqdm-friendly configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Config(BaseModel):
    """Configuration settings for the trading system."""
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    model_name: str = Field("gpt-4-turbo-preview", description="OpenAI model name")
    storage_path: Path = Field(default=Path("trading_data"), description="Path for data storage")
    signals_path: Path = Field(default=Path("trading_data/signals"), description="Path for signals")
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
            ic(self.dict(exclude={'openai_api_key'}))

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

    def save_signals(self, signals: List[TechnicalSignal], symbol: str, signal_type: str) -> None:
        """Save signals in both CSV and JSON formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{symbol}_{signal_type}_{timestamp}"
        
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
                'confidence': s.confidence
            }
            for s in signals
        ])
        csv_file = self.csv_path / f"{base_filename}.csv"
        signals_data.write_csv(csv_file)
        
        if DEBUG:
            ic(f"Files saved: {json_file}, {csv_file}")

class MarketSchedule:
    """Handle market schedule and trading day checks."""
    
    @staticmethod
    def is_trading_day() -> bool:
        """Check if today is a trading day."""
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

class MarketDataAgent(Agent):
    """Agent responsible for market data collection."""
    
    def __init__(self, openai_api_key: str):
        """Initialize Market Data Agent."""
        super().__init__(
            role='Market Data Collector',
            goal='Collect and validate real-time market data',
            backstory='''Expert in financial data collection with experience in 
            handling real-time market feeds and ensuring data quality.''',
            llm=ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4-turbo-preview"
            ),
            verbose=True
        )

class TechnicalAnalysisAgent(Agent):
    """Agent responsible for technical analysis."""
    
    def __init__(self, openai_api_key: str):
        """Initialize Technical Analysis Agent."""
        super().__init__(
            role='Technical Analyst',
            goal='Analyze market data and generate trading signals',
            backstory='''Expert technical analyst with deep knowledge of market 
            patterns and technical indicators.''',
            llm=ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4-turbo-preview"
            ),
            verbose=True
        )

class RiskManagementAgent(Agent):
    """Agent responsible for risk management."""
    
    def __init__(self, openai_api_key: str):
        """Initialize Risk Management Agent."""
        super().__init__(
            role='Risk Manager',
            goal='Monitor and manage trading risks',
            backstory='''Expert risk manager specialising in portfolio protection 
            and trade sizing.''',
            llm=ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-4-turbo-preview"
            ),
            verbose=True
        )

class DayTradingAgents:
    """Main trading system integrating all components."""
    
    def __init__(self, config: Config):
        """Initialize trading system."""
        self.config = config
        self.storage = SignalStorage(config.storage_path)
        
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
        
        self.market_data_agent = MarketDataAgent(
            self.config.openai_api_key.get_secret_value()
        )
        self.technical_analysis_agent = TechnicalAnalysisAgent(
            self.config.openai_api_key.get_secret_value()
        )
        self.risk_management_agent = RiskManagementAgent(
            self.config.openai_api_key.get_secret_value()
        )

    def create_tasks(self) -> List[Task]:
        """Create tasks for the trading crew."""
        return [
            Task(
                description='''Collect and validate real-time market data for the 
                configured stocks. Ensure data quality and completeness.''',
                expected_output='''JSON object containing:
                {
                    "data": {
                        "symbol": {
                            "timestamp": "YYYY-MM-DD HH:MM:SS",
                            "price": float,
                            "volume": int,
                            "quality_score": float
                        }
                    },
                    "validation": {
                        "completeness": bool,
                        "accuracy": bool,
                        "timeliness": bool
                    }
                }''',
                agent=self.market_data_agent
            ),
            Task(
                description='''Analyse market data using technical indicators and 
                generate trading signals with confidence scores.''',
                expected_output='''JSON object containing:
                {
                    "signals": [
                        {
                            "symbol": str,
                            "timestamp": "YYYY-MM-DD HH:MM:SS",
                            "indicator": str,
                            "signal": "BUY/SELL/HOLD",
                            "confidence": float,
                            "technical_data": {
                                "rsi": float,
                                "macd": float,
                                "bollinger_bands": {
                                    "upper": float,
                                    "middle": float,
                                    "lower": float
                                }
                            }
                        }
                    ]
                }''',
                agent=self.technical_analysis_agent
            ),
            Task(
                description='''Evaluate trading signals considering risk parameters 
                and current market conditions.''',
                expected_output='''JSON object containing:
                {
                    "risk_assessment": [
                        {
                            "symbol": str,
                            "signal_id": str,
                            "risk_score": float,
                            "position_size": float,
                            "stop_loss": float,
                            "take_profit": float,
                            "risk_factors": {
                                "market_volatility": float,
                                "position_exposure": float,
                                "correlation_risk": float
                            },
                            "recommendation": {
                                "action": "EXECUTE/MODIFY/REJECT",
                                "reasoning": str
                            }
                        }
                    ],
                    "portfolio_impact": {
                        "current_exposure": float,
                        "new_exposure": float,
                        "risk_metrics": {
                            "var": float,
                            "sharpe_ratio": float,
                            "max_drawdown": float
                        }
                    }
                }''',
                agent=self.risk_management_agent
            )
        ]

    def run_trading_session(self):
        """Run a complete trading session."""
        if not MarketSchedule.is_trading_day():
            next_trading_day = MarketSchedule.get_next_trading_day()
            logger.info(
                f"Markets are closed. Next trading day: "
                f"{next_trading_day.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )
            return
        
        if DEBUG:
            ic("Starting trading session")
        
        try:
            # Process stocks with progress bar
            for stock in tqdm(self.config.stocks, desc="Processing stocks"):
                if DEBUG:
                    ic(f"Processing stock: {stock}")
                
                # Execute crew tasks
                result = self.crew.kickoff()
                
                # Process and store results
                self._process_trading_results(result)
                
                if DEBUG:
                    ic(f"Completed processing for: {stock}")
            
            logger.info("\nTrading Session Complete")
            
        except Exception as e:
            logger.error(f"Error during trading session: {str(e)}")
            if DEBUG:
                ic("Exception details:", str(e))
            raise

    def _process_trading_results(self, results: Any):
        """Process and store trading results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results_file = self.config.storage_path / "results" / f"session_{timestamp}.json"
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
    
    # Initialize configuration
    config = Config(
        openai_api_key=SecretStr(os.getenv('OPENAI_API_KEY')),
        storage_path=Path("trading_data"),
        signals_path=Path("trading_data/signals")
    )
    
    # Create trading system
    trading_system = DayTradingAgents(config)
    
    # Run trading session
    trading_system.run_trading_session()

if __name__ == "__main__":
    main()
