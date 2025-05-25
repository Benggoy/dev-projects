#!/usr/bin/env python3
"""
MCP Server Template
===================

A comprehensive template for creating Model Context Protocol (MCP) servers
with trading and financial analysis capabilities.

Features:
- Stock data retrieval
- Technical analysis functions
- Portfolio management
- Risk calculations
- Market data streaming

Author: Benggoy
License: MIT
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPResponse:
    """Standard MCP response format"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp or datetime.now().isoformat()
        }

class MCPTool(ABC):
    """Abstract base class for MCP tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> MCPResponse:
        pass

class StockDataTool(MCPTool):
    """Tool for retrieving stock market data"""
    
    @property
    def name(self) -> str:
        return "get_stock_data"
    
    @property
    def description(self) -> str:
        return "Retrieve historical stock data and current market information"
    
    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, TSLA)"
                },
                "period": {
                    "type": "string",
                    "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                    "default": "1y"
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
                    "default": "1d"
                },
                "include_info": {
                    "type": "boolean",
                    "description": "Include company information",
                    "default": False
                }
            },
            "required": ["symbol"]
        }
    
    async def execute(self, symbol: str, period: str = "1y", interval: str = "1d", include_info: bool = False) -> MCPResponse:
        """Execute stock data retrieval"""
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Get historical data
            hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                return MCPResponse(
                    success=False,
                    error=f"No data found for symbol {symbol}"
                )
            
            # Prepare response data
            response_data = {
                "symbol": symbol.upper(),
                "period": period,
                "interval": interval,
                "data_points": len(hist_data),
                "latest_price": float(hist_data['Close'].iloc[-1]),
                "price_change": float(hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else 0,
                "price_change_pct": float((hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-2] - 1) * 100) if len(hist_data) > 1 else 0,
                "volume": int(hist_data['Volume'].iloc[-1]) if 'Volume' in hist_data.columns else None,
                "historical_data": hist_data.reset_index().to_dict('records')
            }
            
            # Include company info if requested
            if include_info:
                try:
                    info = ticker.info
                    response_data["company_info"] = {
                        "name": info.get("longName", "N/A"),
                        "sector": info.get("sector", "N/A"),
                        "industry": info.get("industry", "N/A"),
                        "market_cap": info.get("marketCap", "N/A"),
                        "employees": info.get("fullTimeEmployees", "N/A"),
                        "description": info.get("longBusinessSummary", "N/A")[:500] + "..." if len(info.get("longBusinessSummary", "")) > 500 else info.get("longBusinessSummary", "N/A")
                    }
                except:
                    response_data["company_info"] = "Unable to fetch company information"
            
            return MCPResponse(success=True, data=response_data)
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return MCPResponse(
                success=False,
                error=f"Failed to fetch data: {str(e)}"
            )

class TechnicalAnalysisTool(MCPTool):
    """Tool for technical analysis calculations"""
    
    @property
    def name(self) -> str:
        return "technical_analysis"
    
    @property
    def description(self) -> str:
        return "Calculate technical analysis indicators (RSI, SMA, EMA, MACD, Bollinger Bands)"
    
    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators to calculate (rsi, sma, ema, macd, bollinger)",
                    "default": ["rsi"]
                },
                "period": {
                    "type": "string",
                    "description": "Time period for data",
                    "default": "6mo"
                },
                "rsi_period": {
                    "type": "integer",
                    "description": "RSI calculation period",
                    "default": 14
                },
                "sma_period": {
                    "type": "integer",
                    "description": "Simple Moving Average period",
                    "default": 20
                },
                "ema_period": {
                    "type": "integer",
                    "description": "Exponential Moving Average period",
                    "default": 12
                }
            },
            "required": ["symbol"]
        }
    
    async def execute(self, symbol: str, indicators: List[str] = None, period: str = "6mo", 
                     rsi_period: int = 14, sma_period: int = 20, ema_period: int = 12) -> MCPResponse:
        """Execute technical analysis"""
        try:
            if indicators is None:
                indicators = ["rsi"]
            
            # Get stock data
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period=period)
            
            if data.empty:
                return MCPResponse(
                    success=False,
                    error=f"No data found for symbol {symbol}"
                )
            
            results = {
                "symbol": symbol.upper(),
                "period": period,
                "indicators": {}
            }
            
            # Calculate RSI
            if "rsi" in indicators:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(span=rsi_period).mean()
                avg_loss = loss.ewm(span=rsi_period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                results["indicators"]["rsi"] = {
                    "current": float(rsi.iloc[-1]),
                    "period": rsi_period,
                    "signal": "overbought" if rsi.iloc[-1] > 70 else "oversold" if rsi.iloc[-1] < 30 else "neutral",
                    "historical": rsi.dropna().tail(30).to_dict()
                }
            
            # Calculate Simple Moving Average
            if "sma" in indicators:
                sma = data['Close'].rolling(window=sma_period).mean()
                results["indicators"]["sma"] = {
                    "current": float(sma.iloc[-1]),
                    "period": sma_period,
                    "signal": "bullish" if data['Close'].iloc[-1] > sma.iloc[-1] else "bearish",
                    "historical": sma.dropna().tail(30).to_dict()
                }
            
            # Calculate Exponential Moving Average
            if "ema" in indicators:
                ema = data['Close'].ewm(span=ema_period).mean()
                results["indicators"]["ema"] = {
                    "current": float(ema.iloc[-1]),
                    "period": ema_period,
                    "signal": "bullish" if data['Close'].iloc[-1] > ema.iloc[-1] else "bearish",
                    "historical": ema.dropna().tail(30).to_dict()
                }
            
            # Calculate MACD
            if "macd" in indicators:
                ema_12 = data['Close'].ewm(span=12).mean()
                ema_26 = data['Close'].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                histogram = macd_line - signal_line
                
                results["indicators"]["macd"] = {
                    "macd": float(macd_line.iloc[-1]),
                    "signal": float(signal_line.iloc[-1]),
                    "histogram": float(histogram.iloc[-1]),
                    "trend": "bullish" if histogram.iloc[-1] > 0 else "bearish"
                }
            
            # Calculate Bollinger Bands
            if "bollinger" in indicators:
                sma_20 = data['Close'].rolling(window=20).mean()
                std_20 = data['Close'].rolling(window=20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)
                
                current_price = data['Close'].iloc[-1]
                results["indicators"]["bollinger"] = {
                    "upper_band": float(upper_band.iloc[-1]),
                    "middle_band": float(sma_20.iloc[-1]),
                    "lower_band": float(lower_band.iloc[-1]),
                    "current_price": float(current_price),
                    "position": "overbought" if current_price > upper_band.iloc[-1] else "oversold" if current_price < lower_band.iloc[-1] else "normal"
                }
            
            return MCPResponse(success=True, data=results)
            
        except Exception as e:
            logger.error(f"Error calculating technical analysis for {symbol}: {str(e)}")
            return MCPResponse(
                success=False,
                error=f"Technical analysis failed: {str(e)}"
            )

class PortfolioAnalysisTool(MCPTool):
    """Tool for portfolio analysis and optimization"""
    
    @property
    def name(self) -> str:
        return "portfolio_analysis"
    
    @property
    def description(self) -> str:
        return "Analyze portfolio performance, risk metrics, and optimization suggestions"
    
    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "quantity": {"type": "number"},
                            "avg_cost": {"type": "number"}
                        }
                    },
                    "description": "List of portfolio positions"
                },
                "period": {
                    "type": "string",
                    "description": "Analysis period",
                    "default": "1y"
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark symbol for comparison",
                    "default": "SPY"
                }
            },
            "required": ["positions"]
        }
    
    async def execute(self, positions: List[Dict], period: str = "1y", benchmark: str = "SPY") -> MCPResponse:
        """Execute portfolio analysis"""
        try:
            if not positions:
                return MCPResponse(
                    success=False,
                    error="No positions provided"
                )
            
            # Get current prices and calculate portfolio value
            portfolio_data = []
            total_value = 0
            
            for position in positions:
                symbol = position['symbol'].upper()
                quantity = position['quantity']
                avg_cost = position['avg_cost']
                
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period="1d")['Close'].iloc[-1]
                
                position_value = quantity * current_price
                total_value += position_value
                
                portfolio_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "current_price": float(current_price),
                    "position_value": float(position_value),
                    "unrealized_pnl": float((current_price - avg_cost) * quantity),
                    "unrealized_pnl_pct": float((current_price / avg_cost - 1) * 100)
                })
            
            # Calculate portfolio weights
            for position in portfolio_data:
                position["weight"] = position["position_value"] / total_value
            
            # Get benchmark data for comparison
            benchmark_ticker = yf.Ticker(benchmark)
            benchmark_data = benchmark_ticker.history(period=period)
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            # Calculate portfolio metrics
            total_unrealized_pnl = sum(pos["unrealized_pnl"] for pos in portfolio_data)
            total_cost_basis = sum(pos["avg_cost"] * pos["quantity"] for pos in portfolio_data)
            total_return_pct = (total_unrealized_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0
            
            # Risk metrics (simplified)
            symbols = [pos["symbol"] for pos in portfolio_data]
            weights = [pos["weight"] for pos in portfolio_data]
            
            # Get historical data for risk calculation
            hist_data = yf.download(symbols, period=period)['Adj Close']
            if len(symbols) == 1:
                hist_data = hist_data.to_frame()
                hist_data.columns = symbols
            
            returns = hist_data.pct_change().dropna()
            portfolio_returns = (returns * weights).sum(axis=1)
            
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (portfolio_returns.mean() * 252 - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate
            max_drawdown = ((portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()) / portfolio_returns.cumsum().cummax()).max()
            
            results = {
                "portfolio_summary": {
                    "total_value": float(total_value),
                    "total_cost_basis": float(total_cost_basis),
                    "total_unrealized_pnl": float(total_unrealized_pnl),
                    "total_return_pct": float(total_return_pct),
                    "number_of_positions": len(positions)
                },
                "positions": portfolio_data,
                "risk_metrics": {
                    "volatility": float(portfolio_volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown),
                    "beta": 1.0,  # Simplified - would need proper calculation vs benchmark
                },
                "diversification": {
                    "largest_position_weight": max(pos["weight"] for pos in portfolio_data),
                    "number_of_assets": len(portfolio_data),
                    "concentration_risk": "high" if max(pos["weight"] for pos in portfolio_data) > 0.3 else "moderate" if max(pos["weight"] for pos in portfolio_data) > 0.2 else "low"
                }
            }
            
            return MCPResponse(success=True, data=results)
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return MCPResponse(
                success=False,
                error=f"Portfolio analysis failed: {str(e)}"
            )

class MCPServer:
    """Main MCP Server class"""
    
    def __init__(self):
        self.tools = {
            "get_stock_data": StockDataTool(),
            "technical_analysis": TechnicalAnalysisTool(),
            "portfolio_analysis": PortfolioAnalysisTool()
        }
        logger.info(f"MCP Server initialized with {len(self.tools)} tools")
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle incoming MCP requests"""
        try:
            tool_name = request.get("tool")
            parameters = request.get("parameters", {})
            
            if tool_name not in self.tools:
                return MCPResponse(
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                ).to_dict()
            
            tool = self.tools[tool_name]
            response = await tool.execute(**parameters)
            return response.to_dict()
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            return MCPResponse(
                success=False,
                error=f"Request processing failed: {str(e)}"
            ).to_dict()
    
    def get_tool_definitions(self) -> Dict:
        """Get tool definitions for MCP protocol"""
        tools = {}
        for name, tool in self.tools.items():
            tools[name] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        return {"tools": tools}
    
    async def start_server(self, host: str = "localhost", port: int = 8080):
        """Start the MCP server (placeholder for actual server implementation)"""
        logger.info(f"MCP Server starting on {host}:{port}")
        
        # In a real implementation, this would start an HTTP/WebSocket server
        # For now, this is a template showing the structure
        
        print(f"""
🚀 MCP Server Template Ready!

Available Tools:
{'-' * 40}
""")
        
        for name, tool in self.tools.items():
            print(f"• {tool.name}: {tool.description}")
        
        print(f"""
{'-' * 40}

To implement this server:
1. Add HTTP/WebSocket server framework (FastAPI, aiohttp, etc.)
2. Implement proper MCP protocol handling
3. Add authentication and rate limiting
4. Deploy to your preferred hosting platform

Example usage:
{'-' * 20}
server = MCPServer()
response = await server.handle_request({{
    "tool": "get_stock_data",
    "parameters": {{"symbol": "AAPL", "period": "1mo"}}
}})
print(response)
""")

async def main():
    """Example usage and testing"""
    server = MCPServer()
    
    # Test stock data tool
    print("🧪 Testing Stock Data Tool...")
    stock_request = {
        "tool": "get_stock_data",
        "parameters": {
            "symbol": "AAPL",
            "period": "1mo",
            "include_info": True
        }
    }
    stock_response = await server.handle_request(stock_request)
    print(f"Stock Data Response: {stock_response['success']}")
    
    # Test technical analysis tool
    print("\n🧪 Testing Technical Analysis Tool...")
    ta_request = {
        "tool": "technical_analysis",
        "parameters": {
            "symbol": "TSLA",
            "indicators": ["rsi", "sma", "macd"],
            "period": "3mo"
        }
    }
    ta_response = await server.handle_request(ta_request)
    print(f"Technical Analysis Response: {ta_response['success']}")
    
    # Test portfolio analysis tool
    print("\n🧪 Testing Portfolio Analysis Tool...")
    portfolio_request = {
        "tool": "portfolio_analysis",
        "parameters": {
            "positions": [
                {"symbol": "AAPL", "quantity": 10, "avg_cost": 150.0},
                {"symbol": "TSLA", "quantity": 5, "avg_cost": 200.0},
                {"symbol": "MSFT", "quantity": 8, "avg_cost": 300.0}
            ],
            "period": "6mo"
        }
    }
    portfolio_response = await server.handle_request(portfolio_request)
    print(f"Portfolio Analysis Response: {portfolio_response['success']}")
    
    # Show tool definitions
    print("\n📋 Tool Definitions:")
    definitions = server.get_tool_definitions()
    print(json.dumps(definitions, indent=2))
    
    # Start server (demonstration)
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())
