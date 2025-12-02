"""
Live trading module for edge-based Kalshi weather trading.

This module implements a clean architecture approach with:
- WebSocket integration for real-time market data
- Edge classifier ML filtering for high-confidence trades
- Maker→taker order conversion with volume-weighted timeouts
- Comprehensive decision and order logging
"""

__version__ = "1.0.0"
