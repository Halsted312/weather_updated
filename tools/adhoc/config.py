"""
User-Editable Configuration for Ad-Hoc Kalshi Weather Predictions

Edit this file before running predict_now.py to specify:
- Which city to analyze
- What date/time to predict
- Optional Kalshi market prices for edge calculation
"""

# ============================================================================
# PRIMARY SETTINGS (Edit these for each prediction)
# ============================================================================

# City to analyze
CITY = "chicago"  # chicago, austin, denver, los_angeles, miami, philadelphia

# Event date (the day you're betting on)
DATE = "2025-11-28"  # Format: YYYY-MM-DD, or "today" for automatic

# Prediction time (24-hour format)
TIME = "1000"  # Examples: "1000" = 10:00am, "1316" = 1:16pm, "1430" = 2:30pm, "2200" = 10:00pm
               # Can also use "HH:MM" format like "10:00" or "13:16"

# Data fetching strategy
FETCH_FRESH_DATA = False  # True: call Visual Crossing API for latest data
                          # False: use database only (faster, but may be stale)

# ============================================================================
# OPTIONAL: Kalshi Market Prices (for edge calculation)
# ============================================================================

# Enter current Kalshi market prices (in cents, 1-99) for each bracket
# FROM KALSHI SCREENSHOT (Nov 28, 10am Chicago):
MARKET_PRICES = {
    "[<30]": 1,      # 29° or below (market: <1%)
    "[30-31]": 32,   # 30° to 31° (market: 30%)
    "[32-33]": 67,   # 32° to 33° (market: 62% - HIGHLIGHTED)
    "[34-35]": 13,   # 34° to 35° (market: 7%)
    "[36-37]": 1,    # 36° to 37° (market: <1%)
    "[38+]": 1,      # 38° or above (market: <1%)
}

# ============================================================================
# TRADING THRESHOLDS (Advanced - adjust based on your risk tolerance)
# ============================================================================

# Minimum edge required to recommend a trade
MIN_EDGE_PCT = 5.0  # Default: 5% (conservative)
                     # Lower = more trades, higher = more selective

# Minimum probability to show a bracket in output
MIN_CONFIDENCE = 0.15  # Default: 15% (hide very unlikely brackets)

# Maximum uncertainty (settlement std) to trust prediction
MAX_UNCERTAINTY_DEGF = 5.0  # Default: 5°F
                             # Predictions with std > 5°F are flagged as high-uncertainty

# Maximum confidence interval span to trust prediction
MAX_CI_SPAN_DEGF = 7.0  # Default: 7°F (90% CI width)

# ============================================================================
# MODEL SETTINGS (Don't change unless you know what you're doing)
# ============================================================================

# Training snapshot hours (models only trained on these hours)
TRAIN_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]

# Model directory
MODEL_DIR = "models/saved"

# Model filename pattern
MODEL_FILE = "ordinal_catboost_optuna.pkl"
