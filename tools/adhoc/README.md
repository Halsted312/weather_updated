# Ad-Hoc Kalshi Weather Prediction Tool

Quick tool to get bracket probabilities for any city/date/time.

## Quick Start

1. **Edit [config.py](config.py):**
   ```python
   CITY = "chicago"     # Your city
   DATE = "2025-11-28"  # Event date
   TIME = "1316"        # 1:16pm in 24-hour format
   ```

2. **Run prediction:**
   ```bash
   .venv/bin/python tools/adhoc/predict_now.py
   ```

3. **Get bracket probabilities** → Compare to Kalshi → Trade if edge exists!

## Config Options

### TIME Format
- `"1000"` = 10:00am
- `"1316"` = 1:16pm
- `"2145"` = 9:45pm
- Can also use `"13:16"` format

### MARKET_PRICES (Optional)
Add Kalshi prices to get BUY/SELL signals:
```python
MARKET_PRICES = {
    "[67-68]": 38,  # Bracket trading at 38¢
    "[69-70]": 56,  # Bracket trading at 56¢
}
```

Tool will calculate edge and recommend trades when edge > 5%

## Output Explained

### Delta = Settlement - Current Max
- `delta=0`: Settlement equals current observed max
- `delta=+3`: Settlement 3°F higher than current max
- `delta=-1`: Settlement 1°F lower (rare)

### Bracket Probabilities
- `P(settle >= 69)`: Probability settlement is 69°F or higher
- Use this to compare against Kalshi YES price

### Signals
- **🟢 BUY**: Model prob > Market price + 5% (edge)
- **🔴 SELL**: Market price > Model prob + 5% (edge)
- **HOLD**: Edge too small (< 5%)

## Accuracy by Time of Day

| Time | Typical Accuracy | Confidence |
|------|------------------|------------|
| 10am | 45-50% | Low - too early |
| 12pm | 50-55% | Moderate |
| 2pm  | 55-60% | Good |
| 4pm  | 60-65% | Very Good |
| 6pm+ | 65-75% | Excellent |

**Best edge**: Trade in afternoon/evening when model has more observations

## Safety Guardrails

Tool will warn you if:
- ⛔ **Insufficient data** (< 12 observations)
- ⚠️  **High uncertainty** (settlement std > 5°F)
- ⚠️  **Wide CI** (90% interval > 7°F span)
- ⚠️  **Early prediction** (before 2pm, accuracy < 55%)

## Current Limitation: Test Data Only

Right now the tool uses **historical test data** to demonstrate it works.

**To use with LIVE data:**
1. Ensure observations are being ingested continuously, OR
2. Run manual ingestion before prediction:
   ```bash
   .venv/bin/python scripts/ingest_vc_forecast_snapshot.py --city-code CHI
   ```
3. Contact the other agent to modify `predict_now.py` to use `load_inference_data()` for live queries

## Available Cities

- `chicago` (KMDW) - Delta range [-2, +10], 12 classifiers
- `austin` (KAUS) - Delta range [-1, +10], 11 classifiers ⭐ Best accuracy (68%)
- `denver` (KDEN) - Delta range [-1, +10], 11 classifiers
- `los_angeles` (KLAX) - Delta range [-1, +10], 11 classifiers
- `miami` (KMIA) - Delta range [-1, +10], 11 classifiers ⭐ Best MAE (0.46°F)
- `philadelphia` (KPHL) - Delta range [-2, +10], 12 classifiers

## Professional Features

✅ Snaps to nearest training hour (no risky interpolation)
✅ Quality guardrails (data completeness checks)
✅ Uncertainty warnings (high variance, wide CI)
✅ Edge calculation (compare to market prices)
✅ Uses same feature logic as training (no leakage)

## Example Usage

```bash
# Predict Chicago at 1:16pm
# Edit config.py:
#   CITY = "chicago"
#   DATE = "today"
#   TIME = "1316"
#   MARKET_PRICES = {"[69-70]": 56}

.venv/bin/python tools/adhoc/predict_now.py

# Output will show:
# - Delta probabilities
# - Bracket probabilities
# - BUY/SELL signals if edge exists
# - Warnings if uncertainty is high
```

## Next Steps

1. **Start using with test data** to understand the output format
2. **Set up live data ingestion** (coordinate with modeling agent)
3. **Add your Kalshi market prices** to get trade recommendations
4. **Monitor performance** - track predictions vs actual settlements

---

Built with professional quant recommendations: snap to training hours, quality/uncertainty guardrails, calibration awareness, edge-based signals.
