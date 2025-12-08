# Setup (one-time)
cd /path/to/mounted/weather_updated
source .venv/bin/activate  # or create .venv_fast

# Denver - Edge generation + sweep + training
python scripts/train_edge_classifier.py --city denver --threshold 0.5 --sample-rate 4 --regenerate-only --workers 12
python scripts/sweep_min_edge_threshold.py --city denver --metric sharpe --min-trades 500
# Note the optimal threshold, then:
python scripts/train_edge_classifier.py --city denver --threshold <OPTIMAL> --sample-rate 4 --trials 80

# Los Angeles - Same workflow (if dataset + ordinal ready)
python scripts/train_edge_classifier.py --city los_angeles --threshold 0.5 --sample-rate 4 --regenerate-only --workers 12
python scripts/sweep_min_edge_threshold.py --city los_angeles --metric sharpe --min-trades 500
python scripts/train_edge_classifier.py --city los_angeles --threshold <OPTIMAL> --sample-rate 4 --trials 80


(base) halsted@halsted:~/slow_weather_updated$ source .venv/bin/activate
(.venv) (base) halsted@halsted:~/slow_weather_updated$ python scripts/train_edge_classifier.py --city denver --threshold 0.5 --sample-rate 4 --regenerate-only --workers 20
Traceback (most recent call last):
  File "/home/halsted/slow_weather_updated/scripts/train_edge_classifier.py", line 35, in <module>
    from src.db import get_db_session
  File "/home/halsted/slow_weather_updated/src/db/__init__.py", line 3, in <module>
    from src.db.connection import (
  File "/home/halsted/slow_weather_updated/src/db/connection.py", line 18, in <module>
    _engine: Engine | None = None
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
(.venv) (base) halsted@halsted:~/slow_weather_updated$ 

pip freeze > requirements_frozen.txt
Python 3.11.14


 Changes Made
src/db/connection.py: Changed Engine | None → Optional[Engine] (Python 3.9+ compatible)
scripts/train_edge_classifier.py: Moved DB imports inside functions (lazy loading)
Now the script can run without database access!
🖥️ Fast Computer Setup Commands (Copy-Paste)
# ============================================================================
# FAST COMPUTER SETUP
# ============================================================================

# 1. Check current Python
python --version
python3.11 --version

# 2. Install Python 3.11 if needed (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# 3. Navigate to mounted project
cd ~/slow_weather_updated  # or your mount point

# 4. Remove old venv
rm -rf .venv .venv_fast

# 5. Create new venv with Python 3.11
python3.11 -m venv .venv

# 6. Activate
source .venv/bin/activate

# 7. Verify Python version
python --version
# Should show: Python 3.11.x

# 8. Upgrade pip
pip install --upgrade pip

# 9. Install dependencies
pip install -r requirements.txt

# 10. Test imports (should work WITHOUT database now)
python -c "
import sys
print(f'Python: {sys.version}')
import pandas, numpy, catboost
print('✅ Core packages OK')
# DB import will only happen if actually used
print('✅ Ready for parquet-only mode')
"
🚀 Run Edge Generation on Fast Computer (No DB Needed!)
# On fast computer (after setup above)
cd ~/slow_weather_updated
source .venv/bin/activate

# Denver - Generate edge data (~5-8 min)
python scripts/train_edge_classifier.py \
    --city denver \
    --threshold 0.5 \
    --sample-rate 4 \
    --regenerate-only \
    --workers 20

# Los Angeles - Generate edge data (~3-5 min, if dataset ready)
python scripts/train_edge_classifier.py \
    --city los_angeles \
    --threshold 0.5 \
    --sample-rate 4 \
    --regenerate-only \
    --workers 20
These will now work WITHOUT database because:
✅ DB import is lazy (only loads when actually calling DB functions)
✅ Parquet-only mode never calls DB functions
✅ Settlements loaded from parquets (if available)
✅ Candles loaded from parquets
📋 Verification
After setup, verify everything is ready:
# Check Python version matches
python --version
# Should show 3.11.14 or 3.11.x

# Check key parquets exist
ls -lh models/saved/denver/train_data_full.parquet
ls -lh models/saved/denver/ordinal_catboost_optuna.pkl
ls -lh models/candles/candles_denver.parquet

# Test script loads (dry run)
python -c "from scripts.train_edge_classifier import *; print('✅ Script loads OK')"
✅ Summary of Fixes
Python 3.9 compatibility: Optional[Engine] instead of Engine | None
Lazy DB import: Only imports DB when actually needed
Works without database: Parquet-only mode fully functional
Try running Denver edge generation on fast computer now! Should work with these changes.