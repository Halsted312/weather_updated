### Overview of Kalshi weather trading and prediction guide

1. WebSocket Recorder Status and Data Logging

The repository includes a Kalshi WebSocket recorder (scripts/kalshi_ws_recorder.py) designed to run continuously as a daemon, either via systemd or Docker
GitHub
GitHub
. This recorder subscribes to Kalshi’s live market data channels (order book updates, ticker quotes, trades, etc.) for all six weather series (including Los Angeles and Denver)
GitHub
. All incoming messages are saved to the Postgres database in the kalshi.ws_raw table (a TimescaleDB hypertable)
GitHub
GitHub
. Each record captures a UTC timestamp, the data source (“kalshi”), message type (e.g. ticker, orderbook_delta), the market/topic, and the full JSON payload
GitHub
.

 

Frequency & duration: The recorder streams real-time order book snapshots and updates continuously. It batches incoming messages (flushing to DB every 100 messages) to avoid excessive commits
GitHub
GitHub
. In practice, this can mean multiple inserts per second during active trading hours, ensuring granular intra-minute data capture. The ws_raw hypertable is partitioned by day to accommodate long-term continuous logging
GitHub
. As long as the kalshi-ws-recorder service is running, it will automatically reconnect on any disconnect and keep logging indefinitely
GitHub
GitHub
. In summary, the system does support continuous snapshot logging of the order book in real time. No code modifications are needed for basic operation – you should simply ensure the service is running (e.g. via systemctl start kalshi-ws-recorder or docker compose up -d daemons) so that fresh data populates the kalshi.ws_raw table. The table’s design and the recorder’s auto-restart behavior indicate it is intended for 24/7 operation
GitHub
. (If for any reason this service isn’t active, you might see “No WebSocket data recorded yet” in the freshness check
GitHub
GitHub
, in which case starting the recorder will initiate live data logging.)

 

Data structures: In addition to the raw WebSocket log, the database also maintains a 1-minute candlestick table kalshi.candles_1m for each market
GitHub
. Each candlestick row includes the minute timestamp, last trade or ask prices (open/high/low/close), the closing YES bid/ask quotes, and volume/open interest
GitHub
GitHub
. Historically, ~3 years of backfilled 1-min data (1.3M candles) exist
GitHub
, and the recorder can complement this by providing finer detail between those minute bars. Currently, the candlestick table is updated via explicit API polling/backfill scripts, not in real-time (there isn’t yet a live-updating candle daemon in this branch). The WebSocket recorder’s ws_raw table is therefore the primary source of live order book snapshots. In a live trading scenario, you would use either the latest ws_raw entries or query Kalshi’s API for current quotes. In summary, yes – the WebSocket recorder is intended to be running continuously, logging real-time order book data into kalshi.ws_raw. The data is stored at the raw message level (not yet aggregated into a separate “order_book_snapshots” table), but you have the necessary data to reconstruct best bids/asks or derive custom snapshots as needed.

2. CLI Script for Model vs Market Analysis (LA & Denver)

We propose a manual CLI script (e.g. scripts/compare_model_vs_market.py analog for LA/Denver) to evaluate opportunities in Los Angeles and Denver markets. When executed, this script will perform the following steps (mirroring the approach used in the existing Chicago test script
GitHub
):

Load Latest Market State: Fetch the most up-to-date YES/NO bid-ask prices for each temperature bracket of the LA and Denver daily markets. This can be done either by querying the database or by using the Kalshi API:

From DB: If the kalshi.candles_1m table is being updated close to real-time, select the last candlestick for each market on today’s date in LA and Denver. Use the yes_bid_c and yes_ask_c (closing bid/ask of that minute) as the latest quotes
GitHub
. Ensure the timestamp is the current or last minute to get “latest” data.

Alternatively via API: Use the Kalshi REST client to pull current order books. For each active market (six brackets per city), call client.get_orderbook(ticker) to retrieve the best bid and ask prices
GitHub
GitHub
. The script can parse the returned JSON to extract the best YES bid and best YES ask for each contract (the code already handles both list and nested formats)
GitHub
GitHub
. This gives the market’s implied probabilities (YES price in cents ≈ % chance) for each bracket. Using the API ensures we have the exact live order book state, but it requires API credentials; using the DB candlestick snapshot is credential-free but may be a minute behind.

Load Model Forecast & Predictions: Load the latest model prediction for daily high temperature in each city:

Likely you have a trained model for LA and Denver (e.g. saved CatBoost or logistic models in models/saved/los_angeles/... and models/saved/denver/...). The script will load these model objects (similar to how the Chicago model is loaded from models/saved/chicago.../*.pkl
GitHub
).

Query the weather database for the current day’s observations and forecast features. This means retrieving today’s Visual Crossing observations up to now (e.g. from wx.minute_obs or the new wx.vc_minute_weather), as well as the forecast snapshot data (e.g. yesterday’s forecast for today, and/or today’s latest forecast). The Chicago script uses a helper to get an inference data snapshot from the DB
GitHub
. We’d do the same for LA and Denver, obtaining:

t_base: the observed max temperature so far today (used as a baseline)
GitHub
GitHub
.

Daily forecast info (expected high, etc.) and possibly hourly forecast details to build features.

Construct the feature vector for the model (e.g. combining current temps, forecast, and other predictors) using the same feature engineering as training. Then run the model inference to get predicted probabilities for each possible “delta” or each bracket. In the Chicago example, they predict a distribution over Δ (difference between final high and current temp) and convert that to a settlement temperature probability distribution
GitHub
. We will do analogous computation, yielding the model’s estimated probability for each bracket (the script settle_probs_to_bracket_probs likely handles mapping temperature outcomes to specific bracket contracts
GitHub
). We also compute the model’s expected settlement temperature and uncertainty (standard deviation)
GitHub
, which we’ll use for comparisons.

Combine Market and Model Data: For each temperature bracket contract, gather:

Forecasted high temperature: We can use the official daily forecast (e.g. Visual Crossing’s predicted high for today in that city) as a point estimate. This could also be the model’s expected value for today’s high (which might incorporate more info). We will report this as the forecast temp.

Market-implied temperature: This is derived from the market’s pricing across all brackets. Essentially, using the market’s probabilities for each bracket, compute the probability-weighted expected high. For example, sum(probability of each bracket × a representative temperature for that bracket). A simple approach is to take each bracket’s mid-point (or cap/floor) as an estimate of the outcome and compute the weighted average. This yields a single number (in °F) that the market is “implying” for the day’s high. We will list this as well.

Model prediction: The model’s probability for that specific bracket being the winner (i.e. settling YES). We can express this as a percentage. For clarity, we might list the model’s predicted probability or an implied temperature range as well.

Edge & Opportunity: Calculate the edge in percentage points: how much the model’s probability deviates from the market price. For example, if the model thinks a bracket has a 30% chance but the market’s YES price is 20¢ (20%), there is a +10 point edge on the buy side. If the model is lower than market price (e.g. model 50% vs market 70%), then the edge is -20 points (an overpricing from model’s perspective). We compute edge = model_prob (%) – market_prob (%)
GitHub
.

Sharpe ratio / P&L: Using the edge, we estimate the risk-reward of a trade. One approach is to treat each contract as a binary payoff and compute the expected P&L and standard deviation if we take a position at the market price given the model’s true probability. For instance, if model p=0.3 and price=0.2 ($0.20), the expected profit for buying one YES is 0.3*$1 + 0.7*$0 – $0.20 = $0.10 (10¢) and the outcome standard deviation can be computed from the variance of {+0.80, -0.20} outcomes. From these, an implied Sharpe (expected return / std dev) can be calculated. We’ll output a summary stat to indicate the quality of the trade. (E.g., a 10¢ edge on a 20¢ contract is very high expected return, likely a high Sharpe). We may simplify and report a qualitative Sharpe or an approximate one given typical weather forecast error distribution. The key is to convey how attractive the opportunity is in risk-adjusted terms.

High-Confidence Filters (10°F Rule): To focus on the most promising opportunities, the script will apply the 10°F threshold rule as a filter or flag. In practice, this rule means we look for situations where the forecast vs market expectation differs by at least 10°F – a sizable gap:

For example, if our forecast expects a high of 85°F but the market-implied high is only 75°F (a 10° difference), that’s a big discrepancy. Likewise, if a particular bracket’s cutoff is, say, 80°F and our model predicts with high confidence a high well above 90°F (or well below 70°F), that bracket outcome is very likely (or very unlikely) to occur. A 10° gap is far outside normal forecast error, indicating a strong edge.

The script will identify any bracket where the forecasted high is ≥10°F higher or lower than the bracket’s range bounds and the market isn’t already fully pricing that in. These are “high-confidence” opportunities. In such cases, the model’s predicted probability will be near 0% or 100% while the market price might be moderate – this yields a very large edge (and likely high Sharpe).

We will highlight these situations (e.g. mark them with an asterisk or output a separate alert) to signal that the user should strongly consider action there. Essentially, the 10°F rule ensures we act only when there’s a substantial margin of safety between our forecast and the market’s implied outlook, reducing the chance of false signals.

Output Summary per Market: The script will print a compact table for each city’s market (Los Angeles and Denver). For clarity, we separate the output by city:

City Header: e.g. “LOS ANGELES – Today’s High Temp Market vs Model” (with the date). We can include the current time or cutoff used, and perhaps the observed temperature so far, model expected high, etc., as context
GitHub
.

Bracket rows: For each of the six brackets (e.g. “<50°F”, “50–51°F”, ... “≥100°F” – actual ranges depend on city/season), output columns such as:

Bracket (ticker or temperature range) – e.g. the contract name or a shorthand of the range.

Forecast (°F) – the forecasted high temp or relevant reference (this might be the same for all brackets in a city, essentially the city’s forecast high; we might instead list the bracket range midpoint or threshold for clarity).

Market-Implied (°F) – the market’s implied high for that bracket or overall (could also be omitted per bracket and just given as one number above; alternatively, we list the market’s probability for this bracket in %).

Model Pred (%) – model’s probability that this bracket will settle YES (e.g. “12%”).

Edge (% points) – difference between model and market probability. Positive means model thinks it’s more likely than market does (potential buy YES opportunity), negative means model thinks it’s less likely (sell/short YES, i.e. buy NO).

Sharpe/P&L – an estimate of the trade quality. For example, “Sharpe ~1.8, EV=+$0.05” could denote a good trade, whereas a smaller edge might be “Sharpe ~0.5”. We will likely simplify to just a Sharpe-like number or an “EV per contract” in cents.

Action – a one-word recommendation: “Buy”, “Sell”, or “Wait”.

Buy indicates the model expects the probability is much higher than the market price – you would buy YES (or equivalently, buy the contract) to profit if the event occurs.

Sell means the market is overpriced relative to model – you should sell that contract (or buy NO) because the event likely won’t happen as often as the price implies.

Wait means no strong edge either way (model and market roughly agree, or the discrepancy is below our confidence threshold). This helps the trader prioritize only the clear mispricings.

The script may highlight the single best bracket in each city (highest positive edge) with a marker (for example, the Chicago script adds “← Best” next to the top opportunity
GitHub
). It will also note if no bracket meets the 10°F/edge threshold, in which case the recommendation is to stand aside for that city.

In summary, this CLI tool will combine current market data, weather forecasts, and model predictions to produce a concise decision aid. It echoes the process used in the internal test script for Chicago
GitHub
, generalized to Los Angeles and Denver. By manually running this script (e.g. python scripts/live_edge_analysis.py --city la denver), you can quickly see where your model and the market disagree significantly and get a sense of the potential Sharpe ratio of those trades. This keeps you in the loop without fully automated trading – it’s a focused, human-in-the-loop dashboard for daily strategy decisions.

3. Organizing Per-City Prediction Outputs

Currently, the repo’s models/saved/{city} directories hold saved model artifacts (like trained model pickles and perhaps training metadata) per city
GitHub
. It’s best to keep model artifacts separate from live predictions/outputs. I recommend creating a new dedicated folder (and module) for predictions – for example: src/predictions/. Inside this, use subfolders or files organized by city. For instance:

src/predictions/
    los_angeles/
        latest_forecast.json
        latest_model_probs.json
        features_snapshot.csv
        edges.csv
    denver/
        ... (similar structure)


Each city’s folder can store the latest or historical prediction results:

Raw Forecast Data: e.g. a JSON or CSV with the daily forecast info (forecasted high, etc.) for that city.

Model Outputs: probabilities for each bracket or temperature outcome, expected value, and uncertainty.

Market Features: any features derived from the market (implied probabilities, current prices).

Inferred Edges/Signals: the computed edges and suggested action for each bracket.

Storing these in a structured way makes it easy to retrieve and analyze later. For example, you might append each day’s predictions to a CSV (with a timestamp) for performance tracking. Keeping them under src/predictions/ cleanly separates predicted data from the static model definitions in models/saved/.

 

Alternatively, one could consider extending the existing models/saved/{city} hierarchy to include predictions, but this is less ideal. The models/saved directory is more for static assets (like model files, which don’t change frequently). Inserting dynamic prediction outputs there could clutter the model files and mix concerns. A new predictions/ module emphasizes that these are runtime results. It also aligns with the project structure by function: you’d have src/weather/ for data sources, src/kalshi/ for API, src/db/ for DB models, src/models/ for training, and now src/predictions/ for live prediction outputs. Each city’s subfolder can mirror the city keys used elsewhere (e.g. “los_angeles”, “denver” matching CITIES config keys), ensuring consistency.

 

Within each city’s prediction folder, consider storing both the latest snapshot and a history. For example, maintain a rolling Parquet or CSV log of each prediction run (timestamped), which can be used for backtesting how your signals would have performed. If the volume is low (a few snapshots per day), even a simple CSV per city would suffice. If you prefer database storage (see next section), the folder could instead contain code or config, with the actual data in the DB. But having easily accessible flat files (like Parquet) per city can be convenient for quick analysis in pandas or Jupyter notebooks.

 

In summary, a new src/predictions/ module with subdirectories for each city is a clean solution. It keeps prediction outputs version-controlled and organized, without interfering with core model files. You’d treat it as a place where each day’s (or each run’s) predictions and signals are written, which can then feed into reports or be evaluated later.

4. Logging 1-Minute Prediction Snapshots for Research

For deeper analysis and eventual paper-trading simulations, it’s wise to record the model’s predictions at a regular frequency (e.g. each minute) during the trading day. There are a couple of approaches to store these high-frequency prediction snapshots:

Database Table (TimescaleDB): Create a new table (perhaps in the sim schema or a new pred schema) to log each prediction snapshot. For example, a table sim.prediction_snapshot with columns: timestamp, city, model_expected_temp, model_std, market_implied_temp, each bracket’s model probability, each bracket’s market price, and recommended action. Each minute (or whatever interval you choose), the live script can insert a row for LA and for Denver. Using TimescaleDB for this has benefits: it can handle time-series data efficiently, you can easily query and aggregate the data with SQL, and join with other tables (like actual outcomes or historical forecasts). It centralizes the data for your research analysis and for building a track record of the strategy. Since the volume is low (1440 snapshots per day per city), storage is not a concern. We could also add an index on (city, timestamp) and even compress older data if needed.

Parquet Files: Alternatively, the script could append each snapshot to a Parquet file (or a series of files partitioned by date or city). Parquet is efficient for analytics – you could later load the entire history into pandas or Spark and evaluate performance. For instance, you might keep one Parquet file per city per day (or month) containing the minute-by-minute prediction data. This approach is file-based and doesn’t require a database, which can be simpler in some setups. You’d need to ensure the script appends or writes in a safe manner (perhaps writing to a temp file then merging, or using pandas to append – though appending to Parquet isn’t trivial without reading, so more likely one file per day written at end-of-day).

In-Memory with Periodic Dumps: If you want minimal overhead during runtime, the script could hold recent predictions in memory and only periodically flush to disk/DB. But given the low frequency, it’s probably fine to write each minute directly.

Recommendation: Using the Postgres/Timescale database table is a robust option. It leverages existing infrastructure (no new external files to manage) and ensures consistency (ACID writes each minute). You can easily query “what was my model prediction at 10:03 AM on Jan 5 for LA?” and compare to outcome, etc. It also fits with how other data (prices, forecasts) are stored – for example, you might join prediction snapshots with the wx.settlement table to see if high edges indeed corresponded to profitable outcomes. If taking this route, define a clear schema: e.g. sim.live_predictions(city TEXT, ts_utc TIMESTAMPTZ, expected_temp DOUBLE, temp_std DOUBLE, market_implied_temp DOUBLE, yes_bid1 SMALLINT, yes_ask1 SMALLINT, … yes_bid6, yes_ask6, model_prob1,… model_prob6, recommended_action TEXT). This is just an example; you might store probabilities or edges as JSON if that’s easier (using a JSONB column for all bracket data). The Timescale hypertable can be partitioned by day for efficiency.

 

The Parquet approach is also feasible and may be preferred if you want to do a lot of off-line ML experiments on the predictions (since reading a Parquet into pandas is straightforward). Parquet would let you keep the data outside the database, which some prefer for analysis with notebooks. In that case, a directory structure by city and date is helpful: e.g. data/predictions/los_angeles/2025-12-08_predictions.parquet containing all snapshots for that day. Over time you accumulate a folder of daily files. This is similar to how one might store market data for research.

 

It’s worth noting you can even do both: log to the database for reliability, and have a nightly job export the day’s predictions to a Parquet for easier sharing or analysis. But to start, choose one to keep things simple.

 

For future paper-trading: having these snapshots means you can later simulate “If I traded whenever the script said ‘Buy’ with >X edge, what would my P&L be?” – effectively a retroactive analysis using real model outputs and real market prices. This is extremely valuable for refining the strategy. With the data logged, you could compute realized P&L by comparing the trade entry price (market price at that minute) to the settlement (either 0 or 100 cents) and summing over all such signals.

 

In conclusion, set up a regular logging of predictions every minute during market hours (or as frequently as you deem necessary). I suggest using a new database table for live prediction snapshots for ease of integration
GitHub
GitHub
, and optionally exporting to Parquet for machine learning experiments. This will create a rich dataset for research, allowing you to measure strategy Sharpe, drawdowns, and other metrics using real-time model insights versus eventual outcomes. It’s a low-effort, high-reward addition: essentially treating your model+market insight at time t as another time-series to analyze, similar to how you treat price data. By keeping these logs, you’ll be well-prepared to do a realistic paper-trading analysis and smoothly transition to live trading when ready (since you’ll have seen how the signals perform historically).