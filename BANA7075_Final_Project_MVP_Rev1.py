# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 19:27:50 2025

@author: Frank
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 09:44:18 2025

@author: Frank
"""
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
import os


os.chdir(r"C:\Alpaca")

data = pd.read_csv("stock_data_18.csv")
data["date"] = pd.to_datetime(data["timestamp"]).dt.date

#Gen a datatime series from the dataset
date_min = data["date"].min()
date_max = data["date"].max()
date_diff = date_max - date_min

# Get the NYSE market calendar
nyse = mcal.get_calendar('NYSE')
# Get valid trading days between your range
market_days = nyse.schedule(start_date=date_min, end_date=date_max)
# Convert to a list of dates (datetime.date)
date_series = market_days.index.date

# Rank by trade_count within each date group
data["rank"] = data.groupby("date")["trade_count"].rank(method="first", ascending=False)

# Filter top 500 ranked per day
data_count = data[data["rank"] <= 500]

symbol_counts = data_count.groupby("symbol")["open"].count()
symbols_to_keep = symbol_counts[symbol_counts > 374].index
data_count = data_count[data_count["symbol"].isin(symbols_to_keep)]

# Make full grid of all symbols and all dates
full_index = pd.MultiIndex.from_product(
    [symbols_to_keep, date_series], names=["symbol", "date"]
)

data_count = data_count.set_index(["symbol", "date"])
data_count = data_count.reindex(full_index).reset_index()

data_count = data_count.sort_values(by=["symbol", "date"])
data_count["RET"] = data_count.groupby("symbol")["close"].pct_change() * 100
data_count = data_count.dropna()

window_mom = 15

# Apply rolling statistics without named aggregations
ret_stats = (
    data_count.groupby("symbol")["RET"]
    .rolling(window_mom)
    .agg(['mean', 'std', 'skew', 'kurt'])
    .shift(1)
    .reset_index(level=0, drop=True)
)

# Rename columns
ret_stats.columns = ['RET_mean', 'RET_std', 'RET_skew', 'RET_kurt']
data_count = pd.concat([data_count, ret_stats], axis=1)

# TBill
data_tbill = pd.read_excel("HistoricalPrices 3m TBILL.xlsx")

data_tbill = data_tbill.sort_values(by="Date")
data_tbill["RET_rf"] = data_tbill["Close"].pct_change() * 100
data_tbill["RET_rf"] = data_tbill["RET_rf"].shift(1)
data_tbill = data_tbill.set_index("Date")

# Assuming data_count['date'] is datetime.date
data_count = data_count.set_index("date")
data_count["RET_rf"] = data_count.index.map(data_tbill["RET_rf"])

data_count

# SPY Import
data_spy = pd.read_excel("HistoricalPrices_SPY.xlsx")
data_spy = data_spy.sort_values(by = "Date")
data_spy["RET_SPY"] = data_spy["Close"].pct_change() * 100
data_spy["RET_SPY"] = data_spy["RET_SPY"].shift(1)
data_spy = data_spy.set_index("Date")

data_count["RET_SPY"] = data_count.index.map(data_spy["RET_SPY"])

data_count = data_count.dropna()
data_count.to_csv("stock_data_18_rev2.csv")


#-----------------------------------------------Model Building----------------------------------------------------#
# Random Forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
import os
import mlflow.data
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

# Set MLflow tracking to your local SQLite server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("20240412_EXP1")

# Count the size of each group
group_sizes = data_count.groupby("symbol")["symbol"].transform("count")
group_ranks = data_count.groupby("symbol").cumcount()
is_first_half = group_ranks < (group_sizes / 2)

# Data splits
data_model = data_count[is_first_half]
data_backtesting = data_count[~is_first_half]

# Log input dataset
dataset_name = "stock_data_18_rev2"
dataset_source = "C:/Alpaca"  # raw string not needed, no backslashes
dataset = mlflow.data.from_pandas(
    data_model,
    source=dataset_source,
    name=dataset_name
)

# Feature list
features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf", "vwap", "RET_SPY"]
symbols = data_model["symbol"].unique()
model_results = []

# Begin tracking run
with mlflow.start_run(run_name="Model Training") as run:
    mlflow.log_input(dataset, context="training")

    for sym in symbols:
        try:
            df = data_model[data_model["symbol"] == sym].copy()
            df = df.sort_values("date")
            df["target"] = df["RET"]

            X = df[features]
            y = df["target"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            signature = infer_signature(X_test, y_pred)
            input_example = X_train.sample(n=1)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_param(f"{sym}_n_estimators", 100)
            mlflow.log_param(f"{sym}_random_state", 42)
            mlflow.log_metric(f"{sym}_mse", mse)
            mlflow.log_metric(f"{sym}_r2", r2)

            artifact_path = f"model_{sym}"
            mlflow.sklearn.log_model(
                rf,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example
            )

            model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

            # Try to register the model
            try:
                mlflow.register_model(model_uri=model_uri, name=f"Model_{sym}")
                print(f"Registered Model_{sym} | R2: {r2:.4f} | MSE: {mse:.4f}")
            except MlflowException as e:
                print(f"Could not register Model_{sym}: {str(e)}")

            model_results.append({
                "symbol": sym,
                "mse": mse,
                "r2": r2,
                "model": rf
            })

        except Exception as e:
            print(f"Failed for symbol {sym}: {str(e)}")
    
    
    
#------------------------------------------- Backtesting ------------------------------------------------------------------    
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("20240412_EXP1")
experiment_id = experiment.experiment_id
    
# List all runs (you can filter or sort if needed)
runs = client.search_runs(experiment_id, order_by=["metrics.r2 DESC"], max_results=5)

# Show top run info (for example)
for run in runs:
    print(f"Run ID: {run.info.run_id} | R2 Score: {run.data.metrics.get('r2')}")

run_id = runs[0].info.run_id 

# --- Configuration ---
features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf", "vwap", "RET_SPY"]
model_dir = "models/rf"

# --- Predict returns per symbol ---
all_preds = []

dataset = mlflow.data.from_pandas(
        data_backtesting,
        source=dataset_source,
        name=dataset_name
    )

# Start an MLflow run
with mlflow.start_run(run_name="Model Training") as run:
    mlflow.log_input(dataset, context="testing")
    
    # Retrieve the current active run ID
    active_run = mlflow.active_run()
    if active_run is None:
        raise Exception("No active MLflow run found. Ensure that a run is started before loading models.")
    
    for sym in data_backtesting["symbol"].unique():
        # Construct the model URI using the run ID and artifact path
        model_uri = f"runs:/{run_id}/model_{sym}"
    
        try:
            # Load the model from MLflow
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Model for symbol {sym} could not be loaded: {e}")
            continue
    
        df = data_backtesting[data_backtesting["symbol"] == sym].copy()
        df = df.sort_values("date")
    
        if df[features].isnull().any(axis=1).all():
            continue
    
        df["prediction"] = model.predict(df[features])
        all_preds.append(df[["symbol", "prediction", "RET", "RET_rf", "RET_SPY"]].assign(date=df.index))
    
    # --- Combine predictions ---
    pred_df = pd.concat(all_preds, ignore_index=True)
    
    # --- Rank and select top 25 ---
    pred_df = pred_df.sort_values(["date", "prediction"], ascending=[True, False])
    pred_df["rank"] = pred_df.groupby("date")["prediction"].rank(method="first", ascending=False)
    
    top_25 = pred_df[pred_df["rank"] <= 25]
    
    # --- Compute daily portfolio return ---
    daily_returns = (
        top_25.groupby("date")[["RET", "RET_rf", "RET_SPY"]]
        .mean()
        .rename(columns={"RET": "portfolio_return", "RET_rf": "risk_free_rate", "RET_SPY": "RET_SPY"})
    )
    
    # --- Compute Sharpe Ratio ---
    daily_returns["excess_return"] = daily_returns["portfolio_return"] - daily_returns["risk_free_rate"]
    excess_returns = daily_returns["excess_return"] / 100
    
    sharpe_daily = excess_returns.mean() / excess_returns.std()
    sharpe_annual = sharpe_daily * np.sqrt(252)
    
    # --- Compute cumulative return ---
    daily_returns["cumulative_return"] = (1 + daily_returns["portfolio_return"] / 100).cumprod()
    
    daily_returns.to_csv("backtest_top25_with_sharpe.csv")
    # --- Print & save results ---
    print(f"Daily Sharpe Ratio: {sharpe_daily:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe_annual:.4f}")
    
    # --- Compute Maximum Drawdown ---
    # Calculate the running maximum of the cumulative returns
    running_max = daily_returns["cumulative_return"].cummax()
    
    # Compute the drawdown at each point
    drawdown = (daily_returns["cumulative_return"] - running_max) / running_max
    
    # Determine the maximum drawdown
    max_drawdown = drawdown.min()
    
    # Convert to percentage
    max_drawdown_percent = max_drawdown * 100
    
    print(f"Maximum Drawdown: {max_drawdown_percent:.2f}%")
    
    # Compute SPY cumulative returns
    spy_cumulative_return = (1 + daily_returns["RET_SPY"] / 100).cumprod()
    
        
    mlflow.log_metric("sharpe_ratio_daily", sharpe_daily)
    mlflow.log_metric("sharpe_ratio_annual", sharpe_annual)
    mlflow.log_metric("max_drawdown_percent", max_drawdown_percent)


# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(daily_returns.index, daily_returns["cumulative_return"], label="Portfolio", linewidth=2)
plt.plot(daily_returns.index, spy_cumulative_return, label="SPY", linewidth=2, linestyle='--')
plt.title("Cumulative Returns: Portfolio vs SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot drawdown
plt.plot(daily_returns.index, drawdown, label="Drawdown", color="red")
plt.title("Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

