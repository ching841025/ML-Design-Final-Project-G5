#-----------------------------------------------Model Building----------------------------------------------------#
# Random Forrest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
from datetime import date
data_count = pd.read_parquet("clean_data.parquet")

traning_start = date(2019, 1, 1)
traning_end = date(2019, 3, 31)

test_start = date(2025, 1, 1)
test_end = date(2025, 3, 31)

# Now filter using index.get_level_values and compare with datetime.date
data_model = data_count[(data_count.index.get_level_values("date") >= traning_start) &
                        (data_count.index.get_level_values("date") <= traning_end)]

data_backtesting = data_count[(data_count.index.get_level_values("date") >= test_start) &
                              (data_count.index.get_level_values("date") <= test_end)]
# Define minimum required data points
min_required_points = 10  # Adjust based on your needs

# Count data points per symbol in data_model
symbol_counts = data_model.groupby('symbol').size()

# Get symbols with enough data points
valid_symbols = symbol_counts[symbol_counts >= min_required_points].index.tolist()

# Filter data_model to only include these symbols
data_model = data_model[data_model['symbol'].isin(valid_symbols)].copy()
data_backtesting = data_backtesting[data_backtesting['symbol'].isin(valid_symbols)].copy()

features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf","vwap",'RET_SPY']

model_dir = "models/rf"
os.makedirs(model_dir, exist_ok=True)

symbols = data_model["symbol"].unique()
model_results = []

for sym in symbols:
    df = data_model[data_model["symbol"] == sym].copy()
    df = df.sort_values("date")
    
    # Create target: next-day return
    df["target"] = df["RET"]

    X = df[features]
    y = df["target"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Save the model
    model_path = os.path.join(model_dir, f"model_{sym}.joblib")
    joblib.dump(rf, model_path)

    # Save results
    results = {
        "symbol": sym,
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "model": rf,
        "model_path": model_path
    }
    model_results.append(results)
    
    # Optional: print a summary
    print(f"{sym} | R2: {results['r2']:.4f} | MSE: {results['mse']:.4f}")
    
# Backtesting    
# --- Configuration ---
features = ["RET_mean", "RET_std", "RET_skew", "RET_kurt", "RET_rf", "vwap", "RET_SPY"]
model_dir = "models/rf"

# --- Predict returns per symbol ---
all_preds = []

for sym in data_backtesting["symbol"].unique():
    model_path = os.path.join(model_dir, f"model_{sym}.joblib")
    if not os.path.exists(model_path):
        continue
    
    model = joblib.load(model_path)
    df = data_backtesting[data_backtesting["symbol"] == sym].copy()
    df = df.sort_values("date")

    if df[features].isnull().any(axis=1).all():
        continue

    df["prediction"] = model.predict(df[features])
    all_preds.append(df[["symbol", "prediction", "RET", "RET_rf","RET_SPY"]].assign(date=df.index))

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

import mlflow
import mlflow.sklearn
# Set experiment name
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("stock selection")
# --- Prepare metrics ---
metrics = {
    "sharpe_daily": sharpe_daily,
    "sharpe_annual": sharpe_annual,
    "max_drawdown_percent": max_drawdown_percent,
}

run_name = "RandomF"

# --- Start MLflow run context ---
with mlflow.start_run(run_name=run_name) as run:

    # (Optional) Log metadata or config used in backtest
    mlflow.log_params({
        "top_n": 25,
        "model_type": "RandomForest",
        "feature_set": "base_v1"
    })

    # Log backtest performance metrics
    mlflow.log_metrics(metrics)

    # Save and log results table
    result_csv = "backtest_top25_with_sharpe.csv"
    daily_returns.to_csv(result_csv)
    mlflow.log_artifact(result_csv)

import matplotlib.pyplot as plt

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
