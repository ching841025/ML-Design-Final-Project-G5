import datetime
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
import os
import git 
import sys

# define repo and destination folder

repo_url = 'https://github.com/ching841025/ML-Design-Final-Project-G5.git'
repo_path = './ML_Design_Project'
file_dir = os.path.join(repo_path, 'file')

current_path = os.getcwd()
if ("ML_Design_Project" in current_path) == True:
    print("ML_Design_Project found on local machine.")
    print(f"Repo already exists at {repo_path}. Skipping clone.")
    file_paths = [os.path.join(current_path, f"stock_data_120_part_{i}.csv") for i in range(1, 7)
                 ]
    
else:
    print("ML_Design_Project found on local machine.") 
    print("Creating project folder.")
    os.makedirs(file_dir, exist_ok=True)
    
    print(f"Repo not found or empty at {repo_path}. Cloning now...")
    git.Repo.clone_from(repo_url, repo_path)
    print(current_path)

    file_paths = [
        os.path.join(repo_path, f"stock_data_120_part_{i}.csv") for i in range(1, 7)
    ]

# Load and concatenate all CSVs into one DataFrame
dfs = []
for fp in file_paths:
    try:
        df = pd.read_csv(fp)
        dfs.append(df)
        print(f"Loaded: {fp}")
    except Exception as e:
        print(f"Error reading {fp}: {e}")

# Combine all into one DataFrame for later use
data = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {data.shape}")
print(data.head())

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

# Make full grid of all symbols and all dates
all_symbols = data_count["symbol"].unique()
full_index = pd.MultiIndex.from_product(
    [all_symbols, date_series], names=["symbol", "date"]
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
data_count.to_parquet("clean_data.parquet", index=True)

