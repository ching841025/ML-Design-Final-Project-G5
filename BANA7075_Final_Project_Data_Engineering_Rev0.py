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

#Rank the top 500 stocks each day by trade count.
for i in range(len(date_series)):
    rank_wdf = data[data["date"] == date_series[i]]
    print(i, len(rank_wdf), date_series[i])
    rank_sdf = rank_wdf.sort_values(by = ["trade_count"], ascending = False)
    
    rank_sdf["rank"] = range(1, len(rank_sdf)+1)
    
    if i == 0:
        final_df = rank_sdf
        print(i, len(rank_wdf), date_series[i])
    elif i == date_diff.days:
        print(i, len(rank_wdf), date_series[i]) 
        initial_df = rank_sdf
        final_df = pd.concat([final_df, initial_df])       
    else:
        initial_df = rank_sdf
        final_df = pd.concat([final_df, initial_df])

data_count = final_df[final_df["rank"] <= 500]

del(i)
del(initial_df)
del(final_df)
del(rank_wdf)
del(rank_sdf)

#Fill in missing dates
#md_wdf - missing date working dataframe
symbol_list = data_count['symbol'].unique().tolist()

for i in range(len(symbol_list)): 
    md_wdf = data_count[data_count["symbol"] == symbol_list[i]].set_index('date')
    md_wdf = md_wdf.reindex(date_series)
    md_wdf.index.name = 'date'
    md_wdf["symbol"] = symbol_list[i]    
    
    if i == 0:
        final_df = md_wdf
    else:
        initial_df = md_wdf 
        final_df = pd.concat([initial_df,final_df])

data_count = final_df

del(i)
del(initial_df)
del(final_df)
del(md_wdf)

#Calculate the daily change for each stock 
# daily_ret {"RET"} = ((today_close - yesterday_close) / yesterday_close) * 100
#df_wdf - daily returns working dataframe

for i in range(len(symbol_list)):  
    dr_wdf = data_count[data_count["symbol"] == symbol_list[i]].sort_index()
    dr_wdf["RET"] = (dr_wdf.close - dr_wdf.close.shift(1))/dr_wdf.close.shift(1) * 100
    
    if i == 0:
        final_df = dr_wdf
    else:
        initial_df = dr_wdf 
        final_df = pd.concat([initial_df,final_df])    

data_ret = final_df
data_ret['RET'] = data_ret['RET'].fillna(-100)

del(i)
del(initial_df)
del(final_df)
del(dr_wdf)
del(data_count)

# Calculate the 15 day momenturm like factor - skew
#window_mom - is the size of the factor

window_mom = 15
data_ret[['RET_mean', 'RET_std','RET_skew','RET_kurt']] = (
    data_ret.groupby('symbol')['RET']
            .rolling(window_mom)
            .agg(['mean','std','skew','kurt'])
            .shift(1)
)

# Adding into the US 3 month T-Bill dataset for the WSJ Market Dataset
# https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD03M/historical-prices
# risk free return proxy
# RET_rf - daily risk free returns 
data_tbill = pd.read_excel("HistoricalPrices 3m TBILL.xlsx")
data_tbill = data_tbill.sort_values(by = ['Date'])
data_tbill['RET_rf'] = (data_tbill.Close - data_tbill.Close.shift(1)) / data_tbill.Close.shift(1) * 100
data_tbill['Date'] = data_tbill.Date.dt.date

data_tbill = data_tbill.set_index('Date')

data_ret = data_ret.join(data_tbill['RET_rf'])


