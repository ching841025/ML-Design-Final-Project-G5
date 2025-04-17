<h1 align="center">Stock Selection Based on Daily Trading Data</h1>

<p align="center">
  <img src="https://entrepreneurhandbook.co.uk/wp-content/uploads/2022/10/Trading-Dashboard.jpg.webp" alt="Trading Dashboard" width="700"/>
</p>

<p align="center">
  <sub>
    Group 5 — Lucas Faulkner, Frank Myers, Xin Tang, Amy Chen, Dominic DiSalvo
  </sub>
</p>
<hr/>


## PROJECT DESCRIPTION

Which stock will bring someone the most returns? That's the million-dollar question that has stumped many generations. In today's world, Wall Street traders have developed an AI-derived algorithm known as High Frequency Trading (HFT) to quickly trade stocks based on patterns seen in the market today. This has changed the game, especially for people who don't buy into their services: they simply cannot keep up.

Realizing that the stock market is very volatile and virtually impossible to model, our group wanted to create a machine learning model that could predict the top stocks, allowing them to 'beat the market' without predicting the price point of the specific stock itself.

We used the Alpaca API to pull 120 months worth of stock history. From here, we cleaned the data, taking out any stocks with missing values during the observation period and ones with extreme outliers. This reduced data noise, allowing us to focus on good, strong stocks. 

Our Random Forest model was trained using a Random Forest algorithm, with MLFlow to track the logs. Half of our model was training with the other half testing. This was primarily based on daily stock return. In our backtesting, based on the highest predicted returns, top 25 stocks were selected for trading.

From here, we calculated some evaluation metrics. This includes the daily and annual Sharpe Ratio, cumulative returns, and maximum drawdown. We compared it to the SPY benchmark, and logged these for visual in MLFlow. 
