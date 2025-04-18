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

## CODE RUNNING INSTRUCTIONS

Due to GitHub's sizing constraints, we had to compress the data into a zipped file. To run this successfully, you will need ot download the whole repository, unzip the dataset, and make sure it is in the same folder as the python script. Next, you will install the packages using:
#### pip install -r requirements
Next, you will run the script using:
#### python BANA7075_Final_Project_Data_Engineering_Rev0.py

For the updated 120 month portion of the code, the zipped file is too large to hard push to the repository using terminal. To download this data, go to the link below, download, unzip, and place into the project folder as before.
https://mailuc-my.sharepoint.com/:u:/g/personal/faulknla_mail_uc_edu/ERBXK7iYtGhFgUg8sebcAwsBY-ID_xpo0enmC6C4m8g_Gw?
