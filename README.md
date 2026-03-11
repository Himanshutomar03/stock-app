🌐 Live Demo

👉 Click here to view the deployed app
https://stock-app-vfjfzcdbemkc3elf4qyttg.streamlit.app/

📈 Stock Price Prediction & Portfolio Optimization
This is a Streamlit web app that uses Machine Learning and Modern Portfolio Theory to predict stock returns and optimize your portfolio for maximum performance.

ℹ️ About
This project aims to assist retail investors and finance enthusiasts by:

- Leveraging **XGBoost regression** to predict future stock returns based on historical price patterns and log-return features.
- Using **Random Forest** with enriched features (MA5, MA20, RSI) for next-day price prediction, with detailed error metrics.
- Applying **Efficient Frontier** theory from Modern Portfolio Theory to allocate capital in a risk-adjusted optimal way.
- Providing an **interactive web interface** to input stock tickers, capital, and view optimized portfolio allocation.
- Detecting **price breakouts** using support and resistance levels computed with local extrema.

Whether you're a beginner investor or a data-driven trader, this app helps you make smarter portfolio decisions.

🚀 Features
- 🔮 Predict next-day stock prices using **Random Forest** with technical indicators (MA5, MA20, RSI)
- 📉 Display model evaluation metrics: **R², MAE, RMSE, MSE**
- 📊 Overlay **5-day and 20-day moving averages** on price charts
- 📡 **Breakout Strategy** with buy/sell signals based on support & resistance
- 💰 Optimize portfolio allocation with **Efficient Frontier**
- 💼 Allocate capital precisely with **Discrete Allocation**
- 🌐 Supports **US and Indian** stock tickers (e.g., AAPL, MSFT, TCS.NS, INFY.NS)

🧠 Technologies Used

- Python
- Streamlit
- yfinance
- XGBoost
- PyPortfolioOpt
- scikit-learn
- matplotlib
- NumPy
- pandas
- SciPy

