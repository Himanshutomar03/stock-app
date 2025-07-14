import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("Stock Price Prediction & Portfolio Optimization")
MODE = st.sidebar.selectbox("Select Mode", ["Breakout Strategy", "Portfolio Optimization", "Intraday Prediction"])

if MODE == "Breakout Strategy":
    st.header("Breakout Strategy with Support/Resistance")
    stocks = st.text_input("Enter Ticker Symbols (comma separated):")
    if stocks:
        stock_list = [s.strip().upper() for s in stocks.split(",") if s.strip()]
        n = 10
        for symbol in stock_list:
            st.subheader(f"{symbol}")
    
            print(f"\nProcessing {symbol}...")
            df = yf.download(symbol, period='1y', interval='1d', auto_adjust=True)
    
            if df is None or df.empty:
                st.warning(f"Could not retrieve data for {symbol}. Skipping.")
                continue

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
    
            df['resistance'] = df.iloc[argrelextrema(df['High'].to_numpy(), np.greater_equal, order=n)[0]]['High']
            df['support'] = df.iloc[argrelextrema(df['Low'].to_numpy(), np.less_equal, order=n)[0]]['Low']
    
            df['resistance'] = df['resistance'].ffill()
            df['support'] = df['support'].ffill()
    
            # Vectorized breakout strategy for performance and reliability
            df['signal'] = 0  # Default to no signal
            # Buy signal when close breaks above resistance
            df.loc[df['Close'] > df['resistance'], 'signal'] = 1
            # Sell signal when close breaks below support
            df.loc[df['Close'] < df['support'], 'signal'] = -1
    
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.set_title('Support/Resistance Breakout Strategy')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.plot(df.index, df['Close'], label='Close', linewidth=1.5)
            ax.plot(df.index, df['resistance'], label='Resistance', linestyle='--', color='red', alpha=0.5)
            ax.plot(df.index, df['support'], label='Support', linestyle='--', color='green', alpha=0.5)
            ax.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1],
                    marker='^', color='green', label='Buy', alpha=0.8)
            ax.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1],
                    marker='v', color='red', label='Sell', alpha=0.8)
            
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            st.pyplot(fig)

elif MODE == "Portfolio Optimization":
    st.header("Portfolio Optimization with Efficient Frontier")
    # Input Section
    tickers_input = st.text_input("Enter comma-separated stock tickers (e.g., TCS.NS, INFY.NS):")
    capital = st.number_input("Enter total capital to invest :",min_value=100.0, value=10000.0)
    
    if tickers_input and capital:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        
        st.info("Fetching stock data...")
        adj_close_df = pd.DataFrame()
        for ticker in tickers:
            data = yf.download(ticker, start="2020-01-01", end="2025-06-30", auto_adjust=True)
            if data is not None and not data.empty and 'Close' in data.columns:
                adj_close_df[ticker] = data['Close']
    
        if adj_close_df.empty or adj_close_df.shape[0] < 30:
            st.error("Not enough historical data for analysis. Please try different tickers.")
            st.stop()
    
        adj_close_df.ffill(inplace=True)
        adj_close_df.bfill(inplace=True)
    
        # Utility Functions
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=periods).mean()
            avg_loss = loss.rolling(window=periods).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
    
        def prepare_features(df, ticker):
            df = df[[ticker]].copy()
            df['LogReturn'] = np.log(df[ticker] / df[ticker].shift(1))
            df['Lag1'] = df['LogReturn'].shift(1)
            df['Lag5'] = df['LogReturn'].shift(5)
            df['MA5'] = df[ticker].rolling(window=5).mean()
            df['MA10'] = df[ticker].rolling(window=10).mean()
            df['Vol10'] = df['LogReturn'].rolling(window=10).std()
            df['RSI'] = calculate_rsi(df[ticker])
            df = df.dropna()
            X = df[['Lag1', 'Lag5', 'MA5', 'MA10', 'Vol10', 'RSI']]
            y = df['LogReturn']
            return X, y
    
        predicted_returns = {}
        valid_tickers = []
        
        st.subheader("XGBoost Model Performance")
        st.write(f"{'Ticker':<15} {'MAE':<10} {'RMSE':<10} {'R² Score':<10}")
        st.write("-" * 45)
    
        for ticker in tickers:
            try:
                X, y = prepare_features(adj_close_df, ticker)
                if len(X) < 30:
                    st.warning(f"Not enough data to train ML model for {ticker}. Skipping.")
                    continue
    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'reg_alpha': [0, 0.1],
                    'reg_lambda': [1]
                }
                xgb = XGBRegressor(random_state=42)
                grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_xgb = grid_search.best_estimator_
    
                y_pred = best_xgb.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                st.write(f"{ticker:<15} {mae:.6f} {rmse:.6f} {r2:.6f}")
    
                latest_features = X.iloc[-1:].to_numpy()
                predicted_daily_return = best_xgb.predict(latest_features)[0]
                predicted_annual_return = predicted_daily_return * 252
                predicted_returns[ticker] = predicted_annual_return
                valid_tickers.append(ticker)
            except Exception as e:
                st.warning(f"Error with {ticker}: {e}")
    
        if not valid_tickers:
            st.error("No valid stocks with sufficient data for ML prediction.")
            st.stop()
    
        # Portfolio Optimization
        st.subheader("Portfolio Optimization (Efficient Frontier)")
        mu = expected_returns.mean_historical_return(adj_close_df[valid_tickers]) * 252
        S = risk_models.CovarianceShrinkage(adj_close_df[valid_tickers]).ledoit_wolf() * 252
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=0.06)
        optimal_weights = ef.clean_weights()
        weights_dict = {ticker: weight for ticker, weight in optimal_weights.items()}
    
        st.write("**Optimal Weights:**")
        st.json(weights_dict)
    
        port_return, port_volatility, sharpe = ef.portfolio_performance(risk_free_rate=0.06)
        st.write(f"**Expected Annual Return:** {port_return:.2%}")
        st.write(f"**Expected Volatility:** {port_volatility:.2%}")
        st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
    
        # Discrete Allocation
        st.subheader("Discrete Allocation")
        latest_prices = get_latest_prices(adj_close_df[valid_tickers])
        da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=int(capital))
        allocation, leftover = da.greedy_portfolio()
    
        alloc_df = pd.DataFrame([
            {"Ticker": k, "Shares": v, "Price per Share": latest_prices[k], "Total Value": v * latest_prices[k]}
            for k, v in allocation.items()
        ])
        st.dataframe(alloc_df)
        st.write(f"**Total Invested Amount:** {alloc_df['Total Value'].sum():,.2f}")
        st.write(f"**Remaining Capital:** {leftover:.2f}")
    
        # Visualization
        st.subheader("Visualizations")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
        # Pie Chart
        axs[0, 0].pie([weights_dict[t] for t in valid_tickers], labels=valid_tickers, autopct='%1.1f%%', startangle=140)
        axs[0, 0].set_title("Optimal Portfolio Allocation")
    
        # Efficient Frontier
        ef_plot = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        plotting.plot_efficient_frontier(ef_plot, ax=axs[0, 1], show_assets=False)
        for ticker in valid_tickers:
            axs[0, 1].scatter(np.sqrt(S.loc[ticker, ticker]), mu[ticker], s=50, label=ticker)
        axs[0, 1].scatter(port_volatility, port_return, marker='*', color='r', s=200, label='Optimal Portfolio')
        axs[0, 1].set_title('Efficient Frontier')
        axs[0, 1].legend()
    
        # Historical Performance
        log_returns = np.log(adj_close_df[valid_tickers] / adj_close_df[valid_tickers].shift(1)).dropna()
        portfolio_value = (log_returns.mul([weights_dict[t] for t in valid_tickers], axis=1)
                           .sum(axis=1)).cumsum().apply(np.exp)
        axs[1, 0].plot(portfolio_value.index, portfolio_value, label="Optimized Portfolio")
        axs[1, 0].set_title("Historical Portfolio Performance")
        axs[1, 0].grid(True)
        axs[1, 0].legend()
    
        axs[1, 1].axis('off')  # empty
    
        st.pyplot(fig)


elif MODE == "Intraday Prediction":
    st.header("Intraday Prediction with Random Forest")
    ticker = st.text_input("Enter Ticker:","HAL.NS")
    if ticker:
        df = yf.download(ticker, period='10d', interval='30m', auto_adjust=True)
        if df is None or df.empty:
            st.error(f"Could not retrieve data for {ticker}.")
            st.stop()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        st.write("### R² Score:", r2_score(y_test, y_pred))
        st.write("### MSE:", mean_squared_error(y_test, y_pred))
    
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_test.index, y_test, label='Actual', color='blue')
        ax.plot(y_test.index, y_pred, label='Predicted', color='red')
        ax.set_title(f"Intraday Price Prediction for {ticker}")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig) 