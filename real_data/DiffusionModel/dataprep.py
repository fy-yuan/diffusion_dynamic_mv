import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD


from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD

def compute_technical_indicators(data):
    """
    Computes technical indicators for a given dataset of a stock.

    Parameters:
    - data (pd.DataFrame): DataFrame containing stock data with columns:
                           ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
    - pd.DataFrame: Enhanced DataFrame with additional columns for technical indicators and lagged returns.
    """
    # Ensure the dataset has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    # Simple Moving Average (SMA)
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()

    # Exponential Moving Average (EMA)
    data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
    data['EMA_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()

    # Relative Strength Index (RSI)
    data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()

    # Moving Average Convergence Divergence (MACD)
    macd_indicator = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()

    # Returns and lagged returns
    data['Return'] = data['Close'].pct_change()
    data['Lag_Return_1'] = data['Return'].shift(1)
    data['Lag_Return_2'] = data['Return'].shift(2)
    data['Lag_Return_3'] = data['Return'].shift(3)

    # Tomorrow's return (forward-looking)
    data['Tomorrow_Return'] = data['Return'].shift(-1)

    # Dropping NaN values caused by rolling calculations
    data = data.dropna()

    return data



def build_tech_dataset(
    tickers: list,
    start: str = "2015-01-01",
    end:   str = "2025-05-12",
    interval: str = "1d",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Downloads data for each ticker, computes indicators, and concatenates into one DataFrame.
    """
    all_dfs = []
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False
        )
        df.columns = ["Close", "High", "Low", "Open", "Volume"]

        df = compute_technical_indicators(df)
        df['Ticker'] = ticker
        all_dfs.append(df)

    # Combine and reset index
    combined = pd.concat(all_dfs, axis=0)
    combined = combined.reset_index().rename(columns={'Date': 'Datetime'})

    # --- NEW: sort by Ticker then time, and move Ticker/Datetime to front ---
    combined = pd.concat(all_dfs, axis=0).reset_index().rename(columns={'Date': 'Datetime'})

    # Set MultiIndex (Datetime, Ticker) and sort by date
    panel = combined.set_index(['Datetime', 'Ticker']).sort_index(level='Datetime')
    return panel



def create_lstm_dataset_seq2seq(
    panel_df,
    feature_cols,            # e.g. ['SMA_20', …, 'Lag_Return_3']
    target_col='Tomorrow_Return',
    window_size=30,
    pred_len=1               # <- NEW: horizon for every point in the window
):
    """
    X  → (n_samples, window_size,              n_feat * n_tickers)
    y  → (n_samples, window_size, pred_len,    n_tickers)
    -------------------------------------------------------------
    For every j in the input window we attach a trajectory
    of length `pred_len` for all tickers.
    """

    # --- wide (date-indexed) matrices ---------------------------------------
    feats_wide = panel_df[feature_cols].unstack(level='Industry')
    targ_wide  = panel_df[[target_col]].unstack(level='Industry')[target_col]

    X_all = feats_wide.values                     # (n_dates, n_feat * n_tickers)
    y_all = targ_wide.values                      # (n_dates,            n_tickers)

    n_dates, total_feats = X_all.shape
    n_tickers           = y_all.shape[1]

    # need room for the history window *and* the forecast horizon
    n_samples = n_dates - window_size - pred_len + 1
    if n_samples <= 0:
        raise ValueError("window_size + pred_len is larger than the dataset")

    # --- allocate output arrays --------------------------------------------
    X = np.empty((n_samples+1, window_size, total_feats), dtype=X_all.dtype)
    y = np.empty((n_samples+1, window_size, pred_len, n_tickers), dtype=y_all.dtype)

    # --- sliding window -----------------------------------------------------
    for i in range(n_samples+1):
        # history
        X[i] = X_all[i : i + window_size]

        # targets: for each position j inside the window grab the next pred_len rows
        for j in range(window_size):
            y[i, j] = y_all[i + j  : i + j  + pred_len]
    y= np.cumsum(y , axis = 2)
    return X, y