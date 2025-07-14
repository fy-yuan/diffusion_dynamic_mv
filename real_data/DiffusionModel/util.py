from DiffusionModel.sde_lib import VPSDE, subVPSDE, VESDE
import torch
import numpy as np
import random
import os


def get_noise_fn(sde, model, continuous=False):
  """Wraps `noise_fn` so that the model output corresponds to a real time-dependent noise function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A noise prediction function.
  """
  # model_fn = get_model_fn(model, train=train)

  if isinstance(sde, VPSDE) and continuous:
    def noise_fn(x, t, cond):
      # For VP-trained models, t=0 corresponds to the lowest noise level
      # The maximum value of time embedding is assumed to 999 for
      # continuously-trained models.
      labels = t * (sde.N -  1)
      noise = model(x, labels, cond)
      return noise

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return noise_fn


def get_score_fn(sde, model, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, t, cond):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * (sde.N - 1)
        score = model(x, t, cond)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model(x, t, cond)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, t, cond):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model(x, labels, cond=cond)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def seed_torch(seed=1029):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  # torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------
# Helper functions extracted from ScoreGrad.ipynb
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_ou(theta: float, mu: float, sigma: float, T: int, x0: float = None, dt: float = 1.0) -> np.ndarray:
    """Simulate a single path of the Ornstein–Uhlenbeck (OU) process.

    dX_t = θ(μ − X_t) dt + σ dW_t
    """
    if x0 is None:
        x0 = mu
    X = np.empty(T, dtype=np.float32)
    X[0] = x0
    sqrt_dt_sigma = sigma * np.sqrt(dt)
    for t in range(1, T):
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sqrt_dt_sigma * np.random.randn()
    return X


def create_window_sequences(data: np.ndarray, window: int, compare: bool = True) -> tuple:
    """Create sliding‑window (X,y) pairs for sequence‑to‑sequence training."""
    if window < 1:
        raise ValueError("`window` must be at least 1")
    total_samples = data.shape[0] - window
    if total_samples <= 0:
        raise ValueError("`window` is too large for the provided data length.")
    X = np.stack([data[i : i + window] for i in range(total_samples)])
    if compare:
       y = np.stack([data[i  : i + window] for i in range(total_samples)])
       return X, y
    else:
       return X, None


# --- Technical‑indicator helpers ------------------------------------------------

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD


from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a set of common technical indicators on OHLCV data."""
    #required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    #if not required.issubset(df.columns):
    #    raise ValueError(f"Input data missing required OHLCV columns: {required - set(df.columns)}")

    close = df['Close']

    # Moving averages
    df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()

    # RSI
    df['RSI'] = RSIIndicator(close, window=14).rsi()

    # MACD
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Returns and lags
    df['Return'] = close.pct_change()
    for k in range(1, 4):
        df[f'Lag_Return_{k}'] = df['Return'].shift(k)

    # Tomorrow’s return (for supervised targets)
    df['Tomorrow_Return'] = df['Return'].shift(-1)
    
    df = df.dropna()

    return df

def build_tech_dataset(
    tickers: list,
    start: str = "2015-01-01",
    end:   str = "2025-01-01",
    interval: str = "1d",
    auto_adjust: bool = True
) -> pd.DataFrame:
    """
    Downloads data for each ticker, computes indicators, and concatenates into one DataFrame.
    """
    all_dfs = []
    print(yf.__version__)
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
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        #df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

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
    panel_df: pd.DataFrame,
    feature_cols,
    target_col: str = 'Tomorrow_Return',
    window_size: int = 20,
):
    """Transform a multi-ticker panel into Seq2Seq (X,y) numpy arrays."""
    feats_wide = panel_df[feature_cols].unstack(level='Ticker')
    targ_wide = panel_df[[target_col]].unstack(level='Ticker')[target_col]

    X_all = feats_wide.values
    y_all = targ_wide.values

    n_dates = X_all.shape[0]
    n_samples = n_dates - window_size 
    if n_samples <= 0:
        raise ValueError("`window_size` is too large for the available date range.")

    X = np.stack([X_all[i : i + window_size] for i in range(n_samples)])
    y = np.stack([y_all[i + 1: i + window_size + 1] for i in range(n_samples)]) 
    return X, y


def plot_predictions(
    history: np.ndarray,
    pred_arr: np.ndarray,
    actual_arr: np.ndarray,
    start: int = 0,
):
   
   window_len = history.shape[0]
   t_hist = np.arange(start-window_len+1, start+1)
   t_fore = np.arange(start, start + len(actual_arr))
   fore_lower = np.percentile(pred_arr,  2.5, axis=1)
   fore_upper = np.percentile(pred_arr, 97.5, axis=1)
   fore_median = np.median(pred_arr, axis=1)
   sample_path = pred_arr[:,0]

   plt.figure(figsize=(10, 5))
   plt.plot(t_hist, history,                label="History",            lw=1.5)
   plt.plot(t_fore, actual_arr,            label="Actual Prices",    ls="--", lw=1.5)
   plt.plot(t_fore, fore_median,                label="Pred. median",    ls="-.")
   plt.plot(t_fore, sample_path,          label="Pred. sample path",    ls="--")
   plt.fill_between(t_fore, fore_lower, fore_upper,  alpha=0.3, label="Pred. 95% CI")
   

   plt.axvline(start, color="gray", linestyle=":")
   plt.xlabel("Time step")
   plt.ylabel("Value of target")
   plt.title("History + Actual vs. Predictions")
   plt.legend()
   plt.tight_layout()
   plt.show()

