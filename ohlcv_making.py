"""
This script processes tick/execution financial data into OHLCV time bars
and visualizes price movements and returns.
"""

from week2.Day02 import input_size
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

warnings.simplefilter("ignore")

# Define default paths (support both raw execution data and precomputed time bar)
BASE_DIR = Path(__file__).resolve().parent.parent
EXEC_DATA_PATH = BASE_DIR / "db" / "bybit_BTCUSD_2022_3.pkl"
ALT_EXEC_DATA_PATH = BASE_DIR / "db" / "bybit_BTCUSD_2022_2_3.pkl"
TIME_BAR_PATH = BASE_DIR / "db" / "bybit_BTCUSD_2022_3_time_bar.pkl"


def create_bar_data(exec_data: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Resample raw execution / tick data into OHLCV (Open, High, Low, Close, Volume) time bars.

    Args:
        exec_data (pd.DataFrame): Execution tick data with 'price' and 'size' columns.
            Index can be a DatetimeIndex, or a 'timestamp' column must be present.
        freq (str): Sampling frequency string (e.g., '15min', '1h', '1D').

    Returns:
        pd.DataFrame: Resampled OHLCV DataFrame with columns ['op', 'hi', 'lo', 'cl', 'volume'].
    """
    df = exec_data.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
        else:
            raise KeyError("Execution data must contain a 'timestamp' column or have a DatetimeIndex.")

    # Sort chronological index if not already sorted
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)

    # 1. Compute OHLC from execution price
    ohlc = df["price"].resample(freq).ohlc()
    ohlc.columns = ["op", "hi", "lo", "cl"]

    # 2. Compute total volume from size
    if "size" in df.columns:
        volume = df["size"].resample(freq).sum().rename("volume")
    elif "volume" in df.columns:
        volume = df["volume"].resample(freq).sum().rename("volume")
    else:
        volume = pd.Series(0, index=ohlc.index, name="volume")

    # Combine into OHLCV
    bar_data = pd.concat([ohlc, volume], axis=1)
    bar_data.index.name = "timestamp"

    # Forward-fill closing prices for periods without transactions (if any), then drop remaining NaNs
    bar_data["cl"] = bar_data["cl"].ffill()
    bar_data["op"] = bar_data["op"].fillna(bar_data["cl"])
    bar_data["hi"] = bar_data["hi"].fillna(bar_data["cl"])
    bar_data["lo"] = bar_data["lo"].fillna(bar_data["cl"])
    bar_data["volume"] = bar_data["volume"].fillna(0)

    return bar_data


def visualize_bar(bar_data: pd.DataFrame, title: str = "BTC/USD 15-Minute Time Bar") -> None:
    """Visualize OHLC price trajectories and trading volume.

    Args:
        bar_data (pd.DataFrame): DataFrame containing ['op', 'hi', 'lo', 'cl', 'volume'].
        title (str): Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Price Plot (OHLC)
    price_cols = [col for col in ["op", "hi", "lo", "cl"] if col in bar_data.columns]
    bar_data[price_cols].plot(ax=ax1, linewidth=1.2)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Volume Plot
    if "volume" in bar_data.columns:
        ax2.bar(
            bar_data.index,
            bar_data["volume"].astype(float),
            width=0.008,
            color="gray",
            alpha=0.6,
            label="Volume",
        )
        ax2.set_ylabel("Volume", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left")

    plt.xlabel("Timestamp", fontsize=12)
    plt.tight_layout()
    plt.show()

def preprocess_log(bar_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess bar data to compute log returns and add to the DataFrame."""
    target_cols = ["op", "hi", "lo", "cl"]
    for col in target_cols:
        log_col = f"log_{col}"
        diff_log_col = f"diff_log_{col}"
        bar_data[log_col] = np.log(bar_data[col])
        bar_data[diff_log_col] = bar_data[log_col].diff(1)
    
    # Remove the first row with NaN values
    bar_data = bar_data.dropna()
    
    return bar_data

def time_series_train_test_split(data: pd.DataFrame, train_size: float = 0.8) -> pd.DataFrame:
    """Split data into training and testing sets."""
    length = len(data)
    split_index = int(length * train_size)
    targets = ["diff_log_op","diff_log_hi","diff_log_lo","diff_log_cl"]
    features = [col for col in data.columns if col not in targets]
    X_train = data[features].iloc[: split_index]
    X_test = data[features].iloc[split_index:]
    y_train = data[targets].iloc[: split_index]
    y_test = data[targets].iloc[split_index:]
    return X_train, X_test, y_train, y_test

def eval_direction_accuracy(target: np.ndarray, pred: np.ndarray) -> float:
    """Evaluate direction accuracy."""
    from sklearn.metrics import accuracy_score
    target_direction = np.sign(target)
    pred_direction = np.sign(pred)
    accuracy = accuracy_score(target_direction, pred_direction)
    return accuracy

def adf_test(series, sig_level = 0.05) -> bool:
    from statsmodels.tsa.stattools import adfuller
    """
    This function conducts Augmented Dickey-Fuller test to check stationarity.
    H0: Data is non-stationary vs H1: Data is stationary.
    If p value is smaller than sig_level, we reject H0 and conclude that the data is stationary.
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    if p_value <= sig_level:
        print(f"p-value: {p_value:.4f} <= {sig_level}. Reject H0, data is stationary.")
        return True
    else:
        print(f"p-value: {p_value:.4f} > {sig_level}. Do not reject H0, data is non-stationary.")
        return False

def linear_regression(X, y) -> float:
    """
    Simple Linear Regression := y = A1x1 + A2x2 + ... + Akxk + b
    Returns an accuracy score.
    """
    X_train, y_train, X_test, y_test = time_series_train_test_split(X, y)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.show()
    accuracy = eval_direction_accuracy(y_test, y_pred)
    print(f"Linear Regression Accuracy: {accuracy:.4f}")
    return accuracy

def random_forest_regression(X, y):
    """
    Random Forest Regression
    """
    from sklearn.ensemble import RandomForestRegressor
    X_train, y_train, X_test, y_test = time_series_train_test_split(X, y)
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.show()
    accuracy = eval_direction_accuracy(y_test, y_pred)
    print(f"Random Forest Regression Accuracy: {accuracy:.4f}")
    return accuracy

from torch import nn

class LSTMRegressor(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, batch_size: int):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=dim_in, hidden_size=dim_hidden, batch_first=True
        )
        self.output_layer = nn.Linear(dim_hidden, dim_out)
    def forward(self, inputs):
        ## [Batch, Sequence, Input_features]
        h, _ = self.lstm(inputs)
        ## [Batch, Hidden_features]
        output = self.output_layer(h[:, -1, :])
        ## [Batch, Output_features]
        return output

from torch.utils.data import Dataset
class TimeBarDataset(Dataset):
    pass

def lstm_bar_data(X, y):
    pass

from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, batch_size, learning_rate, num_epochs, model, criterion, optimizer, evaluator):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.batch_size, shuffle=False)
        
    def train(self):
        losses_train = []
        losses_eval = []
        for epoch in range(self.num_epochs):
            loss_train = self.run_epoch("train", epoch)
            loss_eval = self.run_epoch("eval", epoch)
            losses_train.append(loss_train)
            losses_eval.append(loss_eval)
        self.visualize_losses(losses_train, losses_eval)
        return losses_train, losses_eval

    def run_epoch(self, mode: str, epoch: int) -> float:
        if mode == "train":
            loader = self.train_loader
        else:
            loader = self.eval_loader
        
        with torch.set_grad_enabled(mode == "train"):
            total_loss = 0
            total_eval = []
            for batch_idx in range(len(loader) // self.batch_size + 1):
                inputs, targets = loader[batch_idx*self.batch_size: (batch_idx+1)*self.batch_size]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
                if mode == "eval":
                    self.evaluator.add_batch(targets, outputs)
        return total_loss / len(loader)

    def visualize_losses(self, losses_train: list[float], losses_eval: list[float]) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(losses_train, label="Train Loss")
        plt.plot(losses_eval, label="Eval Loss")
        plt.legend()
        plt.show()

def main() -> None:
    """Main execution entry point: loads data, creates time bars, and visualizes."""
    # 1. Load data from available pickle files
    if EXEC_DATA_PATH.exists():
        print(f"Loading raw execution tick data from: {EXEC_DATA_PATH}")
        exec_data = pd.read_pickle(EXEC_DATA_PATH)
        bar_data = create_bar_data(exec_data[: "2022-03-01"], freq="15min")
    elif TIME_BAR_PATH.exists():
        print(f"Loading precomputed time bar data from: {TIME_BAR_PATH}")
        bar_data = pd.read_pickle(TIME_BAR_PATH)
        # Sample the first day for clean visualization
        bar_data = bar_data.loc[: "2022-03-01 23:45:00"]
    else:
        raise FileNotFoundError(
            f"Neither execution data ({EXEC_DATA_PATH}) nor time bar data ({TIME_BAR_PATH}) was found."
        )

    print("\n--- Processed OHLCV Bar Data (Head) ---")
    print(bar_data.head())

    # 2. Visualize the bar data
    visualize_bar(bar_data, title="BTC/USD Time Bar (2022-03-01)")

    # Log Return Series
    log_return = preprocess_log(bar_data)


if __name__ == "__main__":
    main()