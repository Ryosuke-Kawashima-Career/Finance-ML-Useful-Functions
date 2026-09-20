"""
This script processes tick/execution financial data into OHLCV time bars
and visualizes price movements and returns.
"""

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


if __name__ == "__main__":
    main()