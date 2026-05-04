import pandas as pd
import matplotlib.pyplot as plt

def plot_timebar(df: pd.DataFrame):
    """
    Args:
        df: OPHLC Tickers pd.DataFrame with columns ['op', 'hi', 'lo', 'cl']
    """
    df[['op', 'hi', 'lo', 'cl']].plot()
    plt.title("Time Bar Ticker")
    plt.show()

def plot_actual_vs_prediction(correct: pd.Series, pred: np.ndarray):
    """
    Plots to compare the prediction and actual values.
    """
    data_time = correct.index
    plt.figure(figsize=(10, 5))
    plt.plot(data_time, correct.values, label="Actual (Ground Truth)", alpha=0.7)
    plt.plot(data_time, pred, label="Prediction", linestyle="--", alpha=0.7)
    plt.title("Actual vs Prediction (Log Returns)")
    plt.legend()
    plt.tight_layout()
    plt.show()
