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

def visualize_performance():
    # 1. Configuration
    cfg = {
        "trade_config": {
            "warmup_period": 1000,
            "initial_margin_balance": "100000USDT",
            "strategy_timeframe": "60min",
            "max_leverage": 2,
            "min_margin_rate": 0.1
        },
        "backtester_config": {
            "ohlcv_data_path": os.path.join(HOME_PATH, "data", "public.pkl"),
            "external_data_paths": [os.path.join(HOME_PATH, "data", "public_froi.pkl")],
            "time_zone": "Asia/Tokyo",
            "start_date": datetime.date(2021, 2, 1),
            "end_date": datetime.date(2023, 4, 30),
            "exchange": "binance",
            "symbol": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
            "backtest_timeframe": "60min",
            "slippage": 0.01,
            "delay": 0,
            "use_wandb": False,
            "save_model": True,
            "logging": True,
            "position_in_fiat": True,
            "daily_position": False,
            "backtest_num_worker": "max",
            "get_model_num_worker": "max",
            "compounding_strategy": False
        },
        "exchange_config": {
            "BTCUSDT": {},
            "ETHUSDT": {},
            "XRPUSDT": {}
        },
        "cv": {
            "type": "cpcv", 
            "n_purge": 10, 
            "n_path": 4 
        },
    }

    print("Loading data...")
    # For speed optimization, using a limited dataframe. Remove [:6000] to use all data.
    df = pd.read_pickle(cfg["backtester_config"]["ohlcv_data_path"])
    df = df.iloc[:6000]

    # 2. Score Calculation
    print("Calculating score via CPCV Backtest...")
    scoring = Scoring(
        config=cfg,
        Strategy=Strategy,
        raw_df=df,
    )
    score = scoring.run()
    print(f"Mean Score (Sharpe Ratio): {score}")
    scoring.finish()

    # 3. Signal Visualization
    print("Visualizing Signals...")
    strategy = Strategy(cfg)
    preprocessed_df = strategy.preprocess(df)
    models = strategy.get_model(preprocessed_df)
    signals_df = strategy.get_signal(preprocessed_df, models)

    # Extract data for the first month
    start_date = df.index.get_level_values('timestamp').min()
    end_date = start_date + pd.DateOffset(months=1)
    filtered_signals_df = signals_df.loc[(slice(start_date, end_date), slice(None)), :]

    # Plot for each symbol
    symbols = cfg["backtester_config"]["symbol"]
    fig, axs = plt.subplots(len(symbols), 1, figsize=(10, 8), sharex=True)
    
    for i, symbol in enumerate(symbols):
        symbol_signals_df = filtered_signals_df.xs(symbol, level='symbol')
        axs[i].plot(symbol_signals_df.index, symbol_signals_df['signal'], label=f'{symbol} Signal')
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_ylabel('Signal')
        axs[i].legend()

    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(HOME_PATH, "strategy_ver0_signals.png")
    plt.savefig(plot_path)
    print(f"Saved signal visualization to: {plot_path}")
    plt.show()

def signal_plot(sample_df: pd.DataFrame, signal_df: pd.DataFrame, annualized_vola_df: pd.DataFrame, symbol: str, start_date: str, end_date: str):
    signal_symbol_df = signal_df.loc[(slice(None), symbol), :].copy()

    # Calculate target position size from signal
    d = 0.35
    signal_symbol_df['target_position_size'] = signal_symbol_df['signal'].apply(
        lambda m: 0.0 if pd.isna(m) else math.floor(m / d) * 0.5 if m >= 0 else math.ceil(m / d) * 0.5  # if NaN, set to 0.0
    )

    # extract data
    date_filtered_df = signal_symbol_df.loc[(slice(start_date, end_date), slice(None)), :]

    # Calculate bet size and calculate each time as latest_timestamp
    bet_sizes = []
    signals = []
    target_positions = []
    volas = []
    timestamps = []

    for latest_timestamp in date_filtered_df.index.get_level_values(0).unique():

        latest_bar = sample_df.loc[(latest_timestamp, symbol), :]

        # Calculate annualized volatility
        relevant_vola_df = annualized_vola_df.xs(symbol, level='symbol')
        relevant_vola = relevant_vola_df[relevant_vola_df.index <= latest_timestamp].iloc[-1]['volatility']
        annualized_vola = relevant_vola / latest_bar['close']

        # Calculate signal, target position size, and bet size
        signal = date_filtered_df.loc[(latest_timestamp, symbol), 'signal']
        target_position_size = date_filtered_df.loc[(latest_timestamp, symbol), 'target_position_size']
        bet_size = (target_position_size * 0.5) / annualized_vola

        signals.append(signal)
        target_positions.append(target_position_size)
        bet_sizes.append(bet_size)
        volas.append(annualized_vola)
        timestamps.append(latest_timestamp)

    # plot
    plt.figure(figsize=(14, 8))
    plt.plot(timestamps, signals, label='Signal', linestyle=':')
    plt.plot(timestamps, target_positions, label='Target Position Size', linestyle='--')
    plt.plot(timestamps, bet_sizes, label='Position Size')
    plt.plot(timestamps, volas, label='Annualized Volatility', linestyle='-.')

    # Customize y-axis ticks
    """
    yticks = np.arange(-2.0, 2.0, 1.0)
    yticks = np.append(yticks, [-0.7, -0.35, 0.35, 0.7])
    yticks = np.sort(yticks)
    plt.yticks([-0.7, -0.35, 0.35, 0.7])
    """
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
