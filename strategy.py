import numpy as np
import pandas as pd
import math
from mlbacktester import AssetInfo, BaseStrategy, Order

class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cf
    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df

    @abstractmethod
    def get_signal(self, preprocessed_df: pd.DataFrame, model: Any) -> pd.DataFrame:
        pass

    def get_model(self, train_df: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    def get_orders(
        self,
        latest_timestamp: pd.Timestamp,
        latest_bar: pd.Series,
        latest_signal: pd.Series,
        asset_info: AssetInfo,
    ) -> list[Order]:
        pass

class Strategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.annualized_vola_df = pd.DataFrame()
        self.symbols = cfg["backtester_config"]["symbol"]

    def preprocess(self, df):
        """
        Preprocess the OHLCV data to calculate the annualized volatility of each asset.

        Parameters
        ==========
        df: pandas.DataFrame
            OHLCVデータを含むdataframe

        Returns
        ==========
        preprocessed_df: pandas.DataFrame
            dataframe storing preprocessed data
        """
        # Calculate the annualized risk from the past 30 days of price data
        span = 24 * 7 * 4  # 4 weeks
        df['volatility'] = np.nan

        # Calculate the annualized volatility for each symbol
        for symbol in df.index.get_level_values('symbol').unique():
            # Get the data for each symbol
            symbol_df = df.xs(symbol, level='symbol')

            # Calculate the log return
            log_return = np.log(symbol_df['close']).diff()

            # Calculate the rolling standard deviation
            rolling_std = log_return.rolling(window=span).std()

            # Calculate the annualized volatility
            annualized_vola_symbol = rolling_std * np.sqrt(365.25 * 24)
            annualized_vola_symbol.fillna(1, inplace=True)

            # Store the annualized volatility in the original dataframe
            df.loc[(symbol_df.index, symbol), 'volatility'] = annualized_vola_symbol.values

        # Return the preprocessed dataframe
        preprocessed_df = df.copy()
        return preprocessed_df

    def get_model(self, train_df):
        """
        Optimal parameters are searched for each asset from the given trainig data.

        Parameters
        ==========
        train_df: pd.DataFrame
            DataFrame storing training data partitioned by CPCV.

        Returns
        =======
        models: list
            List storing optimal parameters for each asset.
        """
        df = train_df.copy()

        windows = [9, 14, 18]
        models = []  # models[0]: BTCのbest_window, models[1]: ETHのbest_window, models[2]: XRPのbest_window

        for symbol in self.symbols:
            model = 1
            _df = df.loc[(slice(None), symbol), :].copy()
            best_return = -np.inf

            _df['delta'] = _df['close'].diff()
            _df['gain'] = _df['delta'].clip(lower=0)
            _df['loss'] = _df['delta'].clip(upper=0)

            for window in windows:
                _df['avg_gain'] = _df['gain'].rolling(window=window, min_periods=1).mean()
                _df['avg_loss'] = -_df['loss'].rolling(window=window, min_periods=1).mean()
                _df['RS'] = _df['avg_gain'] / _df['avg_loss']
                _df['RSI'] = 100 - (100 / (1 + _df['RS']))

                _df['signal'] = _df['RSI'] /50 -1 #RSI is converted to signal -1 ~ 1

                _df['daily_return'] = _df['close'].pct_change()
                _df['strategy_return'] = _df['daily_return'] * _df['signal'].shift(1)
                _df['strategy_return'] = _df['strategy_return'].fillna(0)

                cumulative_return = sum(_df['strategy_return'])

                if cumulative_return > best_return:
                    best_return = cumulative_return
                    model = window

            models.append(model)
        return models

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        Use preprocessed_df to create continuous signals.

        Parameters
        ==========
        preprocessed_df: pd.DataFrame
            DataFrame storing preprocessed data
        models: list
            List of optimal parameters returned by get_model.

        Returns
        =======
        df: pd.DataFrame
            DataFrame with "signal" column containing continuous signal values.
        """

        df = preprocessed_df.copy()
        dfs = []

        for symbol, model in zip(self.symbols, models):
            _df = df.loc[(slice(None), symbol), :].copy()

            _df['delta'] = _df['close'].diff()
            _df['gain'] = _df['delta'].clip(lower=0)
            _df['loss'] = -_df['delta'].clip(upper=0)

            _df['avg_gain'] = _df['gain'].rolling(window=model, min_periods=1).mean()
            _df['avg_loss'] = _df['loss'].rolling(window=model, min_periods=1).mean()

            _df['RS'] = _df['avg_gain'] / _df['avg_loss']
            _df['RSI'] = 100 - (100 / (1 + _df['RS']))
            # The column name for storing signal information is fixed to "signal"
            _df['signal'] = _df['RSI'] /50 -1 #RSI is converted to signal -1 ~ 1

            dfs.append(_df)

        df = pd.concat(dfs)
        df = df.sort_index(level=1).sort_index(level=0)
        return df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Create orders based on the order time, the status of positions at that time, and the signal obtained from OHLCV.

        Parameters
        ==========
        latest_timestamp: pandas.Timestamp
            Time to place orders
        latest_bar: pandas.Series
            OHLCV data at the time of placing orders (raw data)
        latest_signal: pandas.Series
            Signal data at the time of placing orders (data created by get_signal function)
        asset_info: dict
            Dictionary storing information about assets at the time of placing orders

        Returns
        =======
        order_lst: list 
            List of orders at the current time (contains Order class objects)
            Contains 'type', 'side', 'size', 'price' items
        """

        order_lst = []
        d = 0.35  # Discretization level
        size_ratio = {"BTCUSDT": 0.1, "ETHUSDT": 1.5, "XRPUSDT": 4000}  # Order size ratio of BTC:ETH:XRP

        # Calculate risk weights from the volatility of each symbol
        volatilities = {symbol: latest_signal.loc[(slice(None), symbol), :].iloc[0]["volatility"]
                        for symbol in self.cfg["backtester_config"]["symbol"]}
        total_inv_vol = sum(1 / vol for vol in volatilities.values())
        risk_weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items()}

        for symbol in self.cfg["backtester_config"]["symbol"]:
            # Get the latest signal and OHLCV data for each symbol
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]

            # Get the current position size
            pos_size = asset_info.signed_pos_sizes[symbol]
            total_pos_abs = abs(pos_size)

            # Calculate target position size based on signal and discretization level
            signal_value = latest_signal_symbol['signal']
            if pd.isna(signal_value):
                signal_value = 0.0
            if signal_value > 0:
                target_position_size = math.floor(signal_value / d) * 0.5
            else:
                target_position_size = math.ceil(signal_value / d) * 0.5

            # Set annualized risk target according to target position size
            match target_position_size:
                case 1:
                    annualized_risk_target = 0.5
                case 0.5:
                    annualized_risk_target = 0.25
                case -0.5:
                    annualized_risk_target = -0.25
                case -1:
                    annualized_risk_target = -0.5
                case _:
                    annualized_risk_target = 0

            # Get the latest volatility for each symbol
            relevant_vola = latest_signal_symbol["volatility"]

            # Calculate target size considering risk weights and size ratios
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol] * risk_weights[symbol]
            order_size = target_size - pos_size
            side = "BUY" if order_size > 0 else "SELL"

            # Add order only if the minimum trading unit is met
            if abs(order_size) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(Order(type="MARKET",
                                      side=side,
                                      size=abs(order_size),
                                      price=None,
                                      symbol=symbol))

        return order_lst

class MinimumStrategy(BaseStrategy):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.symbols = cfg["backtester_config"]["symbol"]

    def preprocess(self, df):
        """
        preprocessing function for minimum strategy

        Parameters
        ==========
        df: pandas.DataFrame
            dataframe contains ohlcv data

        Returns
        ==========
        preprocessed_df: pandas.DataFrame
            preprocessed dataframe
        """
        ### preprocessing for get_orders
        # calculate annual risk from past 30 days price data
        span = 24 * 7 * 4  # 4 weeks
        df['volatility'] = np.nan

        # calculate volatility for each symbol
        for symbol in df.index.get_level_values('symbol').unique():
            # get data for each symbol
            symbol_df = df.xs(symbol, level='symbol')

            # calculate log return
            log_return = np.log(symbol_df['close']).diff()

            # calculate rolling std
            rolling_std = log_return.rolling(window=span).std()

            # calculate annualized volatility
            annualized_vola_symbol = rolling_std * np.sqrt(365.25 * 24)
            annualized_vola_symbol.fillna(1, inplace=True)

            # store result to original dataframe
            df.loc[(symbol_df.index, symbol), 'volatility'] = annualized_vola_symbol.values
        ### preprocessing for get_orders end

        preprocessed_df = df.copy()
        # add preprocessing

        return preprocessed_df

    def get_model(self, train_df: pd.DataFrame) -> Any:
        return None

    def get_signal(self, preprocessed_df: pd.DataFrame, model: Any) -> pd.DataFrame:
        return preprocessed_df.copy().assign(signal=1.0)

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        create orders based on the order time, the status of positions at that time, and the signal obtained from OHLCV

        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            current time (latest_timestamp)
        latest_bar : pandas.DataFrame
            OHLCV data (multi index of symbol and time)
        latest_signal : pandas.DataFrame
            latest signal for each symbol (multi index of symbol and time)
        asset_info : AssetInfo
            asset information (position, margin, etc.)

        Returns
        =======
        list[Order]
            list of orders for latest_timestamp
        """

        order_lst = []
        d = 0.35  # Discretization level

        # Calculate size ratio dynamically based on price ratios at the current time
        latest_prices = {
            symbol: latest_bar.loc[(slice(None), symbol), :].iloc[0]["close"]
            for symbol in self.cfg["backtester_config"]["symbol"]
        }
        inv_prices = {symbol: 1 / price for symbol, price in latest_prices.items()}
        total_inv_price = sum(inv_prices.values())
        size_ratio = {
            symbol: inv_price / total_inv_price
            for symbol, inv_price in inv_prices.items()
        }

        # Create orders for each symbol
        for symbol in self.cfg["backtester_config"]["symbol"]:
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]

            pos_size = asset_info.signed_pos_sizes[symbol]

            # Get signal value and discretize
            signal_value = latest_signal_symbol["signal"]
            if pd.isna(signal_value):
                signal_value = 0.0

            target_position_size = (
                math.floor(signal_value / d) * 0.5
                if signal_value > 0
                else math.ceil(signal_value / d) * 0.5
            )

            # Set annualized risk target
            match target_position_size:
                case 1:
                    annualized_risk_target = 0.5
                case 0.5:
                    annualized_risk_target = 0.25
                case -0.5:
                    annualized_risk_target = -0.25
                case -1:
                    annualized_risk_target = -0.5
                case _:
                    annualized_risk_target = 0

            # Get relevant volatility
            relevant_vola = latest_signal_symbol["volatility"]

            # Calculate target size considering dynamic size ratio
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol]
            order_size = target_size - pos_size
            side = "BUY" if order_size > 0 else "SELL"

            # Add order only if the minimum trading unit is met
            if abs(order_size) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(
                    Order(
                        type="MARKET",
                        side=side,
                        size=abs(order_size),
                        price=None,
                        symbol=symbol,
                    )
                )
        return order_lst

class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbol = "BTCUSDT"

    def preprocess(self, df):
        """
        Do nothing
        """
        return df

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        No preprocessing
        """
        return preprocessed_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Generate orders at the latest_timestamp.

        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            Current time (latest_timestamp).
        latest_bar : pandas.DataFrame
            Recent OHLCV (multi index of symbol and time).
        latest_signal : pandas.DataFrame
            Recent signal corresponding to latest_bar (symbol x time).
        asset_info : AssetInfo
            Asset information (position, margin, etc.).

        Returns
        =======
        list[Order]
            List of orders for latest_timestamp.
        """
        order_lst = []

        latest_bar_symbol = latest_bar.loc[(slice(None), self.symbol), :].iloc[0]

        current_position = asset_info.signed_pos_sizes[self.symbol]

        # Calculate the position size corresponding to the specified amount
        target_position = (
            self.cfg["strategy_config"]["sizing"]["target_size"]
            / latest_bar_symbol["close"]
        )

        # Only place orders if the current position is zero
        if current_position == 0:
            order_lst.append(
                Order(
                    type="MARKET",
                    side="BUY",
                    size=target_position,
                    price=None,
                    symbol=self.symbol,
                )
            )

        return order_lst

class KellyStrategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbol = "BTCUSDT"

    def preprocess(self, df):
        """
        Passthrough
        """
        return df

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        Passthrough
        """
        return preprocessed_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            Current time (latest_timestamp).
        latest_bar : pandas.DataFrame
            Recent OHLCV (multi index of symbol and time).
        latest_signal : pandas.DataFrame
            Recent signal corresponding to latest_bar (symbol x time).
        asset_info : AssetInfo
            Asset information (position, margin, etc.).

        Returns
        =======
        list[Order]
            List of orders for latest_timestamp.
        """


        order_lst = []

        latest_bar_symbol = latest_bar.loc[(slice(None), self.symbol), :].iloc[0]

        current_position = asset_info.signed_pos_sizes[self.symbol]
        target_position = (
            self.cfg["strategy_config"]["sizing"]["target_size"]
            / latest_bar_symbol["close"]
        )

        # Calculate the difference between the target position and the current position
        position_diff = target_position - current_position

        # Place an order to close the position difference if it exceeds the minimum trading unit
        if abs(position_diff) >= self.cfg["exchange_config"][self.symbol]["min_lot"]:
            order_lst.append(
                Order(
                    type="MARKET",
                    side="BUY" if position_diff > 0 else "SELL",
                    size=abs(position_diff),
                    price=None,
                    symbol=self.symbol,
                )
            )

        return order_lst

class VolatilityTargetingStrategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbol = "BTCUSDT"

    def preprocess(self, df):
        """
        Preprocess the input data.
        """
        df_symbol = df.loc[(slice(None), self.symbol), :].copy()

        # Calculate annualized volatility
        df.loc[(slice(None), self.symbol), "annualized_volatility"] = df_symbol[
            "close"
        ].pct_change().rolling(
            self.cfg["strategy_config"]["volatility"]["window"]
        ).std() * np.sqrt(self.cfg["strategy_config"]["volatility"]["annualizer"])

        return df

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        Get the signal from the preprocessed data.
        """
        return preprocessed_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Generate orders at the latest_timestamp.

        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            Current time (latest_timestamp).
        latest_bar : pandas.DataFrame
            Recent OHLCV (multi index of symbol and time).
        latest_signal : pandas.DataFrame
            Recent signal corresponding to latest_bar (symbol x time).
        asset_info : AssetInfo
            Asset information (position, margin, etc.).

        Returns
        =======
        list[Order]
            List of orders for latest_timestamp.
        """
        order_lst = []

        latest_bar_symbol = latest_bar.loc[(slice(None), self.symbol), :].iloc[0]
        latest_signal_symbol = latest_signal.loc[(slice(None), self.symbol), :].iloc[0]

        current_position = asset_info.signed_pos_sizes[self.symbol]

        # Volatility ratio for keeping constant risk amount
        volatility_ratio = (
            self.cfg["strategy_config"]["volatility"]["target_risk"]
            / latest_signal_symbol["annualized_volatility"]
        )

        # Target position size that maintains constant risk amount
        target_position = (
            self.cfg["strategy_config"]["sizing"]["target_size"]
            / latest_bar_symbol["close"]
            * volatility_ratio
        )

        # Difference between target position and current position
        position_diff = target_position - current_position

        if abs(position_diff) >= self.cfg["exchange_config"][self.symbol]["min_lot"]:
            order_lst.append(
                Order(
                    type="MARKET",
                    side="BUY" if position_diff > 0 else "SELL",
                    size=abs(position_diff),
                    price=None,
                    symbol=self.symbol,
                )
            )

        return order_lst

class RiskParityStrategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols = cfg["backtester_config"]["symbol"]

    def preprocess(self, df):
        """
        Preprocess the input data.
        """
        # Multiple symbols
        for symbol in self.symbols:
            df_symbol = df.loc[(slice(None), symbol), :].copy()
            df.loc[(slice(None), symbol), "annualized_volatility"] = df_symbol[
                "close"
            ].pct_change().rolling(
                self.cfg["strategy_config"]["volatility"]["window"]
            ).std() * np.sqrt(self.cfg["strategy_config"]["volatility"]["annualizer"])

        return df

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        Get the signal from the preprocessed data.
        """
        return preprocessed_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Generate orders at the latest_timestamp.

        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            Current time (latest_timestamp).
        latest_bar : pandas.DataFrame
            Recent OHLCV (multi index of symbol and time).
        latest_signal : pandas.DataFrame
            Recent signal corresponding to latest_bar (symbol x time).
        asset_info : AssetInfo
            Asset information (position, margin, etc.).

        Returns
        =======
        list[Order]
            List of orders for latest_timestamp.
        """
        order_lst = []
        # Risk Parity
        target_risk_per_symbol = self.cfg["strategy_config"]["volatility"][
            "target_risk"
        ] / len(self.symbols)

        # Expand to multiple stocks
        for symbol in self.symbols:
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]

            current_position = asset_info.signed_pos_sizes[symbol]
            volatility_ratio = (
                target_risk_per_symbol / latest_signal_symbol["annualized_volatility"]
            )
            target_position = (
                self.cfg["strategy_config"]["sizing"]["target_size"]
                / latest_bar_symbol["close"]
                * volatility_ratio
            )
            position_diff = target_position - current_position

            if abs(position_diff) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(
                    Order(
                        type="MARKET",
                        side="BUY" if position_diff > 0 else "SELL",
                        size=abs(position_diff),
                        price=None,
                        symbol=symbol,
                    )
                )

        return order_lst

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols = cfg["backtester_config"]["symbol"]

    def preprocess(self, df):
        """
        Preprocessing OHLCV data for trend following strategy.
        """
        for symbol in self.symbols:
            df_symbol = df.loc[(slice(None), symbol), :].copy()
            df.loc[(slice(None), symbol), "annualized_volatility"] = df_symbol[
                "close"
            ].pct_change().rolling(
                self.cfg["strategy_config"]["volatility"]["window"]
            ).std() * np.sqrt(self.cfg["strategy_config"]["volatility"]["annualizer"])

            # Calculate short moving average
            sma_short = (
                df_symbol["close"]
                .rolling(self.cfg["strategy_config"]["trend_follow"]["window"])
                .mean()
            )

            # Calculate long moving average
            sma_long = (
                df_symbol["close"]
                .rolling(self.cfg["strategy_config"]["trend_follow"]["window"] * 4)
                .mean()
            )

            # Price-dimensional volatility 
            price_volatility = (
                df_symbol["close"]
                .diff()
                .rolling(self.cfg["strategy_config"]["volatility"]["window"])
                .std()
            )

            # Calculate raw signal
            raw_signal = (sma_short - sma_long) / price_volatility

            # Calculate rolling min-max scaled signal
            rolling_max = raw_signal.rolling(
                self.cfg["strategy_config"]["trend_follow"]["scaling_window"]
            ).max()
            rolling_min = raw_signal.rolling(
                self.cfg["strategy_config"]["trend_follow"]["scaling_window"]
            ).min()
            scaled_signal = (raw_signal - rolling_min) / (rolling_max - rolling_min)

            # Convert the signal range from 0~1 to -1~1
            scaled_signal = scaled_signal * 2 - 1

            df.loc[(slice(None), symbol), "signal"] = scaled_signal

        return df

    def get_signal(self, preprocessed_df: pd.DataFrame, models: list):
        """
        Get the signal from the preprocessed_df 
        """
        return preprocessed_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        Generate orders at the latest_timestamp.

        Parameters
        ==========
        latest_timestamp : pandas.Timestamp
            Current time (latest_timestamp).
        latest_bar : pandas.DataFrame
            Recent OHLCV (multi index of symbol and time).
        latest_signal : pandas.DataFrame
            Recent signal corresponding to latest_bar (symbol x time).
        asset_info : AssetInfo
            Asset information (position, margin, etc.).

        Returns
        =======
        list[Order]
            List of orders for latest_timestamp.
        """


        order_lst = []

        target_risk_per_symbol = self.cfg["strategy_config"]["volatility"][
            "target_risk"
        ] / len(self.symbols)

        for symbol in self.symbols:
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]

            current_position = asset_info.signed_pos_sizes[symbol]

            volatility_ratio = (
                target_risk_per_symbol / latest_signal_symbol["annualized_volatility"]
            )

            # Calculate target position based on the magnitude and sign of the signal
            target_position = (
                self.cfg["strategy_config"]["sizing"]["target_size"]
                / latest_bar_symbol["close"]
                * volatility_ratio
                * latest_signal_symbol["signal"]
            )

            position_diff = target_position - current_position

            if abs(position_diff) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(
                    Order(
                        type="MARKET",
                        side="BUY" if position_diff > 0 else "SELL",
                        size=abs(position_diff),
                        price=None,
                        symbol=symbol,
                    )
                )

        return order_lst

