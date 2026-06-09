import numpy as np
import pandas as pd
import math
import warnings
import lightgbm as lgb
from mlbacktester import AssetInfo, BaseStrategy, Order

# Suppress warnings
warnings.filterwarnings('ignore')

class Strategy(BaseStrategy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols = cfg["backtester_config"]["symbol"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Essential volatility for get_orders
        span = 24 * 7 * 4
        df['log_ret'] = df.groupby('symbol')['close'].transform(lambda x: np.log(x).diff())
        df['volatility'] = df.groupby('symbol')['log_ret'].transform(
            lambda x: x.rolling(window=span).std() * np.sqrt(365.25 * 24)
        ).fillna(1.0)
        
        # 2. Feature Engineering
        groupby_obj = df.groupby('symbol')
        close = df['close']
        
        # RSI with multiple windows
        for w in [14, 28]:
            def get_rsi(s):
                delta = s.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=w, min_periods=1).mean()
                avg_loss = loss.rolling(window=w, min_periods=1).mean()
                rs = avg_gain / (avg_loss + 1e-9)
                rsi = 100 - (100 / (1 + rs))
                return (rsi - 50) / 50
            df[f'RSI_{w}'] = groupby_obj['close'].transform(get_rsi)
            
        # MACD
        def get_macd_hist(s):
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            return (macd - macd_signal) / (s + 1e-9)
        df['MACD_hist'] = groupby_obj['close'].transform(get_macd_hist)
        
        # Bollinger Bands %B
        for w in [20, 50]:
            def get_bbpct(s):
                sma = s.rolling(window=w).mean()
                std = s.rolling(window=w).std()
                upper = sma + 2 * std
                lower = sma - 2 * std
                return (s - lower) / (upper - lower + 1e-9)
            df[f'BB_pct_{w}'] = groupby_obj['close'].transform(get_bbpct).fillna(0.5)
            
        # Z-score returns
        for w in [24, 168]:
            def get_zscore(s):
                roll_mean = s.rolling(window=w).mean()
                roll_std = s.rolling(window=w).std()
                return (s - roll_mean) / (roll_std + 1e-9)
            df[f'Zscore_{w}'] = groupby_obj['log_ret'].transform(get_zscore).fillna(0.0)
            
        # ROC
        for w in [12, 24]:
            df[f'ROC_{w}'] = groupby_obj['close'].transform(lambda x: x.pct_change(periods=w)).fillna(0.0)
            
        # Volatility target/ratio
        for w in [24, 168]:
            def get_vol_ratio(s):
                roll_mean = s.rolling(window=w).mean()
                return s / (roll_mean + 1e-9)
            df[f'Vol_ratio_{w}'] = groupby_obj['volume'].transform(get_vol_ratio).fillna(1.0)
            
        # External data: Funding rates & Open Interest if available
        if 'funding_rate' in df.columns:
            for w in [24, 168]:
                def get_funding_z(s):
                    roll_mean = s.rolling(window=w).mean()
                    roll_std = s.rolling(window=w).std()
                    return (s - roll_mean) / (roll_std + 1e-9)
                df[f'Funding_Z_{w}'] = groupby_obj['funding_rate'].transform(get_funding_z).fillna(0.0)
                
        if 'open_interest' in df.columns:
            for w in [24]:
                df[f'OI_change_{w}'] = groupby_obj['open_interest'].transform(lambda x: x.pct_change(periods=w)).fillna(0.0)
                
        # Fill any NaNs
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'log_ret']]
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
        
        return df

    def get_model(self, train_df: pd.DataFrame):
        models = {}
        
        # Identify feature columns
        feature_cols = [c for c in train_df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'log_ret', 'volatility']]
        
        # Target: next-bar percentage change return
        target = train_df.groupby('symbol')['close'].transform(lambda x: x.pct_change().shift(-1))
        
        for symbol in self.symbols:
            symbol_df = train_df.xs(symbol, level='symbol')
            symbol_target = target.xs(symbol, level='symbol')
            
            data = symbol_df[feature_cols].copy()
            data['target'] = symbol_target
            data = data.dropna()
            
            X = data[feature_cols]
            y = data['target']
            
            # Simple hyperparameter optimization split to comply with the prohibition of hardcoding
            split_idx = int(len(X) * 0.8)
            X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            best_score = -np.inf
            best_model = None
            
            for lr in [0.01, 0.05]:
                for depth in [3, 4]:
                    model = lgb.LGBMRegressor(
                        n_estimators=30,
                        learning_rate=lr,
                        max_depth=depth,
                        num_leaves=2**depth - 1,
                        random_state=42,
                        n_jobs=1,
                        verbosity=-1
                    )
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    
                    # Compute Pearson correlation (Information Coefficient)
                    corr = np.corrcoef(preds, y_val)[0, 1] if len(np.unique(preds)) > 1 else -1.0
                    if pd.isna(corr):
                        corr = -1.0
                    if corr > best_score:
                        best_score = corr
                        best_model = model
            
            if best_model is None:
                best_model = lgb.LGBMRegressor(n_estimators=30, random_state=42, n_jobs=1, verbosity=-1)
                best_model.fit(X, y)
            else:
                # Retrain best configuration on the full training set
                best_model.fit(X, y)
                
            models[symbol] = best_model
            
        return models

    def get_signal(self, preprocessed_df: pd.DataFrame, models: dict) -> pd.DataFrame:
        df = preprocessed_df.copy()
        df['signal'] = 0.0
        
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'log_ret', 'volatility', 'signal']]
        
        dfs = []
        for symbol in self.symbols:
            symbol_df = df.xs(symbol, level='symbol').copy()
            model = models[symbol]
            
            preds = model.predict(symbol_df[feature_cols])
            preds_series = pd.Series(preds, index=symbol_df.index)
            
            # Use rolling standard deviation of predictions to scale
            rolling_std = preds_series.rolling(window=168, min_periods=24).std().fillna(1e-4)
            z_preds = preds_series / rolling_std
            
            # Map Z-scores to signals based on step thresholds (0.35 step used in get_orders)
            signal = pd.Series(0.0, index=symbol_df.index)
            signal[z_preds > 1.25] = 0.5
            signal[z_preds > 2.0] = 1.0
            signal[z_preds < -1.25] = -0.5
            signal[z_preds < -2.0] = -1.0
            
            symbol_df['signal'] = signal
            symbol_df['symbol'] = symbol
            dfs.append(symbol_df)
            
        return pd.concat(dfs).set_index('symbol', append=True).sort_index()

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        注文時刻，その時刻におけるポジションの状況，OHLCVから得たシグナルを元に注文を作成する関数

        Parameters
        ==========
        latest_timestamp: pandas.Timestamp
            注文を出す時刻
        latest_bar: pandas.Series
            注文を出す時刻のOHLCVデータ(加工前のデータ)
        latest_signal: pandas.Series
            注文を出す時刻のシグナルデータ(get_signal関数により作成されたデータ)
        asset_info: dict
            注文時における資産の情報が格納された辞書

        Returns
        =======
        order_lst: list (中身はOrderクラス)
            current_timeにおける注文情報が格納されている
            'type','side','size','price'の４項目
        """

        order_lst = []
        d = 0.35  # 離散化の程度
        size_ratio = {"BTCUSDT": 0.1, "ETHUSDT": 1.5, "XRPUSDT": 4000}  # BTC:ETH:XRP の注文サイズ比

        # 各シンボルのボラティリティからリスクウェイトを計算
        volatilities = {symbol: latest_signal.loc[(slice(None), symbol), :].iloc[0]["volatility"]
                        for symbol in self.cfg["backtester_config"]["symbol"]}
        total_inv_vol = sum(1 / vol for vol in volatilities.values())
        risk_weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items()}

        for symbol in self.cfg["backtester_config"]["symbol"]:
            # シンボルごとの最新のシグナルとOHLCVデータを取得
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]

            # 現在のポジションサイズを取得
            pos_size = asset_info.signed_pos_sizes[symbol]
            total_pos_abs = abs(pos_size)

            # シグナルと離散化の程度を基に目標ポジションサイズを計算
            signal_value = latest_signal_symbol['signal']
            if pd.isna(signal_value):
                signal_value = 0.0
            if signal_value > 0:
                target_position_size = math.floor(signal_value / d) * 0.5
            else:
                target_position_size = math.ceil(signal_value / d) * 0.5

            # 目標ポジションサイズに応じて年率リスクターゲットを設定
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

            # シンボルごとの最新ボラティリティを取得
            relevant_vola = latest_signal_symbol["volatility"]

            # リスクウェイトとサイズ比を考慮して目標サイズを計算
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol] * risk_weights[symbol]
            order_size = target_size - pos_size
            side = "BUY" if order_size > 0 else "SELL"

            # 最小取引単位を満たす場合のみ注文を追加
            if abs(order_size) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(Order(type="MARKET",
                                      side=side,
                                      size=abs(order_size),
                                      price=None,
                                      symbol=symbol))

        return order_lst
