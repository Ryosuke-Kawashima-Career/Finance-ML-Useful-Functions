import math
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from mlbacktester import AssetInfo, BaseStrategy, Order

warnings.filterwarnings("ignore")

HOURS_PER_YEAR = 365.25 * 24


class Strategy(BaseStrategy):
    """
    Refined LightGBM signal strategy.

    Key design choices (vs. the original next-bar-return model):
      1. Longer, risk-adjusted label: HORIZON-bar forward return divided by
         local volatility -> far better signal-to-noise than 1-bar pct change.
      2. Compact, stationary feature set (vol-normalized momentum, RSI, BB%,
         volume z-score, funding/OI z-scores, cross-sectional market factor).
         Raw non-stationary columns (price, OI, funding) are never fed to the
         model -- the original accidentally included them.
      3. Seed-averaged LightGBM ensemble with bagging/regularization, tuned
         inside get_model on a time-ordered split with an embargo gap
         (no externally tuned, hard-coded parameters).
      4. Prediction scale taken from in-sample smoothed predictions, instead
         of a fragile 168-bar rolling std on the (short) test fold.
      5. EWM smoothing + hysteresis on the discrete signal: positions are
         entered on strong evidence and held until the score decays, which
         cuts turnover (and therefore slippage drag) dramatically.
    """

    HORIZON = 6        # forecast horizon, in bars (hours)
    SMOOTH_SPAN = 4    # EWM span used to smooth raw model predictions
    ENTRY_Z = 0.9      # |z| needed to open a half position
    FULL_Z = 1.8       # |z| needed to scale to a full position
    EXIT_Z = 0.3       # |z| below which an open position is closed
    SEEDS = [7, 42, 2023]

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.symbols = cfg["backtester_config"]["symbol"]

    # ------------------------------------------------------------------ #
    # Feature definitions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list:
        """Explicit whitelist so raw OHLCV / external columns never leak in."""
        base = [
            "mom_4", "mom_12", "mom_24", "mom_72", "mom_168",
            "rsi_14", "bb_20", "vol_z",
            "mkt_mom_24", "rel_mom_24",
            "funding_z", "oi_chg_24",
        ]
        return [c for c in base if c in df.columns]

    # ------------------------------------------------------------------ #
    # Preprocess (whole-period, backward-looking only; last row preserved)
    # ------------------------------------------------------------------ #
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        g = df.groupby("symbol", group_keys=False)

        # --- annualized volatility (required by get_orders for sizing) ---
        span = 24 * 7 * 4
        df["log_ret"] = g["close"].transform(lambda x: np.log(x).diff())
        df["volatility"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.rolling(span, min_periods=48).std() * np.sqrt(HOURS_PER_YEAR)
        )
        df["volatility"] = (
            df.groupby("symbol")["volatility"]
            .transform(lambda s: s.ffill())
            .fillna(1.0)
            .clip(lower=0.05)
        )
        vol_h = df["volatility"] / np.sqrt(HOURS_PER_YEAR)  # per-bar vol

        # --- vol-normalized momentum over several horizons ---
        for h in [4, 12, 24, 72, 168]:
            ret_h = g["close"].transform(lambda x, h=h: x.pct_change(h))
            df[f"mom_{h}"] = (ret_h / (vol_h * np.sqrt(h) + 1e-9)).clip(-5, 5)

        # --- RSI (centered to [-1, 1]) ---
        def _rsi(s, w=14):
            d = s.diff()
            up = d.clip(lower=0).rolling(w, min_periods=w // 2).mean()
            dn = (-d.clip(upper=0)).rolling(w, min_periods=w // 2).mean()
            rs = up / (dn + 1e-9)
            return ((100.0 - 100.0 / (1.0 + rs)) - 50.0) / 50.0

        df["rsi_14"] = g["close"].transform(_rsi)

        # --- Bollinger position (z relative to 2-sigma band) ---
        def _bb(s, w=20):
            m = s.rolling(w, min_periods=w // 2).mean()
            sd = s.rolling(w, min_periods=w // 2).std()
            return ((s - m) / (2.0 * sd + 1e-9)).clip(-3, 3)

        df["bb_20"] = g["close"].transform(_bb)

        # --- volume regime (z-score of log volume) ---
        def _volz(s, w=168):
            lv = np.log1p(s)
            m = lv.rolling(w, min_periods=24).mean()
            sd = lv.rolling(w, min_periods=24).std()
            return ((lv - m) / (sd + 1e-9)).clip(-5, 5)

        df["vol_z"] = g["volume"].transform(_volz)

        # --- external data, used only as stationary transforms ---
        if "funding_rate" in df.columns:
            def _fz(s, w=168):
                m = s.rolling(w, min_periods=24).mean()
                sd = s.rolling(w, min_periods=24).std()
                return ((s - m) / (sd + 1e-9)).clip(-5, 5)

            df["funding_z"] = g["funding_rate"].transform(_fz)

        if "open_interest" in df.columns:
            df["oi_chg_24"] = g["open_interest"].transform(
                lambda x: x.pct_change(24)
            ).clip(-1, 1)

        # --- contemporaneous cross-sectional market factor ---
        mkt = df.groupby(level=0)["mom_24"].mean()
        df["mkt_mom_24"] = df.index.get_level_values(0).map(mkt)
        df["rel_mom_24"] = df["mom_24"] - df["mkt_mom_24"]

        # Fill feature NaNs in place -- never drop rows (last row must stay).
        feats = self._feature_columns(df)
        df[feats] = df[feats].fillna(0.0)

        return df

    # ------------------------------------------------------------------ #
    # Model fitting (labels created here, not in preprocess)
    # ------------------------------------------------------------------ #
    def _make_regressor(self, lr: float, leaves: int, seed: int) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=lr,
            num_leaves=leaves,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )

    def get_model(self, train_df: pd.DataFrame):
        feats = self._feature_columns(train_df)
        payload = {"features": feats, "models": {}, "scale": {}}

        # Risk-adjusted forward return over HORIZON bars.
        vol_h = train_df["volatility"] / np.sqrt(HOURS_PER_YEAR)
        fwd = train_df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(self.HORIZON).shift(-self.HORIZON)
        )
        label = (fwd / (vol_h * np.sqrt(self.HORIZON) + 1e-9)).clip(-5, 5)

        for symbol in self.symbols:
            X_all = train_df.xs(symbol, level="symbol")[feats]
            y_all = label.xs(symbol, level="symbol")
            mask = y_all.notna() & np.isfinite(y_all)
            X, y = X_all[mask], y_all[mask]

            if len(X) < 200:
                payload["models"][symbol] = None
                payload["scale"][symbol] = 1.0
                continue

            # In-sample hyperparameter selection on a time-ordered split with
            # an embargo gap (overlapping labels), so nothing is hard-coded
            # from outside tuning.
            split = int(len(X) * 0.75)
            gap = self.HORIZON
            X_tr, y_tr = X.iloc[:split], y.iloc[:split]
            X_va, y_va = X.iloc[split + gap:], y.iloc[split + gap:]

            best_cfg, best_ic = (0.03, 7), -np.inf
            if len(X_va) > 50:
                for lr in [0.03, 0.07]:
                    for leaves in [7, 15]:
                        preds_va = np.zeros(len(X_va))
                        for seed in self.SEEDS:
                            m = self._make_regressor(lr, leaves, seed)
                            m.fit(X_tr, y_tr)
                            preds_va += m.predict(X_va)
                        preds_va /= len(self.SEEDS)
                        if len(np.unique(preds_va)) > 1:
                            ic = np.corrcoef(preds_va, y_va)[0, 1]
                        else:
                            ic = -1.0
                        if np.isfinite(ic) and ic > best_ic:
                            best_ic, best_cfg = ic, (lr, leaves)

            # Refit the chosen configuration on the full training fold.
            lr, leaves = best_cfg
            ensemble = []
            for seed in self.SEEDS:
                m = self._make_regressor(lr, leaves, seed)
                m.fit(X, y)
                ensemble.append(m)
            payload["models"][symbol] = ensemble

            # Robust prediction scale from smoothed in-sample predictions.
            tr_pred = np.mean([m.predict(X) for m in ensemble], axis=0)
            tr_pred = pd.Series(tr_pred, index=X.index).ewm(span=self.SMOOTH_SPAN).mean()
            scale = float(tr_pred.std())
            payload["scale"][symbol] = scale if scale > 1e-8 else 1.0

        return payload

    # ------------------------------------------------------------------ #
    # Signal generation
    # ------------------------------------------------------------------ #
    def _apply_hysteresis(self, z: np.ndarray) -> np.ndarray:
        """Causal state machine: enter on strong z, hold until z decays.

        Only uses information up to and including bar i, so it is valid in
        both backtest and forward-test modes.
        """
        out = np.zeros(len(z), dtype=float)
        cur = 0.0
        for i in range(len(z)):
            zi = z[i]
            if zi >= self.FULL_Z:
                cur = 1.0
            elif zi >= self.ENTRY_Z:
                cur = max(cur, 0.5)
            elif zi <= -self.FULL_Z:
                cur = -1.0
            elif zi <= -self.ENTRY_Z:
                cur = min(cur, -0.5)
            else:
                if cur > 0 and zi < self.EXIT_Z:
                    cur = 0.0
                elif cur < 0 and zi > -self.EXIT_Z:
                    cur = 0.0
            out[i] = cur
        return out

    def get_signal(self, preprocessed_df: pd.DataFrame, payload: dict) -> pd.DataFrame:
        df = preprocessed_df.copy()
        feats = payload["features"]

        out = []
        for symbol in self.symbols:
            sdf = df.xs(symbol, level="symbol").copy()
            models = payload["models"].get(symbol)

            if not models:
                sdf["signal"] = 0.0
            else:
                preds = np.mean([m.predict(sdf[feats]) for m in models], axis=0)
                z = (
                    pd.Series(preds, index=sdf.index)
                    .ewm(span=self.SMOOTH_SPAN)
                    .mean()
                    .fillna(0.0)
                    / payload["scale"][symbol]
                )
                sdf["signal"] = self._apply_hysteresis(z.to_numpy())

            sdf["symbol"] = symbol
            out.append(sdf)

        return pd.concat(out).set_index("symbol", append=True).sort_index()

    # ------------------------------------------------------------------ #
    # Orders -- organizer-provided implementation, kept verbatim
    # ------------------------------------------------------------------ #
    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        注文時刻，その時刻におけるポジションの状況，OHLCVから得たシグナルを元に注文を作成する関数
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