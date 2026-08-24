import pandas as pd
import numpy as np
# ------------------ VOLATILITY REGIME ------------------
def volatility_regime(df, asset="HH", window=20):
    vol = df[asset].pct_change().rolling(window).std()

    threshold = vol.rolling(window * 3).mean()

    regime = np.where(vol > threshold, "HIGH_VOL", "LOW_VOL")

    return pd.Series(regime, index=df.index)
# ------------------ TREND REGIME ------------------
def trend_regime(df, asset="HH", window=20):
    mom = df[asset].pct_change(window)

    regime = np.where(mom > 0, "UPTREND", "DOWNTREND")

    return pd.Series(regime, index=df.index)
# ------------------ SPREAD REGIME ------------------
def spread_regime(df):
    spread = df["HH_TTF_Z"]

    regime = np.where(
        spread > 1.5, "EXPENSIVE_US_GAS",
        np.where(spread < -1.5, "CHEAP_US_GAS", "NEUTRAL")
    )

    return pd.Series(regime, index=df.index)
# ------------------ MASTER REGIME MAP ------------------
def build_regime_map(df):
    df = df.copy()

    df["VOL_REGIME"] = volatility_regime(df)
    df["TREND_REGIME"] = trend_regime(df)
    df["SPREAD_REGIME"] = spread_regime(df)

    # combined state label (very important)
    df["MARKET_STATE"] = (
        df["VOL_REGIME"] + "_" +
        df["TREND_REGIME"] + "_" +
        df["SPREAD_REGIME"]
    )

    return df