# src/strategies.py

import numpy as np
import pandas as pd

# ------------------ MEAN REVERSION ------------------

def mean_reversion_signal(z_score, entry=2.0):
    """
    Mean reversion:
    - Short when Z is high
    - Long when Z is low
    """
    if pd.isna(z_score):
        return None

    if z_score > entry:
        return {
            "signal": "SHORT",
            "conviction": min(abs(z_score) / 3, 1),
            "reason": f"Z-score {z_score:.2f} > {entry}"
        }

    elif z_score < -entry:
        return {
            "signal": "LONG",
            "conviction": min(abs(z_score) / 3, 1),
            "reason": f"Z-score {z_score:.2f} < -{entry}"
        }

    return None


# ------------------ TREND ------------------

def trend_signal(returns, vol):
    """
    Trend:
    - Long if returns positive
    - Short if negative
    - Scaled by volatility
    """
    if pd.isna(returns) or pd.isna(vol) or vol == 0:
        return None

    direction = "LONG" if returns > 0 else "SHORT"
    conviction = min(abs(returns) / (vol + 1e-6), 1)

    return {
        "signal": direction,
        "conviction": conviction,
        "reason": f"Return {returns:.4f}, Vol {vol:.4f}"
    }


# ------------------ MASTER WRAPPER ------------------

def get_all_strategy_signals(df):
    """
    Returns a dataframe of all strategy signals for latest timestep.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    latest = df.iloc[-1]

    results = []

    assets = ["HH", "TTF", "JKM", "HH_TTF", "TTF_JKM"]

    for asset in assets:
        z = latest.get(f"{asset}_Z", np.nan)
        mom = latest.get(f"{asset}_MOM", np.nan)
        vol = latest.get(f"{asset}_VOL", np.nan)

        # ---- Mean Reversion ----
        mr = mean_reversion_signal(z)

        if mr:
            results.append({
                "Asset": asset,
                "Strategy": "Mean Reversion",
                "Signal": mr["signal"],
                "Conviction": round(mr["conviction"], 2),
                "Reason": mr["reason"]
            })

        # ---- Trend ----
        tr = trend_signal(mom, vol)

        if tr:
            results.append({
                "Asset": asset,
                "Strategy": "Trend",
                "Signal": tr["signal"],
                "Conviction": round(tr["conviction"], 2),
                "Reason": tr["reason"]
            })

    return pd.DataFrame(results)