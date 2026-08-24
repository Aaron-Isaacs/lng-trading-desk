import pandas as pd
import numpy as np

# ---------------------------
# Helper functions
# ---------------------------

def smooth_series(series, window=5):
    return series.rolling(window).mean()

def calc_score(z, mom, vol):
    """
    Converts Z-score, momentum, and volatility into a 0-100 score.
    """
    z_scaled = np.tanh(z)
    mom_scaled = np.tanh(mom * 5)
    vol_scaled = np.tanh(-vol * 5)
    score = 0.5 * z_scaled + 0.3 * mom_scaled + 0.2 * vol_scaled
    return int((score + 1) / 2 * 100)

def arb_to_int(margin):
    """
    Convert arbitrage margin into a signal integer.
    """
    if margin > 1:
        return 1   # favorable → long
    elif margin > 0:
        return 0   # hold
    else:
        return -1  # unfavorable → short

# ---------------------------
# Signal calculations
# ---------------------------

def generate_spread_signals(df):
    """
    Generate signals for predefined spreads (HH_TTF, TTF_JKM)
    """
    latest = df.iloc[-1]
    results = []

    for spread in ["HH_TTF", "TTF_JKM"]:
        score = calc_score(
            latest[f"{spread}_Z"],
            latest[f"{spread}_MOM"],
            latest[f"{spread}_VOL"]
        )

        signal = "HOLD"
        if score > 65:
            signal = "BUY"
        elif score < 35:
            signal = "SELL"

        results.append({
            "Spread": spread,
            "Score": score,
            "Signal": signal
        })

    return pd.DataFrame(results)

def composite_signal(df):
    """
    Composite signal incorporating spread scores and arbitrage influence.
    Returns dictionary with numeric score and string signal.
    """
    latest = df.iloc[-1]

    # base spread scores
    spread_scores = []
    for spread in ["HH_TTF", "TTF_JKM"]:
        z = smooth_series(df[f"{spread}"]).iloc[-1]
        mom = latest[f"{spread}_MOM"]
        vol = latest[f"{spread}_VOL"]
        score = 0.4 * np.tanh(z) + 0.3 * np.tanh(mom * 5) + 0.2 * np.tanh(-vol * 5)
        spread_scores.append(score)

    # arbitrage influence
    arb_score = 0.5 * np.tanh(latest.get("ARB_US_EU", 0)) + 0.5 * np.tanh(latest.get("ARB_US_ASIA", 0))

    # final composite score
    final_score = np.mean(spread_scores) + 0.5 * arb_score

    # convert to discrete signal
    signal = "HOLD"
    if final_score > 0.2:
        signal = "BUY"
    elif final_score < -0.2:
        signal = "SELL"

    return {"score": float(final_score), "signal": signal}

def compute_arbitrage_signal(df):
    """
    Compute arbitrage signals for US→Europe and US→Asia.
    """
    df = df.copy()
    df["US_Europe_Arb"] = df["TTF"] - df["HH"] - 3.5
    df["US_Asia_Arb"] = df["JKM"] - df["HH"] - 7.0

    latest = df.iloc[-1]
    return {
        "US→Europe": arb_to_int(latest["US_Europe_Arb"]),
        "US→Asia": arb_to_int(latest["US_Asia_Arb"])
    }

def compute_all_signals(df):
    """
    Unified function that computes spread signals, composite score, and arbitrage signals.
    Returns dictionary with numeric signal values for each category.
    """
    spread_df = generate_spread_signals(df)
    composite = composite_signal(df)
    arb = compute_arbitrage_signal(df)

    signals = {}

    # spread signals
    for row in spread_df.itertuples():
        sig = 1 if row.Signal == "BUY" else -1 if row.Signal == "SELL" else 0
        signals[row.Spread] = sig

    # composite signal
    sig = 1 if composite["signal"] == "BUY" else -1 if composite["signal"] == "SELL" else 0
    signals["Composite"] = sig

    # arbitrage signals
    for k, v in arb.items():
        signals[k] = v

    return signals