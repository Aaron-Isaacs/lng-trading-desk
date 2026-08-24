# src/data.py

import yfinance as yf
import pandas as pd
import numpy as np

def load_market_data(period="6mo"):

    assets = {
        "HH": "NG=F",
        "TTF_raw": "TTF=F",
        "BRENT": "BZ=F",
        "EURUSD": "EURUSD=X",
    }

    data = {}

    for name, ticker in assets.items():

        temp = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False
        )

        if temp.empty or "Close" not in temp.columns:
            raise ValueError(f"{name} failed to load")

        data[name] = temp["Close"]

    df = pd.concat(data.values(), axis=1)
    df.columns = data.keys()

    df = df.ffill().bfill()

    # ---------------- TTF ----------------

    df["TTF"] = df["TTF_raw"] * df["EURUSD"] / 3.412

    # ---------------- JKM ----------------

    ttf = df["TTF"]

    base_shipping = 2.5

    vol_premium = ttf.pct_change().rolling(20).std() * 20
    vol_premium = vol_premium.fillna(0)

    df["JKM"] = ttf + base_shipping + vol_premium

    # ===================================================
    # COAL (proxy implementation)
    # ===================================================

    # Approximate ARA coal using Brent relationship
    df["ARA_COAL"] = (
        df["BRENT"] * 1.15
        + np.random.normal(0, 2, len(df))
    )

    # Approximate Newcastle as ARA plus export premium
    df["NEWC_COAL"] = (
        df["ARA_COAL"]
        + 8
        + np.random.normal(0, 1, len(df))
    )

    return df[[
        "HH",
        "TTF",
        "JKM",
        "ARA_COAL",
        "NEWC_COAL"
    ]]