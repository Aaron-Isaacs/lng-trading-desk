# src/features.py

import numpy as np

def build_features(df):
    df = df.copy()

    # ---- SPREADS ----
    df["HH_TTF"] = df["HH"] - df["TTF"]
    df["TTF_JKM"] = df["TTF"] - df["JKM"]

    assets = ["HH", "TTF", "JKM", "HH_TTF", "TTF_JKM"]

    for col in assets:
        returns = df[col].pct_change()

        df[f"{col}_MOM"] = returns
        df[f"{col}_VOL"] = returns.rolling(20).std()

        rolling_mean = df[col].rolling(20).mean()
        rolling_std = df[col].rolling(20).std()

        df[f"{col}_Z"] = (df[col] - rolling_mean) / rolling_std

        # ---- LNG ARBITRAGE MODEL ----

        # Dynamic shipping costs
        shipping_eu = 2.0 + df["TTF"].pct_change().rolling(20).std() * 10
        shipping_asia = 3.5 + df["TTF"].pct_change().rolling(20).std() * 12

        shipping_eu = shipping_eu.fillna(2.0)
        shipping_asia = shipping_asia.fillna(3.5)

        # Liquefaction cost
        liq_cost = 2.0

        # Arbitrage margins
        df["ARB_US_EU"] = df["TTF"] - df["HH"] - liq_cost - shipping_eu
        df["ARB_US_ASIA"] = df["JKM"] - df["HH"] - liq_cost - shipping_asia

        # Signals
        df["ARB_EU_SIGNAL"] = np.where(df["ARB_US_EU"] > 1, 1,
                                       np.where(df["ARB_US_EU"] > 0, 0, -1))

        df["ARB_ASIA_SIGNAL"] = np.where(df["ARB_US_ASIA"] > 1, 1,
                                       np.where(df["ARB_US_ASIA"] > 0, 0, -1))

    # ==========================
    # COAL FEATURES
    # ==========================

    # Spread
    df["ARA_NEWC"] = df["ARA_COAL"] - df["NEWC_COAL"]

    coal_assets = [
        "ARA_COAL",
        "NEWC_COAL",
        "ARA_NEWC"
    ]

    for col in coal_assets:
        returns = df[col].pct_change()

        df[f"{col}_MOM"] = returns

        df[f"{col}_VOL"] = returns.rolling(20).std()

        rolling_mean = df[col].rolling(20).mean()
        rolling_std = df[col].rolling(20).std()

        df[f"{col}_Z"] = (
                                 df[col] - rolling_mean
                         ) / rolling_std

    df["COAL_TREND"] = np.where(
        df["ARA_COAL"] > df["ARA_COAL"].rolling(20).mean(),
        "UPTREND",
        "DOWNTREND"
    )

    return df.dropna()




