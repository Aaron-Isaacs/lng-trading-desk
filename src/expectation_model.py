import pandas as pd
# ------------------ EXPECTED RETURN BY REGIME ------------------
def expected_return(df, asset="HH", feature="HH_TTF_Z", bins=5, horizon=60):
    """
    Compute expected forward return conditioned on feature regime.
    """

    temp = df.copy()

    # forward return
    temp["fwd_return"] = temp[asset].pct_change(horizon).shift(-horizon)

    temp = temp.dropna(subset=[feature, "fwd_return"])

    # bucket feature
    temp["bucket"] = pd.qcut(temp[feature], bins, duplicates="drop")

    expected = temp.groupby("bucket")["fwd_return"].mean().reset_index()
    expected.columns = ["bucket", "expected_return"]

    return expected
# ------------------ MAP TRADE TO EXPECTATION ------------------
def trade_expectation(trade, expected_df):
    """
    Match a trade's regime to expected return.
    """

    z = trade.get("HH_TTF_Z", None)

    if z is None:
        return None

    # find bucket
    for _, row in expected_df.iterrows():
        b = row["bucket"]

        if b.left <= z <= b.right:
            return row["expected_return"]

    return None