import pandas as pd
import numpy as np
# ------------------ FORWARD RETURNS ------------------
def forward_returns(df, asset="HH", horizon=60):
    """
    Compute forward returns over a horizon (e.g. 60 days).
    """
    fwd = df[asset].pct_change(periods=horizon).shift(-horizon)
    return fwd
# ------------------ SIGNAL OUTCOME ANALYSIS ------------------
def signal_outcomes(df, signal_series, asset="HH", horizon=60):
    """
    Align signals with future returns to evaluate outcomes.
    """
    fwd = forward_returns(df, asset, horizon)

    aligned = pd.DataFrame({
        "signal": signal_series,
        "fwd_return": fwd
    }).dropna()

    return aligned
# ------------------ FEATURE BUCKET ANALYSIS ------------------
def bucket_analysis(df, feature_col, asset="HH", horizon=60, bins=5):
    """
    Group forward returns by feature buckets (e.g. Z-score buckets).
    """
    fwd = forward_returns(df, asset, horizon)

    temp = pd.DataFrame({
        "feature": df[feature_col],
        "fwd_return": fwd
    }).dropna()

    temp["bucket"] = pd.qcut(temp["feature"], bins, duplicates="drop")

    temp["bucket"] = temp["bucket"].apply(
        lambda x: f"{round(x.left, 2)} to {round(x.right, 2)}"
    )

    return temp.groupby("bucket")["fwd_return"].mean().reset_index()
# ------------------ SIMILAR REGIME SEARCH ------------------
def find_similar(df, cols, current, top_n=20):
    df = df.copy()

    # --- FORCE NUMERIC MATRIX ---
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    current = pd.Series(current)

    # align and coerce
    current = pd.to_numeric(current, errors="coerce")

    # drop invalid rows
    X = X.dropna()
    current = current.reindex(X.columns)

    # compute euclidean distance
    distances = np.sqrt(((X - current) ** 2).sum(axis=1))

    distances = pd.Series(distances, index=X.index)
    distances = distances.dropna()

    return df.loc[distances.nsmallest(top_n).index]








