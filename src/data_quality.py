import pandas as pd
import numpy as np

def compute_data_reliability(df):
    """
    Returns:
        overall_score (0-100)
        per_asset_scores dict
    """

    scores = {}

    for col in ["HH", "TTF", "JKM"]:

        if col not in df.columns:
            scores[col] = 0
            continue

        series = df[col].dropna()

        if len(series) < 30:
            scores[col] = 0
            continue

        # ---- 1. Freshness ----
        last_date = series.index[-1]
        days_old = (pd.Timestamp.today() - last_date).days

        freshness_score = max(0, 100 - days_old * 20)

        # ---- 2. Missing data ----
        missing_ratio = series.isna().mean()
        missing_score = max(0, 100 - missing_ratio * 200)

        # ---- 3. Flatline detection ----
        recent_std = series.pct_change().tail(5).std()
        flat_score = 100 if recent_std > 0.001 else 20

        # ---- FINAL ----
        total = 0.5 * freshness_score + 0.3 * missing_score + 0.2 * flat_score
        scores[col] = np.clip(total, 0, 100)

    overall = np.mean(list(scores.values()))

    return overall, scores
