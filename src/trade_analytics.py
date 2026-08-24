import pandas as pd
# ------------------ LOAD + PREP ------------------
def prepare_trade_df(journal_df):
    df = journal_df.copy()

    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Drop broken rows
    df = df.dropna(subset=["Timestamp"])

    return df
# ------------------ Z-SCORE BUCKETING ------------------
def assign_z_bucket(df, col="HH_TTF_Z", bins=5):
    df = df.copy()

    if col not in df.columns:
        return df

    try:
        df["Z_bucket"] = pd.qcut(df[col], bins, duplicates="drop")
        df["Z_bucket"] = df["Z_bucket"].apply(
            lambda x: f"{round(x.left,2)} to {round(x.right,2)}"
        )
    except Exception:
        df["Z_bucket"] = "Unknown"

    return df
# ------------------ BASIC PERFORMANCE BY BUCKET ------------------
def performance_by_bucket(df):
    if df.empty or "Z_bucket" not in df.columns:
        return pd.DataFrame()

    return df.groupby("Z_bucket")["PnL"].agg(
        Trades="count",
        Avg_PnL="mean",
        Total_PnL="sum"
    ).reset_index()
# ------------------ FOLLOWED SIGNAL ANALYSIS ------------------
def add_signal_follow_flag(df):
    df = df.copy()

    if "HH_TTF_Z" not in df.columns:
        df["Followed"] = "Unknown"
        return df

    def classify(row):
        z = row.get("HH_TTF_Z", None)
        direction = row.get("Direction", "")

        if pd.isna(z):
            return "Unknown"

        if z < -1.5 and direction == "Long":
            return "Yes"
        if z > 1.5 and direction == "Short":
            return "Yes"

        return "No"

    df["Followed"] = df.apply(classify, axis=1)

    return df

def performance_by_follow(df):
    if df.empty or "Followed" not in df.columns:
        return pd.DataFrame()

    return df.groupby("Followed")["PnL"].agg(
        Trades="count",
        Avg_PnL="mean",
        Total_PnL="sum"
    ).reset_index()

def learning_curve(df):
    """
    Tracks whether your decision quality improves over time.
    """

    if df.empty:
        return df

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp")

    # rolling mean of performance vs expectation
    window = 5

    df["Rolling_PnL"] = df["PnL"].rolling(window).mean()

    if "Expected" in df.columns:
        df["Skill"] = df["PnL"] - df["Expected"]
        df["Rolling_Skill"] = df["Skill"].rolling(window).mean()
    else:
        df["Skill"] = df["PnL"]
        df["Rolling_Skill"] = df["Rolling_PnL"]

    return df