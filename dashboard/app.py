import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_market_data
from src.features import build_features
from src.signals import (generate_spread_signals, composite_signal, compute_arbitrage_signal)
from src.demo_trades import demo_trade_panel
from src.charts import plot_lines
from src.strategies import get_all_strategy_signals
from src.data_quality import compute_data_reliability
from src.analytics import (forward_returns, signal_outcomes, bucket_analysis, find_similar)
from src.trade_analytics import (prepare_trade_df, assign_z_bucket, performance_by_bucket, add_signal_follow_flag, performance_by_follow)
from src.expectation_model import expected_return, trade_expectation
from src.regimes import build_regime_map

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Mini LNG Desk", layout="wide")
st.sidebar.title("Mini LNG Desk")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ------------------ LOAD + PREP DATA ------------------
@st.cache_data(ttl=60)
def load_and_prepare():
    raw = load_market_data()
    features = build_features(raw)

    # compute spreads & z/mom/vol
    features["HH_TTF"] = features["HH"] - features["TTF"]
    features["TTF_JKM"] = features["TTF"] - features["JKM"]

    for col in ["HH", "TTF", "JKM", "HH_TTF", "TTF_JKM"]:
        features[f"{col}_Z"] = (features[col] - features[col].rolling(20).mean()) / features[col].rolling(20).std()
        features[f"{col}_MOM"] = features[col].pct_change()
        features[f"{col}_VOL"] = features[col].pct_change().rolling(20).std()

    # Arbitrage
    features["ARB_US_EU"] = features["TTF"] - features["HH"] - 3.5
    features["ARB_US_ASIA"] = features["JKM"] - features["HH"] - 7.0

    features = features.dropna()
    return features

hist_df = load_and_prepare()
hist_df = build_regime_map(hist_df)
data_score, asset_scores = compute_data_reliability(hist_df)

def recent_prices_table(df, rows=5):
    cols = ["HH", "TTF", "JKM", "HH_TTF", "TTF_JKM"]
    table = df[cols].tail(rows).copy()
    table = table.round(2)
    table = table.reset_index().rename(columns={"index": "Date"})
    return table

# ------------------ LIVE PRICES ------------------
def get_live_prices():
    return {col: hist_df[col].iloc[-1] for col in ["HH","TTF","JKM"]}

# ------------------ DASHBOARD ------------------
def dashboard_page():
    st.title("LNG Dashboard")

    # --- Data Quality ---
    st.subheader("Data Reliability")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall", f"{data_score:.1f}/100")
    col2.metric("HH", f"{asset_scores['HH']:.1f}")
    col3.metric("TTF", f"{asset_scores['TTF']:.1f}")
    col4.metric("JKM", f"{asset_scores['JKM']:.1f}")

    if data_score < 60:
        st.error("Data unreliable — trading should be avoided")
    elif data_score < 80:
        st.warning("Data quality moderate")
    else:
        st.success("Data quality strong")

    # --- Live Prices ---
    live_prices = get_live_prices()
    col1, col2, col3 = st.columns(3)
    col1.metric("HH", f"${live_prices['HH']:.2f}")
    col2.metric("TTF", f"${live_prices['TTF']:.2f}")
    col3.metric("JKM", f"${live_prices['JKM']:.2f}")

    # --- Charts ---
    st.subheader("Prices")
    st.plotly_chart(plot_lines(hist_df, ["HH","TTF","JKM"], "Prices"), use_container_width=True)
    st.subheader("Spreads")
    st.plotly_chart(plot_lines(hist_df, ["HH_TTF","TTF_JKM"], "Spreads"), use_container_width=True)
    st.subheader("Z-Scores")
    st.plotly_chart(plot_lines(hist_df, ["HH_TTF_Z","TTF_JKM_Z"], "Z-Scores"), use_container_width=True)
    st.subheader("Volatility")
    st.plotly_chart(plot_lines(hist_df, ["HH_TTF_VOL","TTF_JKM_VOL"], "Volatility"), use_container_width=True)
    st.subheader("Arbitrage Margins")
    st.plotly_chart(plot_lines(hist_df, ["ARB_US_EU","ARB_US_ASIA"], "Arbitrage Margins"), use_container_width=True)

    # --- Signals ---
    st.subheader("Strategy Scores")
    signals_df = get_all_strategy_signals(hist_df)
    st.dataframe(signals_df, use_container_width=True)

    st.subheader("Signals")
    spread_signals = generate_spread_signals(hist_df)
    st.dataframe(spread_signals, use_container_width=True)

    st.subheader("Composite Signal")
    comp = composite_signal(hist_df)
    st.metric("Score", f"{comp['score']:.2f}")
    st.metric("Signal", comp["signal"])

    st.subheader("LNG Arbitrage Signals")
    arb_sig = compute_arbitrage_signal(hist_df)
    for route, sig in arb_sig.items():
        st.metric(route, sig)

# ------------------ NAVIGATION ------------------
page = st.sidebar.selectbox("Navigation", ["Dashboard", "Demo Trades", "Analytics", "Coal"])

st.subheader("Recent LNG Prices")
st.dataframe(recent_prices_table(hist_df), use_container_width=True)

if page == "Dashboard":
    dashboard_page()
    st.subheader("Market Regimes")
    st.dataframe(
        hist_df[["VOL_REGIME", "TREND_REGIME", "SPREAD_REGIME", "MARKET_STATE"]].tail(10)
    )
elif page == "Demo Trades":
    st.title("Demo Trades")
    if data_score < 60:
        st.error("Trading disabled due to poor data quality")
    else:
        demo_trade_panel(hist_df, get_live_prices())
elif page == "Analytics":
    st.title("Market Learning Analytics")

    asset = st.selectbox("Asset", ["HH", "TTF", "JKM"])
    horizon = st.slider("Forward Horizon (days)", 10, 120, 60)

    st.subheader("Forward Returns Distribution")

    fwd = forward_returns(hist_df, asset, horizon)
    st.line_chart(fwd.cumsum())

    # ---------------- SIGNAL OUTCOMES ----------------
    st.subheader("Signal Outcomes")

    signals = hist_df["HH_TTF_Z"]  # example signal proxy
    outcomes = signal_outcomes(hist_df, signals, asset, horizon)

    st.write("Average return by signal sign:")
    outcomes["signal_sign"] = outcomes["signal"].apply(lambda x: 1 if x > 0 else -1)
    st.dataframe(outcomes.groupby("signal_sign")["fwd_return"].mean())

    # ---------------- BUCKET ANALYSIS ----------------
    st.subheader("Z-Score Bucket Analysis")

    bucket_df = bucket_analysis(hist_df, "HH_TTF_Z", asset, horizon)
    st.dataframe(bucket_df)

    # ---------------- SIMILAR REGIMES ----------------
    st.subheader("Similar Historical Regimes")

    current = hist_df.iloc[-1]
    cols = ["HH_TTF_Z", "TTF_JKM_Z", "HH_VOL"]

    similar = find_similar(hist_df, cols, current, top_n=20)

    st.write("Similar periods:")
    st.dataframe(similar.tail(10))

    sim_fwd = forward_returns(similar, asset, horizon)
    st.line_chart(sim_fwd.cumsum())

    # ---------------- TRADE ANALYTICS ----------------
    st.subheader("Your Trade Behaviour")

    from src.demo_trades import load_journal
    journal_df = load_journal()

    trade_df = prepare_trade_df(journal_df)

    if trade_df.empty:
        st.info("No trades yet")
    else:
        # ---- Bucket analysis ----
        trade_df = assign_z_bucket(trade_df)

        st.write("Performance by Z-Score at Entry")
        st.dataframe(performance_by_bucket(trade_df))

        st.subheader("Performance by Market Regime")

        journal_df = load_journal()

        if not journal_df.empty:
            merged = journal_df.copy()

            # approximate join by time (simple version)
            merged["Timestamp"] = pd.to_datetime(merged["Timestamp"])
            hist_df_indexed = hist_df.copy()
            hist_df_indexed["Timestamp"] = hist_df_indexed.index

            merged = pd.merge_asof(
                merged.sort_values("Timestamp"),
                hist_df_indexed[["Timestamp", "MARKET_STATE"]].sort_values("Timestamp"),
                on="Timestamp"
            )

            regime_perf = merged.groupby("MARKET_STATE")["PnL"].agg(
                Trades="count",
                Avg_PnL="mean",
                Total_PnL="sum"
            ).reset_index()

            st.dataframe(regime_perf)

        # ---- Followed signal ----
        trade_df = add_signal_follow_flag(trade_df)

        st.write("Signal Followed?")
        st.dataframe(performance_by_follow(trade_df))

        st.subheader("Expected vs Actual Trade Performance")

        journal_df = load_journal()

        if not journal_df.empty:
            # compute expected returns from market history
            exp = expected_return(hist_df, asset="HH", feature="HH_TTF_Z")

            journal_df["Expected"] = journal_df.apply(
                lambda row: trade_expectation(row, exp),
                axis=1
            )

            journal_df["Deviation"] = journal_df["PnL"] - journal_df["Expected"]

            st.dataframe(journal_df[[
                "Asset", "Direction", "PnL", "Expected", "Deviation"
            ]])

            st.write("Average Deviation:")
            st.metric(
                "Skill Score (PnL - Expected)",
                round(journal_df["Deviation"].mean(), 4)
            )

            st.subheader("Improvement")

            from src.demo_trades import load_journal

            journal_df = load_journal()

            if not journal_df.empty:
                from src.trade_analytics import learning_curve

                curve = learning_curve(journal_df)

                st.line_chart(curve.set_index("Timestamp")[["Rolling_Skill"]])
elif page == "Coal":
    def coal_page():

        st.title("Coal Desk")

        latest = hist_df.iloc[-1]

        # ---------------- Prices ----------------
        st.subheader("Coal Prices")

        col1, col2 = st.columns(2)

        col1.metric(
            "ARA Coal",
            f"${latest['ARA_COAL']:.2f}"
        )

        col2.metric(
            "Newcastle Coal",
            f"${latest['NEWC_COAL']:.2f}"
        )

        st.plotly_chart(
            plot_lines(
                hist_df,
                ["ARA_COAL", "NEWC_COAL"],
                "Coal Prices"
            ),
            use_container_width=True
        )

        # ---------------- Spread ----------------
        st.subheader("Coal Spread")

        st.metric(
            "ARA - Newcastle",
            f"{latest['ARA_NEWC']:.2f}"
        )

        st.plotly_chart(
            plot_lines(
                hist_df,
                ["ARA_NEWC"],
                "Coal Spread"
            ),
            use_container_width=True
        )

        # ---------------- Z Score ----------------
        st.subheader("Spread Z-Score")

        st.metric(
            "Current Z",
            f"{latest['ARA_NEWC_Z']:.2f}"
        )

        st.plotly_chart(
            plot_lines(
                hist_df,
                ["ARA_NEWC_Z"],
                "Spread Z-Score"
            ),
            use_container_width=True
        )

        # ---------------- Volatility ----------------
        st.subheader("Spread Volatility")

        st.metric(
            "Volatility",
            f"{latest['ARA_NEWC_VOL']:.4f}"
        )

        st.plotly_chart(
            plot_lines(
                hist_df,
                ["ARA_NEWC_VOL"],
                "Spread Volatility"
            ),
            use_container_width=True
        )

        # ---------------- Trend ----------------
        st.subheader("Trend")

        st.metric(
            "Current Trend",
            latest["COAL_TREND"]
        )
    coal_page()











