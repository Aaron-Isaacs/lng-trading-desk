import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px

# ------------------ DATA PATHS ------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

JOURNAL_FILE = DATA_DIR / "trade_journal.csv"
OPEN_TRADES_FILE = DATA_DIR / "open_trades.csv"

# ------------------ INIT FILES ------------------
if not JOURNAL_FILE.exists() or JOURNAL_FILE.stat().st_size == 0:
    pd.DataFrame(columns=[
        "Timestamp", "Asset", "Direction",
        "Entry Price", "Exit Price", "PnL", "Notes", "Strategy", "HH", "TTF", "JKM", "HH_TTF_Z", "TTF_JKM_Z", "ARB_US_EU", "ARB_US_AISA"
    ]).to_csv(JOURNAL_FILE, index=False)

if not OPEN_TRADES_FILE.exists() or OPEN_TRADES_FILE.stat().st_size == 0:
    pd.DataFrame(columns=[
        "Asset", "Direction", "Qty", "Price",
        "Stop Loss", "Take Profit", "PnL", "Notes", "Opened At", "Strategy", "HH", "TTF", "JKM", "HH_TTF_Z", "TTF_JKM_Z", "ARB_US_EU", "ARB_US_AISA"
    ]).to_csv(OPEN_TRADES_FILE, index=False)

# ------------------ HELPERS ------------------
def load_journal():
    return pd.read_csv(JOURNAL_FILE)

def save_journal(df):
    df.to_csv(JOURNAL_FILE, index=False)

def load_open_trades():
    # If file is missing or empty, write headers
    required_cols = [
        "Asset", "Direction", "Qty", "Price",
        "Stop Loss", "Take Profit", "PnL", "Notes", "Opened At", "Strategy"
    ]
    if not OPEN_TRADES_FILE.exists() or OPEN_TRADES_FILE.stat().st_size == 0:
        pd.DataFrame(columns=required_cols).to_csv(OPEN_TRADES_FILE, index=False)

    # Try reading the file
    try:
        df = pd.read_csv(OPEN_TRADES_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=required_cols)
        df.to_csv(OPEN_TRADES_FILE, index=False)

    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    return df.to_dict("records")

def save_open_trades(trades):
    df = pd.DataFrame(trades)
    required_cols = [
        "Asset", "Direction", "Qty", "Price",
        "Stop Loss", "Take Profit", "PnL", "Notes", "Opened At", "Strategy"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df.to_csv(OPEN_TRADES_FILE, index=False)

def compute_pnl(entry, current, direction, qty):
    return (current - entry) * qty if direction == "Long" else (entry - current) * qty

# ------------------ PANEL ------------------
def demo_trade_panel(hist_df, live_prices):
    st.subheader("Demo Trades")

    # ---- LOAD STATE FROM CSV ----
    if "trades" not in st.session_state:
        st.session_state.trades = load_open_trades()

    # ---- ADD TRADE FORM ----
    with st.form("add_trade"):
        strategy = st.selectbox('Strategy', ['Trend', 'Mean Reversion'])
        asset = st.selectbox("Asset", ["HH", "TTF", "JKM", "HH_TTF", "TTF_JKM"])
        direction = st.selectbox("Direction", ["Long", "Short"])
        qty = st.number_input("Quantity", min_value=1, value=1)

        price = st.number_input(
            "Entry Price",
            value=float(live_prices.get(asset.split("_")[0], 0.0))
        )
        stop_loss = st.number_input("Stop Loss", value=0.0)
        take_profit = st.number_input("Take Profit", value=0.0)
        notes = st.text_area("Notes")

        if st.form_submit_button("Add Trade"):
            latest = hist_df.iloc[-1]

            trade = {
                "Strategy": strategy,
                "Asset": asset,
                "Direction": direction,
                "Qty": qty,
                "Price": price,
                "Stop Loss": stop_loss,
                "Take Profit": take_profit,
                "PnL": 0.0,
                "Notes": notes,
                "Opened At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

                # --- CONTEXT (THIS IS THE KEY CHANGE) ---
                "HH": latest.get("HH"),
                "TTF": latest.get("TTF"),
                "JKM": latest.get("JKM"),

                "HH_TTF_Z": latest.get("HH_TTF_Z"),
                "TTF_JKM_Z": latest.get("TTF_JKM_Z"),

                "ARB_US_EU": latest.get("ARB_US_EU"),
                "ARB_US_ASIA": latest.get("ARB_US_ASIA"),
            }

            st.session_state.trades.append(trade)
            save_open_trades(st.session_state.trades)
            st.success("Trade added!")

    # ---- UPDATE PnL + AUTO CLOSE ----
    trades_to_close = []
    for trade in st.session_state.trades:
        live_price = live_prices.get(trade["Asset"].split("_")[0], trade["Price"])
        trade["PnL"] = compute_pnl(trade["Price"], live_price, trade["Direction"], trade["Qty"])

        if (trade["Take Profit"] > 0 and trade["PnL"] >= trade["Take Profit"]) or \
           (trade["Stop Loss"] > 0 and trade["PnL"] <= -trade["Stop Loss"]):

            journal_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Asset": trade["Asset"],
                "Direction": trade["Direction"],
                "Entry Price": trade["Price"],
                "Exit Price": live_price,
                "PnL": trade["PnL"],
                "Notes": trade["Notes"],
                "Strategy": trade["Strategy"]
            }

            df = load_journal()
            df = pd.concat([df, pd.DataFrame([journal_entry])], ignore_index=True)
            save_journal(df)
            trades_to_close.append(trade)

    if trades_to_close:
        st.session_state.trades = [t for t in st.session_state.trades if t not in trades_to_close]
        save_open_trades(st.session_state.trades)

    # ---- DISPLAY OPEN TRADES ----
    if st.session_state.trades:
        st.subheader("Open Trades")
        for i, trade in enumerate(st.session_state.trades):
            col1, col2, col3, col4 = st.columns([2,2,2,1])
            col1.write(f"{trade['Asset']} ({trade['Direction']})")
            col2.write(f"Qty: {trade['Qty']} @ {trade['Price']}")
            col3.write(f"PnL: ${trade['PnL']:.2f}")
            if col4.button("Close", key=f"close_{i}"):
                live_price = live_prices.get(trade["Asset"].split("_")[0], trade["Price"])
                journal_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Asset": trade["Asset"],
                    "Direction": trade["Direction"],
                    "Entry Price": trade["Price"],
                    "Exit Price": live_price,
                    "PnL": compute_pnl(trade["Price"], live_price, trade["Direction"], trade["Qty"]),
                    "Notes": trade["Notes"],
                    "Strategy": trade["Strategy"]
                }
                df = load_journal()
                df = pd.concat([df, pd.DataFrame([journal_entry])], ignore_index=True)
                save_journal(df)
                st.session_state.trades.pop(i)
                save_open_trades(st.session_state.trades)
    else:
        st.info("No open trades")

    # ---- JOURNAL ----
    st.subheader("Trade Journal")
    journal_df = load_journal()
    st.dataframe(journal_df, use_container_width=True)

    # ---- TOTAL PnL CHART ----
    st.subheader("Total PnL")
    if not journal_df.empty:
        journal_df["Timestamp"] = pd.to_datetime(journal_df["Timestamp"])
        journal_df = journal_df.sort_values("Timestamp")
        journal_df["CumulativePnL"] = journal_df["PnL"].cumsum()
    else:
        journal_df = pd.DataFrame(columns=["Timestamp","CumulativePnL"])

    if st.session_state.trades:
        total_open_pnl = sum(t["PnL"] for t in st.session_state.trades)
        last_val = journal_df["CumulativePnL"].iloc[-1] if not journal_df.empty else 0
        journal_df = pd.concat([
            journal_df,
            pd.DataFrame([{"Timestamp": datetime.now(), "CumulativePnL": last_val + total_open_pnl}])
        ])

    if not journal_df.empty:
        fig = px.line(journal_df, x="Timestamp", y="CumulativePnL", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    from src.performance import compute_performance_metrics

    st.subheader("Performance Analytics")

    metrics = compute_performance_metrics(journal_df)

    if metrics:
        col1, col2, col3 = st.columns(3)

        col1.metric("Total PnL", f"${metrics['Total PnL']:.2f}")
        col2.metric("Win Rate", f"{metrics['Win Rate'] * 100:.1f}%")
        col3.metric("Trades", metrics["Trades"])

        col4, col5, col6 = st.columns(3)

        col4.metric("Avg Win", f"${metrics['Avg Win']:.2f}")
        col5.metric("Avg Loss", f"${metrics['Avg Loss']:.2f}")
        col6.metric("Max Drawdown", f"${metrics['Max Drawdown']:.2f}")

        st.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")

        st.subheader("PnL by Strategy")

        if not journal_df.empty and "Strategy" in journal_df.columns:
            strat_pnl = journal_df.groupby("Strategy")["PnL"].sum().reset_index()
            st.dataframe(strat_pnl)












