import os
import time
import threading
import requests
import pandas as pd
import yfinance as yf
from flask import Flask
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Custom imports
from features import build_features
from regimes import build_regime_map
from signals import compute_all_signals
from strategies import get_all_strategy_signals

# Environment credentials
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID")

LAST_SIGNAL_STATE = None

# Initialize Flask web server for Render health checks
app = Flask(__name__)

@app.route("/")
def home():
    return "🟢 LNG Alert Engine is active and running!"

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")

def fetch_market_data():
    print("Fetching latest Henry Hub & TTF market data...")
    try:
        hh_raw = yf.Ticker("NG=F").history(period="5d", interval="5m")
        ttf_raw = yf.Ticker("TTF=F").history(period="5d", interval="5m")

        if hh_raw.empty or ttf_raw.empty:
            return pd.DataFrame()

        hh = hh_raw["Close"].rename("HH")
        ttf = ttf_raw["Close"].rename("TTF")

        df = pd.concat([hh, ttf], axis=1).ffill().dropna()

        if df.empty or len(df) < 25:
            return pd.DataFrame()

        df["JKM"] = df["TTF"] * 1.15
        df["ARA_COAL"] = 100.0
        df["NEWC_COAL"] = 100.0

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def evaluate_strategy():
    global LAST_SIGNAL_STATE
    df = fetch_market_data()
    if df.empty or len(df) < 25:
        return

    df_features = build_features(df)
    df_regimes = build_regime_map(df_features)
    if df_regimes.empty:
        return

    latest_row = df_regimes.iloc[-1]
    current_price = latest_row["HH"]
    market_state = latest_row["MARKET_STATE"]

    signals_dict = compute_all_signals(df_regimes)
    strategy_df = get_all_strategy_signals(df_regimes)

    hh_strategies = strategy_df[strategy_df["Asset"] == "HH"]
    if hh_strategies.empty:
        print(f"No active HH strategy signals. Current HH: ${current_price:.3f}")
        return

    top_signal = hh_strategies.sort_values(by="Conviction", ascending=False).iloc[0]
    direction = top_signal["Signal"]
    strategy_name = top_signal["Strategy"]
    conviction = top_signal["Conviction"]
    reason = top_signal["Reason"]

    composite_score = signals_dict.get("Composite", 0)
    composite_text = "BUY" if composite_score == 1 else "SELL" if composite_score == -1 else "HOLD"

    current_signal_state = f"{direction}_{strategy_name}_{composite_text}"

    if conviction > 0.60 and current_signal_state != LAST_SIGNAL_STATE:
        LAST_SIGNAL_STATE = current_signal_state
        alert_text = (
            f"🚨 *Henry Hub Trade Alert* 🚨\n\n"
            f"**Action:** `{direction}`\n"
            f"**Price:** `${current_price:.3f}`\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Conviction:** {conviction * 100:.1f}%\n"
            f"**Trigger:** {reason}\n"
            f"**Market State:** `{market_state}`\n"
            f"**Composite Model:** {composite_text}"
        )
        send_telegram_alert(alert_text)
        print(f"Alert sent! {direction} at ${current_price:.3f}")

def run_loop():
    print("Starting background strategy loop...")
    send_telegram_alert("🟢 *LNG Alert Engine is Online*\nMonitoring Henry Hub & TTF signals every 5 minutes.")
    while True:
        try:
            evaluate_strategy()
        except Exception as e:
            print(f"Engine error: {e}")
        time.sleep(300)

# Start background monitoring thread automatically
threading.Thread(target=run_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)