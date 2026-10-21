import os
import sys
import time
import threading
import requests
import pandas as pd
import yfinance as yf
from flask import Flask

# Point to 'src' directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from features import build_features
from regimes import build_regime_map
from signals import compute_all_signals
from strategies import get_all_strategy_signals

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
LAST_SIGNAL_STATE = None

app = Flask(__name__)

@app.route("/")
def home():
    return "🟢 LNG Alert Engine is active and running!"

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ CRITICAL: TELEGRAM_TOKEN or CHAT_ID environment variables are missing!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            print(f"❌ Telegram API Error ({res.status_code}): {res.text}")
        else:
            print("✅ Telegram alert delivered successfully.")
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")

def calculate_trade_parameters(entry_price: float, signal_type: str, rolling_std: float, rolling_mean: float):
    stop_distance = max(1.5 * rolling_std, 0.02)
    if signal_type in ["SHORT", "SELL"]:
        stop_loss = entry_price + stop_distance
        take_profit = min(rolling_mean, entry_price - (1.5 * stop_distance))
    else:
        stop_loss = entry_price - stop_distance
        take_profit = max(rolling_mean, entry_price + (1.5 * stop_distance))
    
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr = reward / risk if risk > 0 else 0
    return {"sl": round(stop_loss, 3), "tp": round(take_profit, 3), "rr": round(rr, 2)}

def fetch_market_data():
    try:
        # Request 7 days to guarantee sufficient candles even across market closes
        hh_raw = yf.Ticker("NG=F").history(period="7d", interval="5m")
        if hh_raw.empty:
            print("⚠️ Warning: yfinance returned empty DataFrame for NG=F (Possible Rate Limit).")
            return pd.DataFrame()

        df = pd.DataFrame()
        df["HH"] = hh_raw["Close"].ffill().dropna()

        if len(df) < 25:
            print(f"⚠️ Warning: Insufficient candles fetched ({len(df)}/25 minimum).")
            return pd.DataFrame()

        df["TTF"] = df["HH"] * 4.0
        df["JKM"] = df["TTF"] * 1.15
        df["ARA_COAL"] = 100.0
        df["NEWC_COAL"] = 100.0
        return df
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return pd.DataFrame()

def evaluate_strategy():
    global LAST_SIGNAL_STATE
    df = fetch_market_data()
    if df.empty:
        return

    df_features = build_features(df)
    df_regimes = build_regime_map(df_features)
    if df_regimes.empty:
        print("⚠️ Warning: Regime map returned empty.")
        return

    latest_row = df_regimes.iloc[-1]
    current_price = latest_row["HH"]
    market_state = latest_row["MARKET_STATE"]

    signals_dict = compute_all_signals(df_regimes)
    strategy_df = get_all_strategy_signals(df_regimes)

    if strategy_df.empty:
        print(f"ℹ️ Engine active. Current HH: ${current_price:.3f} (No signals generated).")
        return

    hh_strategies = strategy_df[strategy_df["Asset"] == "HH"]
    if hh_strategies.empty:
        print(f"ℹ️ Engine active. Current HH: ${current_price:.3f} (No HH strategies active).")
        return

    top_signal = hh_strategies.sort_values(by="Conviction", ascending=False).iloc[0]
    direction = top_signal["Signal"]
    strategy_name = top_signal["Strategy"]
    conviction = top_signal["Conviction"]
    reason = top_signal["Reason"]

    print(f"📊 Evaluated HH: ${current_price:.3f} | Top: {strategy_name} {direction} | Conviction: {conviction*100:.1f}%")

    if conviction < 0.60:
        return

    rolling_std = latest_row["HH_VOL"]
    rolling_mean = df_regimes["HH"].rolling(20).mean().iloc[-1]
    risk_params = calculate_trade_parameters(current_price, direction, rolling_std, rolling_mean)

    composite_score = signals_dict.get("Composite", 0)
    composite_text = "BUY" if composite_score == 1 else "SELL" if composite_score == -1 else "HOLD"
    current_signal_state = f"HH_{direction}_{strategy_name}_{composite_text}"

    if current_signal_state != LAST_SIGNAL_STATE:
        LAST_SIGNAL_STATE = current_signal_state
        alert_text = (
            f"🚨 *Henry Hub Trade Alert* 🚨\n\n"
            f"**Asset:** `Henry Hub (NG=F)`\n"
            f"**Action:** `{direction}`\n"
            f"**Entry Price:** `${current_price:.3f}`\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Conviction:** {conviction * 100:.1f}%\n"
            f"**Trigger:** {reason}\n\n"
            f"--- *RISK MANAGEMENT* ---\n"
            f"**Stop Loss:** `${risk_params['sl']}`\n"
            f"**Take Profit:** `${risk_params['tp']}`\n"
            f"**Risk/Reward:** `1:{risk_params['rr']}`\n"
            f"**Max Hold:** `60 mins`\n\n"
            f"**Market State:** `{market_state}`"
        )
        send_telegram_alert(alert_text)

def run_loop():
    print("🚀 Starting background strategy monitoring loop...")
    send_telegram_alert("🟢 *LNG Alert Engine Online*\nTesting Telegram integration & starting live loop...")
    while True:
        try:
            evaluate_strategy()
        except Exception as e:
            print(f"❌ Engine loop error: {e}")
        time.sleep(300)

# Start background thread
loop_thread = threading.Thread(target=run_loop, daemon=True)
loop_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
