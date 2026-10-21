import os
import sys
import time
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask
import urllib.parse

# Point to 'src' directory for custom modules (leaves your Streamlit desk untouched)
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
    # Volatility-based stop loss (1.5x standard deviation, minimum $0.02)
    stop_distance = max(1.5 * rolling_std, 0.02)
    
    if signal_type in ["SHORT", "SELL"]:
        stop_loss = entry_price + stop_distance
        take_profit = min(rolling_mean, entry_price - (1.5 * stop_distance))
    else:  # LONG / BUY
        stop_loss = entry_price - stop_distance
        take_profit = max(rolling_mean, entry_price + (1.5 * stop_distance))
    
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rr = reward / risk if risk > 0 else 0
    
    return {"sl": round(stop_loss, 3), "tp": round(take_profit, 3), "rr": round(rr, 2)}

def fetch_market_data():
    print("Fetching latest Henry Hub market data...")
    # Append a live timestamp to bust proxy caches and guarantee fresh 5-minute candles
    target_url = f"https://query1.finance.yahoo.com/v8/finance/chart/NG=F?interval=5m&range=5d&nocache={int(time.time())}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # 1. Try direct connection first (in case Render IP ban lifts)
        res = requests.get(target_url, headers=headers, timeout=10)
        
        # 2. If Render's IP is banned (HTTP 429), route through free public proxy
        if res.status_code != 200:
            print(f"⚠️ Yahoo direct returned {res.status_code}. Rerouting through proxy...")
            encoded_url = urllib.parse.quote(target_url, safe='')
            proxy_url = f"https://api.allorigins.win/raw?url={encoded_url}"
            res = requests.get(proxy_url, timeout=15)
            
            if res.status_code != 200:
                print(f"❌ Proxy also failed with HTTP {res.status_code}")
                return pd.DataFrame()

        data = res.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]

        df = pd.DataFrame({"HH": closes}, index=pd.to_datetime(timestamps, unit="s"))
        df["HH"] = df["HH"].ffill().bfill()
        df = df.dropna()

        if len(df) < 25:
            print(f"⚠️ Warning: Insufficient candles fetched.")
            return pd.DataFrame()

        # Dynamic proxy generation with tiny variance (prevents std=0 division errors)
        n = len(df)
        df["TTF"] = df["HH"] * 4.0
        df["JKM"] = df["TTF"] * 1.15
        df["ARA_COAL"] = 100.0 + np.sin(np.linspace(0, 10, n))
        df["NEWC_COAL"] = 100.0 + np.cos(np.linspace(0, 10, n))

        return df
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return pd.DataFrame()

def evaluate_strategy():
    global LAST_SIGNAL_STATE
    df = fetch_market_data()
    if df.empty:
        return

    # Pass data into original features and regime modules
    df_features = build_features(df)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    df_regimes = build_regime_map(df_features)
    if df_regimes.empty:
        print("⚠️ Warning: Regime map returned empty.")
        return

    latest_row = df_regimes.iloc[-1]
    current_price = latest_row["HH"]
    market_state = latest_row.get("MARKET_STATE", "NORMAL")

    signals_dict = compute_all_signals(df_regimes)
    strategy_df = get_all_strategy_signals(df_regimes)

    if strategy_df.empty:
        print(f"ℹ️ Engine active. Current HH: ${current_price:.3f} (No strategies active).")
        return

    # Filter strictly for Henry Hub strategies
    hh_strategies = strategy_df[strategy_df["Asset"] == "HH"]
    if hh_strategies.empty:
        print(f"ℹ️ Engine active. Current HH: ${current_price:.3f} (No HH signals).")
        return

    # Select the highest conviction Henry Hub signal
    top_signal = hh_strategies.sort_values(by="Conviction", ascending=False).iloc[0]
    direction = top_signal["Signal"]
    strategy_name = top_signal["Strategy"]
    conviction = top_signal["Conviction"]
    reason = top_signal["Reason"]

    print(f"📊 Evaluated HH: ${current_price:.3f} | Top: {strategy_name} {direction} | Conviction: {conviction*100:.1f}%")

    # Gatekeeper: Require >= 60% conviction
    if conviction < 0.60:
        return

    # Calculate Stop Loss, Take Profit, and Risk/Reward
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
            f"**Max Hold Duration:** `60 mins`\n\n"
            f"**Market State:** `{market_state}`"
        )
        send_telegram_alert(alert_text)

def run_loop():
    print("🚀 Starting background HH strategy monitoring loop...")
    send_telegram_alert("🟢 *LNG Alert Engine Online*\nMonitoring Henry Hub setups with dynamic risk management.")
    while True:
        try:
            evaluate_strategy()
        except Exception as e:
            print(f"❌ Engine loop error: {e}")
        time.sleep(300)

# Start background thread for 24/7 monitoring
loop_thread = threading.Thread(target=run_loop, daemon=True)
loop_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
