import os
import sys
import time
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask
import urllib.parse
import json
from openai import OpenAI
import datetime

# Point to 'src' directory for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from features import build_features
from regimes import build_regime_map
from signals import compute_all_signals
from strategies import get_all_strategy_signals

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LAST_SIGNAL_STATE = None
ACTIVE_TRADES = []

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


def calculate_trade_parameters(current_price: float, direction: str, current_atr: float):
    """
    Calculates Stop Loss using 2x Average True Range (ATR) to avoid getting stopped out by noise.
    """
    stop_distance = max(2.0 * current_atr, 0.02) 
    
    if direction in ["SHORT", "SELL"]:
        sl = current_price + stop_distance
        tp = current_price - (stop_distance * 1.5) # 1:1.5 Risk/Reward
    else:  # LONG / BUY
        sl = current_price - stop_distance
        tp = current_price + (stop_distance * 1.5) # 1:1.5 Risk/Reward
        
    return {"sl": round(sl, 3), "tp": round(tp, 3)}


def fetch_market_data():
    print("Fetching latest Henry Hub market data...")
    # Append a live timestamp to bust proxy caches and guarantee fresh 5-minute candles
    target_url = f"https://query1.finance.yahoo.com/v8/finance/chart/NG=F?interval=5m&range=5d&nocache={int(time.time())}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(target_url, headers=headers, timeout=10)

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
        highs = result["indicators"]["quote"][0]["high"]
        lows = result["indicators"]["quote"][0]["low"]
        volumes = result["indicators"]["quote"][0]["volume"]

        df = pd.DataFrame({
            "HH": closes,
            "Close": closes,
            "High": highs,
            "Low": lows,
            "Volume": volumes
        }, index=pd.to_datetime(timestamps, unit="s"))
        
        df = df.ffill().bfill().dropna()

        if len(df) < 25:
            print(f"⚠️ Warning: Insufficient candles fetched.")
            return pd.DataFrame()
            
        # --- NEW QUANTITATIVE METRICS ---
        # 1. Average True Range (ATR)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()

        # 2. Daily VWAP
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical_Price'] * df['Volume']
        df['VWAP'] = df.groupby(df.index.date)['VP'].cumsum() / df.groupby(df.index.date)['Volume'].cumsum()
        
        df = df.ffill().bfill() # Clean up the initial NaN rolling values

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


def evaluate_trade_with_ai(current_price, direction, strategy_name, conviction, market_state):
    """
    Acts as the Chief Risk Officer. Takes quantitative data and returns a JSON decision.
    """
    if not OPENAI_API_KEY:
        print("⚠️ No OpenAI API key found. Defaulting to standard 60-min hold.")
        return {"approved": True, "max_hold_mins": 60, "reason": "No AI configured. Default approval."}

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=60.0,
        max_retries=3
    )

    current_utc_time = time.strftime("%H:%M UTC")

    prompt = f"""
    You are the Chief Risk Officer for a quantitative natural gas (Henry Hub) trading desk.
    Evaluate the following algorithmic trade setup:

    - Asset: Henry Hub (NG=F)
    - Current Price: ${current_price:.3f}
    - Action: {direction}
    - Strategy: {strategy_name}
    - Algo Conviction: {conviction * 100:.1f}%
    - Market State Regime: {market_state}
    - Current Time: {current_utc_time}

    Rules:
    1. Mean Reversion trades in low-volume/downtrend US afternoon sessions (18:00 - 21:00 UTC) are highly dangerous.
    2. Overnight sessions (00:00 - 12:00 UTC) are algorithmic and range-bound, favoring longer hold times (up to 120 mins).
    3. Mean Reversion trades are quick "snap-backs" and should have a max hold of 60 mins.
    4. Trend trades require time to develop momentum. Approve high-conviction Trend trades with hold windows of 180 to 360 mins.

    Respond ONLY in raw JSON format with exactly three keys:
    "approved" (boolean), "max_hold_mins" (integer), and "reason" (string under 30 words).
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a quantitative risk management API. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        ai_decision = json.loads(response.choices[0].message.content)
        return ai_decision

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"❌ AI API Crash -> [{error_type}]: {error_msg}")
        return {
            "approved": True,
            "max_hold_mins": 60,
            "reason": f"AI Fallback triggered due to {error_type}"
        }


def evaluate_strategy():
    global LAST_SIGNAL_STATE
    global ACTIVE_TRADES
    
    df = fetch_market_data()
    if df.empty:
        return

    # Extract the new mathematical metrics before passing to src modules
    latest_raw = df.iloc[-1]
    current_atr = latest_raw["ATR"]
    daily_vwap = latest_raw["VWAP"]

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

    hh_strategies = strategy_df[strategy_df["Asset"] == "HH"]
    if hh_strategies.empty:
        print(f"ℹ️ Engine active. Current HH: ${current_price:.3f} (No HH signals).")
        return

    top_signal = hh_strategies.sort_values(by="Conviction", ascending=False).iloc[0]
    direction = top_signal["Signal"]
    strategy_name = top_signal["Strategy"]
    conviction = top_signal["Conviction"]

    print(f"📊 Evaluated HH: ${current_price:.3f} | Top: {strategy_name} {direction} | Conviction: {conviction * 100:.1f}%")

    if conviction < 0.60:
        return

    current_signal_state = f"HH_{direction}_{strategy_name}"
    if current_signal_state == LAST_SIGNAL_STATE:
        return

    # --- NEW: THE VWAP INSTITUTIONAL FILTER ---
    if strategy_name == "Trend":
        if direction == "LONG" and current_price < daily_vwap:
            print(f"🚫 Trend LONG vetoed: Price (${current_price:.3f}) is below VWAP (${daily_vwap:.3f}).")
            return
        elif direction == "SHORT" and current_price > daily_vwap:
            print(f"🚫 Trend SHORT vetoed: Price (${current_price:.3f}) is above VWAP (${daily_vwap:.3f}).")
            return

    LAST_SIGNAL_STATE = current_signal_state

    print("🧠 Math edge detected. Pinging AI Risk Manager for approval...")
    ai_decision = evaluate_trade_with_ai(current_price, direction, strategy_name, conviction, market_state)

    if not ai_decision["approved"]:
        print(f"🛑 Trade Vetoed by AI: {ai_decision['reason']}")
        send_telegram_alert(f"🛑 *TRADE VETOED BY AI*\n\n**Action:** `{direction} HH`\n**Reason:** {ai_decision['reason']}")
        return

    # Updated Risk Calculator utilizing ATR
    risk_params = calculate_trade_parameters(current_price, direction, current_atr)

    alert_text = (
        f"✅ *AI-Approved Trade Alert* 🚨\n\n"
        f"**Asset:** `Henry Hub (NG=F)`\n"
        f"**Action:** `{direction}`\n"
        f"**Entry Price:** `${current_price:.3f}`\n"
        f"**Strategy:** {strategy_name}\n"
        f"**Math Conviction:** {conviction * 100:.1f}%\n"
        f"**Market State:** `{market_state}`\n\n"
        f"--- *AI RISK PARAMETERS* ---\n"
        f"**Stop Loss:** `${risk_params['sl']}`\n"
        f"**Take Profit:** `${risk_params['tp']}`\n"
        f"**AI Max Hold:** `{ai_decision['max_hold_mins']} mins`\n"
        f"**AI Rationale:** {ai_decision['reason']}"
    )
    send_telegram_alert(alert_text)

    if ai_decision['max_hold_mins'] > 0:
        expiry_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=ai_decision['max_hold_mins'])
        trade_record = {
            "id": int(time.time()),
            "direction": direction,
            "entry_price": current_price,
            "strategy": strategy_name,
            "expiry_time": expiry_time
        }
        ACTIVE_TRADES.append(trade_record)
        print(f"📝 Trade stored in memory. Expiry set for: {expiry_time.strftime('%H:%M:%S UTC')}")


def order_manager_loop():
    global ACTIVE_TRADES
    
    print("🛡️ Order Manager online. Monitoring active positions...")
    while True:
        try:
            now = datetime.datetime.utcnow()
            for trade in ACTIVE_TRADES[:]:
                if now >= trade["expiry_time"]:
                    alert_text = (
                        f"⏱️ *AI TIME-STOP TRIGGERED* ⏱️\n\n"
                        f"**Action:** `CLOSE {trade['direction']}`\n"
                        f"**Strategy:** {trade['strategy']}\n"
                        f"**Original Entry:** `${trade['entry_price']:.3f}`\n\n"
                        f"Reason: The AI's dynamic hold window has expired. Close at market immediately."
                    )
                    send_telegram_alert(alert_text)
                    print(f"⏱️ Time stop enforced for {trade['strategy']} {trade['direction']}")
                    ACTIVE_TRADES.remove(trade)

        except Exception as e:
            print(f"❌ Order manager error: {e}")

        time.sleep(60)


def run_loop():
    print("🚀 Starting background HH strategy monitoring loop...")
    send_telegram_alert("🟢 *LNG Alert Engine Online*\nMonitoring Henry Hub setups with dynamic ATR & VWAP risk management.")
    while True:
        try:
            evaluate_strategy()
        except Exception as e:
            print(f"❌ Engine loop error: {e}")
        time.sleep(300)


loop_thread = threading.Thread(target=run_loop, daemon=True)
loop_thread.start()

manager_thread = threading.Thread(target=order_manager_loop, daemon=True)
manager_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
