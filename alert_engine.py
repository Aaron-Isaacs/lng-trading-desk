import time
import requests
import pandas as pd
import yfinance as yf

# Import your custom modules
# Adjust the import paths if your files are inside a src/ folder
from features import build_features
from regimes import build_regime_map
from signals import compute_all_signals
from strategies import get_all_strategy_signals

# Telegram Configuration
import os

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_FALLBACK")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_FALLBACK")

# Global state to prevent spamming the same alert
LAST_SIGNAL_STATE = None

def send_telegram_alert(message: str):
    """Sends a trade alert directly to your phone via Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send alert: {e}")


def fetch_market_data():
    """Fetches live HH and TTF from Yahoo Finance and generates placeholders for JKM/Coal."""
    print("Fetching latest Henry Hub & TTF market data...")
    try:
        # Download 5-minute intraday data for Henry Hub (NG=F) and Dutch TTF (TTF=F)
        hh_raw = yf.Ticker("NG=F").history(period="5d", interval="5m")
        ttf_raw = yf.Ticker("TTF=F").history(period="5d", interval="5m")

        if hh_raw.empty or ttf_raw.empty:
            print("Failed to retrieve HH or TTF data from Yahoo Finance.")
            return pd.DataFrame()

        # Extract Close price series
        hh = hh_raw["Close"].rename("HH")
        ttf = ttf_raw["Close"].rename("TTF")

        # Combine into a single DataFrame on timestamp index
        df = pd.concat([hh, ttf], axis=1).ffill().dropna()

        if df.empty or len(df) < 25:
            print(f"Insufficient overlapping rows ({len(df)} rows). Need at least 25.")
            return pd.DataFrame()

        # Populate synthetic placeholders so features.py executes without KeyError
        df["JKM"] = df["TTF"] * 1.15   # Synthetic JKM proxy based on TTF
        df["ARA_COAL"] = 100.0         # Dummy coal placeholder
        df["NEWC_COAL"] = 100.0        # Dummy coal placeholder

        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def evaluate_strategy():
    """Runs data through your modules and dispatches alerts."""
    global LAST_SIGNAL_STATE

    df = fetch_market_data()
    if df.empty or len(df) < 25:
        print("Not enough data to compute 20-period rolling features.")
        return

    # 1. Build Features and Regimes
    df_features = build_features(df)
    df_regimes = build_regime_map(df_features)

    if df_regimes.empty:
        return

    # 2. Get the latest price and state
    latest_row = df_regimes.iloc[-1]
    current_price = latest_row["HH"]
    market_state = latest_row["MARKET_STATE"]

    # 3. Compute Signals using your modules
    signals_dict = compute_all_signals(df_regimes)
    strategy_df = get_all_strategy_signals(df_regimes)

    # Filter strategy_df for Henry Hub specifically
    hh_strategies = strategy_df[strategy_df["Asset"] == "HH"]

    if hh_strategies.empty:
        print(f"No active HH strategy signals. Current price: ${current_price:.3f}")
        return

    # Let's extract the highest conviction signal for Henry Hub
    top_signal = hh_strategies.sort_values(by="Conviction", ascending=False).iloc[0]

    direction = top_signal["Signal"]
    strategy_name = top_signal["Strategy"]
    conviction = top_signal["Conviction"]
    reason = top_signal["Reason"]

    # Check Composite Signal alignment (from your signals.py)
    # signals_dict["Composite"] returns 1 (BUY), -1 (SELL), or 0
    composite_score = signals_dict.get("Composite", 0)
    composite_text = "BUY" if composite_score == 1 else "SELL" if composite_score == -1 else "HOLD"

    # Define a unique signature for this state so we don't spam the same alert
    current_signal_state = f"{direction}_{strategy_name}_{composite_text}"

    # Only alert if Conviction > 0.60 AND the signal state has changed
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
    else:
        print(f"State unchanged or conviction too low ({conviction:.2f}). Current HH: ${current_price:.3f}")


if __name__ == "__main__":
    print("Starting LNG Alert Engine...")
    send_telegram_alert("🟢 *LNG Alert Engine is Online*\nMonitoring Henry Hub & TTF signals every 5 minutes.")

    while True:
        try:
            evaluate_strategy()
        except Exception as e:
            print(f"Engine encountered an error: {e}")

        # Wait 5 minutes (300 seconds) before the next cycle
        time.sleep(300)