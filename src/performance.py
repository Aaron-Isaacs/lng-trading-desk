import pandas as pd
import numpy as np

def compute_performance_metrics(journal_df):
    if journal_df.empty:
        return {}

    pnl = journal_df["PnL"]

    total_pnl = pnl.sum()
    num_trades = len(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0

    # Risk metrics
    returns = pnl  # proxy
    sharpe = (returns.mean() / (returns.std() + 1e-6)) * np.sqrt(252) if len(returns) > 1 else 0

    # Drawdown
    cumulative = pnl.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_drawdown = drawdown.min()

    return {
        "Total PnL": total_pnl,
        "Trades": num_trades,
        "Win Rate": win_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown
    }