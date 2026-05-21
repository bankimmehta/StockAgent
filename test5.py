import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from langchain_google_genai import ChatGoogleGenerativeAI
from tradingview_screener import Query, Column


# ==========================================
# FINANCIAL STATEMENTS ENGINE
# Fetches quarterly (last 8Q) and annual
# (last 5Y) Revenue, Net Income, Op Income
# ==========================================

def fetch_financial_statements(ticker_obj):
    """
    Returns dicts for quarterly and annual financial data.
    Each dict has: labels, revenue, net_income, op_income
    All values in billions (USD) for display.
    """
    result = {
        'quarterly': None,
        'annual':    None,
        'error':     None,
    }

    if ticker_obj is None:
        result['error'] = "No ticker object."
        return result

    def _to_billions(series):
        return [round(v / 1e9, 3) if pd.notna(v) else None for v in series]

    def _format_label_q(dt):
        return f"{dt.year}/Q{(dt.month - 1) // 3 + 1}"

    def _format_label_y(dt):
        return str(dt.year)

    def _growth(series):
        """QoQ or YoY % growth, None for first element."""
        out = [None]
        for i in range(1, len(series)):
            if series[i - 1] and series[i - 1] != 0 and series[i] is not None:
                out.append(round((series[i] - series[i - 1]) / abs(series[i - 1]) * 100, 2))
            else:
                out.append(None)
        return out

    try:
        # ---- QUARTERLY ----
        q_inc = ticker_obj.quarterly_income_stmt
        if q_inc is not None and not q_inc.empty:
            # Columns are dates, newest first — reverse for chronological order
            q_inc   = q_inc.iloc[:, ::-1].iloc[:, -8:]   # last 8 quarters

            rev_row  = next((r for r in ['Total Revenue', 'Revenue'] if r in q_inc.index), None)
            ni_row   = next((r for r in ['Net Income', 'Net Income Common Stockholders'] if r in q_inc.index), None)
            oi_row   = next((r for r in ['Operating Income', 'EBIT'] if r in q_inc.index), None)

            q_labels  = [_format_label_q(c) for c in q_inc.columns]
            q_rev     = _to_billions(q_inc.loc[rev_row]  if rev_row  else [None] * len(q_labels))
            q_ni      = _to_billions(q_inc.loc[ni_row]   if ni_row   else [None] * len(q_labels))
            q_oi      = _to_billions(q_inc.loc[oi_row]   if oi_row   else [None] * len(q_labels))

            result['quarterly'] = {
                'labels':    q_labels,
                'revenue':   q_rev,
                'net_income':q_ni,
                'op_income': q_oi,
                'rev_growth':_growth(q_rev),
                'ni_growth': _growth(q_ni),
                'oi_growth': _growth(q_oi),
            }

    except Exception as e:
        result['error'] = f"Quarterly fetch: {e}"

    try:
        # ---- ANNUAL ----
        a_inc = ticker_obj.income_stmt
        if a_inc is not None and not a_inc.empty:
            a_inc   = a_inc.iloc[:, ::-1].iloc[:, -5:]   # last 5 years

            rev_row  = next((r for r in ['Total Revenue', 'Revenue'] if r in a_inc.index), None)
            ni_row   = next((r for r in ['Net Income', 'Net Income Common Stockholders'] if r in a_inc.index), None)
            oi_row   = next((r for r in ['Operating Income', 'EBIT'] if r in a_inc.index), None)

            a_labels  = [_format_label_y(c) for c in a_inc.columns]
            a_rev     = _to_billions(a_inc.loc[rev_row]  if rev_row  else [None] * len(a_labels))
            a_ni      = _to_billions(a_inc.loc[ni_row]   if ni_row   else [None] * len(a_labels))
            a_oi      = _to_billions(a_inc.loc[oi_row]   if oi_row   else [None] * len(a_labels))

            result['annual'] = {
                'labels':    a_labels,
                'revenue':   a_rev,
                'net_income':a_ni,
                'op_income': a_oi,
                'rev_growth':_growth(a_rev),
                'ni_growth': _growth(a_ni),
                'oi_growth': _growth(a_oi),
            }

    except Exception as e:
        result['error'] = (result['error'] or "") + f" | Annual fetch: {e}"

    return result


def build_financial_bar_chart(labels, values, growth, title, bar_color, period_label):
    """
    Builds the bar + growth-line chart matching the design in the screenshot:
    - Dark background
    - Blue (or custom) bars for absolute values
    - Orange line for QoQ / YoY growth %
    - Value labels on top of each bar
    - Colour-coded growth: green positive, red negative
    - Dual y-axis: left = value (B), right = growth %
    """
    fig = go.Figure()

    # --- Bars ---
    bar_text = [f"{v:.2f}B" if v is not None else "N/A" for v in values]
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        name=title,
        marker_color=bar_color,
        text=bar_text,
        textposition='outside',
        textfont=dict(color='#56b4e9', size=11, family='monospace'),
        yaxis='y1',
        hovertemplate='%{x}<br>' + title + ': %{y:.3f}B<extra></extra>',
    ))

    # --- Growth line ---
    growth_colors = ['#26a69a' if (g is not None and g >= 0) else '#ef5350' for g in growth]
    fig.add_trace(go.Scatter(
        x=labels,
        y=growth,
        name=f'{period_label} Growth %',
        mode='lines+markers',
        line=dict(color='#f5a623', width=2.5),
        marker=dict(
            color=growth_colors,
            size=8,
            line=dict(color='#f5a623', width=1.5)
        ),
        yaxis='y2',
        hovertemplate='%{x}<br>Growth: %{y:.2f}%<extra></extra>',
        connectgaps=True,
    ))

    # --- Growth text labels on line ---
    growth_text = [f"{g:+.2f}%" if g is not None else "" for g in growth]
    fig.add_trace(go.Scatter(
        x=labels,
        y=growth,
        mode='text',
        text=growth_text,
        textfont=dict(
            color=['#26a69a' if (g is not None and g >= 0) else '#ef5350' for g in growth],
            size=10,
        ),
        textposition='top center',
        yaxis='y2',
        showlegend=False,
        hoverinfo='skip',
    ))

    # --- Layout ---
    fig.update_layout(
        title=dict(text=title, font=dict(color='#e0e0e0', size=14)),
        plot_bgcolor='rgba(18,18,28,1)',
        paper_bgcolor='rgba(18,18,28,1)',
        font=dict(color='#b0b0c0'),
        bargap=0.35,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-0.25,
            xanchor='center', x=0.5,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor='rgba(60,60,80,0.4)',
            tickfont=dict(size=11),
            linecolor='rgba(60,60,80,0.5)',
        ),
        yaxis=dict(
            title=dict(text='Value (USD B)', font=dict(size=11)),
            gridcolor='rgba(60,60,80,0.3)',
            tickformat='.2f',
            zeroline=False,
        ),
        yaxis2=dict(
            title=dict(text=f'{period_label} Growth %', font=dict(size=11)),
            overlaying='y',
            side='right',
            tickformat='.1f',
            ticksuffix='%',
            zeroline=True,
            zerolinecolor='rgba(120,120,140,0.4)',
            gridcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=50, b=60, l=60, r=60),
    )

    return fig


def build_financial_table(labels, values, growth, metric_name, period_label):
    """
    Builds the data table shown below the chart.
    Returns a styled DataFrame.
    """
    rows = {
        'Period':           labels,
        metric_name + ' (B)': [f"{v:.3f}" if v is not None else "N/A" for v in values],
        f'{period_label} Growth': [
            (f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%") if g is not None else "—"
            for g in growth
        ],
    }
    return pd.DataFrame(rows)


# ==========================================
# FINANCIAL RATIOS ENGINE
# Computes / fetches D/E, ROE, ROA and
# additional solvency/efficiency ratios
# ==========================================

def fetch_financial_ratios(ticker_obj, fund):
    """
    Returns a comprehensive ratio dict including
    D/E, ROE, ROA, current ratio, quick ratio,
    asset turnover, gross margin, op margin.
    Derives from yfinance .info + balance sheet
    where available.
    """
    ratios = {
        'de_ratio':        None,
        'roe':             None,
        'roa':             None,
        'current_ratio':   None,
        'quick_ratio':     None,
        'gross_margin':    None,
        'op_margin':       None,
        'asset_turnover':  None,
        'interest_coverage': None,
        'pb_ratio':        None,
        'ps_ratio':        None,
        'ev_ebitda':       None,
    }

    if ticker_obj is None:
        return ratios

    try:
        info = ticker_obj.info
        ratios['de_ratio']      = info.get('debtToEquity')
        ratios['roe']           = info.get('returnOnEquity')
        ratios['roa']           = info.get('returnOnAssets')
        ratios['current_ratio'] = info.get('currentRatio')
        ratios['quick_ratio']   = info.get('quickRatio')
        ratios['gross_margin']  = info.get('grossMargins')
        ratios['op_margin']     = info.get('operatingMargins')
        ratios['pb_ratio']      = info.get('priceToBook')
        ratios['ps_ratio']      = info.get('priceToSalesTrailing12Months')
        ratios['ev_ebitda']     = info.get('enterpriseToEbitda')

        # Asset turnover: Revenue / Total Assets (from balance sheet)
        try:
            bs = ticker_obj.balance_sheet
            inc = ticker_obj.income_stmt
            if bs is not None and not bs.empty and inc is not None and not inc.empty:
                ta_row  = next((r for r in ['Total Assets'] if r in bs.index), None)
                rev_row = next((r for r in ['Total Revenue', 'Revenue'] if r in inc.index), None)
                ei_row  = next((r for r in ['EBIT', 'Operating Income'] if r in inc.index), None)
                int_row = next((r for r in ['Interest Expense'] if r in inc.index), None)

                if ta_row and rev_row:
                    ta  = float(bs.loc[ta_row].iloc[0])
                    rev = float(inc.loc[rev_row].iloc[0])
                    if ta > 0:
                        ratios['asset_turnover'] = round(rev / ta, 3)

                if ei_row and int_row:
                    ebit     = float(inc.loc[ei_row].iloc[0])
                    int_exp  = abs(float(inc.loc[int_row].iloc[0]))
                    if int_exp > 0:
                        ratios['interest_coverage'] = round(ebit / int_exp, 2)

        except Exception:
            pass

    except Exception:
        pass

    return ratios


# ==========================================
# 1. TECHNICAL INDICATORS
# ==========================================

class RSIIndicator:
    def calculate(self, df):
        delta    = df['Close'].diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def get_summary(self, df):
        val = df['RSI'].iloc[-1]
        if pd.isna(val): return "RSI unavailable."
        state = "Overbought" if val > 70 else "Oversold" if val < 30 else "Neutral"
        return f"RSI: {val:.2f} ({state})"


class SMAIndicator:
    def __init__(self, window=50):
        self.window = window

    def calculate(self, df):
        df[f'SMA_{self.window}'] = df['Close'].rolling(self.window).mean()
        return df

    def get_summary(self, df):
        sma   = df[f'SMA_{self.window}'].iloc[-1]
        if pd.isna(sma): return f"SMA {self.window}: unavailable"
        close = df['Close'].iloc[-1]
        return f"{self.window} SMA: Price is {'Above' if close > sma else 'Below'} SMA ({sma:.2f})"


class BollingerBandsIndicator:
    def calculate(self, df):
        ma             = df['Close'].rolling(20).mean()
        std            = df['Close'].rolling(20).std()
        df['BB_UPPER'] = ma + (2 * std)
        df['BB_LOWER'] = ma - (2 * std)
        df['BB_MID']   = ma
        return df

    def get_summary(self, df):
        upper = df['BB_UPPER'].iloc[-1]
        lower = df['BB_LOWER'].iloc[-1]
        close = df['Close'].iloc[-1]
        if pd.isna(upper): return "Bollinger Bands unavailable."
        status = "Upper Breakout" if close > upper else "Lower Breakdown" if close < lower else "Inside Bands"
        bw = ((upper - lower) / df['BB_MID'].iloc[-1]) * 100 if df['BB_MID'].iloc[-1] > 0 else 0
        return f"Bollinger Bands: {status} | BandWidth: {bw:.1f}% | Upper: {upper:.2f} | Lower: {lower:.2f}"


class PriceTrendIndicator:
    def calculate(self, df):
        df['EMA_20']  = df['Close'].ewm(span=20,  adjust=False).mean()
        df['EMA_50']  = df['Close'].ewm(span=50,  adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        return df

    def get_summary(self, df):
        ema20  = df['EMA_20'].iloc[-1]
        ema50  = df['EMA_50'].iloc[-1]
        ema200 = df['EMA_200'].iloc[-1]
        if pd.isna(ema200): return "Trend unavailable."
        structure = (
            "Bullish" if ema20 > ema50 > ema200 else
            "Bearish" if ema20 < ema50 < ema200 else
            "Mixed / Transitioning"
        )
        return f"Trend: {structure} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f} | EMA200: {ema200:.2f}"


class SupportResistanceIndicator:
    def calculate(self, df):
        df['SUPPORT']       = df['Low'].rolling(40).min()
        df['RESISTANCE']    = df['High'].rolling(40).max()
        df['SUPPORT_20']    = df['Low'].rolling(20).min()
        df['RESISTANCE_20'] = df['High'].rolling(20).max()
        return df

    def get_summary(self, df):
        s40 = df['SUPPORT'].iloc[-1]
        r40 = df['RESISTANCE'].iloc[-1]
        s20 = df['SUPPORT_20'].iloc[-1]
        r20 = df['RESISTANCE_20'].iloc[-1]
        return (
            f"Support (40-bar): {s40:.2f} | Resistance (40-bar): {r40:.2f} | "
            f"Support (20-bar): {s20:.2f} | Resistance (20-bar): {r20:.2f}"
        )


class MACDIndicator:
    def calculate(self, df):
        ema12             = df['Close'].ewm(span=12, adjust=False).mean()
        ema26             = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']        = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
        return df

    def get_summary(self, df):
        macd   = df['MACD'].iloc[-1]
        signal = df['MACD_Signal'].iloc[-1]
        hist   = df['MACD_Hist'].iloc[-1]
        if pd.isna(macd): return "MACD unavailable."
        cross = "Bullish Crossover" if macd > signal else "Bearish Crossover"
        return f"MACD: {macd:.3f} | Signal: {signal:.3f} | Histogram: {hist:.3f} ({cross})"


class ATRIndicator:
    def calculate(self, df, period=14):
        high_low   = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close  = (df['Low']  - df['Close'].shift()).abs()
        tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR']  = tr.rolling(period).mean()
        return df

    def get_summary(self, df):
        atr   = df['ATR'].iloc[-1]
        close = df['Close'].iloc[-1]
        if pd.isna(atr): return "ATR unavailable."
        atr_pct = (atr / close) * 100
        regime  = "High Volatility" if atr_pct > 3 else "Low Volatility" if atr_pct < 1 else "Moderate Volatility"
        return f"ATR(14): {atr:.2f} ({atr_pct:.1f}% of price) — {regime}"


class VWAPIndicator:
    def calculate(self, df):
        typical    = (df['High'] + df['Low'] + df['Close']) / 3
        cum_vol    = df['Volume'].cumsum()
        cum_tpv    = (typical * df['Volume']).cumsum()
        df['VWAP'] = cum_tpv / (cum_vol + 1e-9)
        return df

    def get_summary(self, df):
        vwap  = df['VWAP'].iloc[-1]
        close = df['Close'].iloc[-1]
        if pd.isna(vwap): return "VWAP unavailable."
        pos = "Above VWAP (Bullish Bias)" if close > vwap else "Below VWAP (Bearish Bias)"
        return f"VWAP: {vwap:.2f} | Price is {pos}"


# ==========================================
# HISTORICAL BUY/SELL ZONE FINDER
# Scans 1-year OHLCV to find non-overlapping
# trade windows where ≥20% profit occurred.
# ==========================================

def find_historical_trade_zones(df, profit_target=0.20, max_hold_days=126):
    """
    Walks the 1-year Close series for local lows that achieved
    +profit_target within max_hold_days. Trades do not overlap.
    Local low filter: current close must be at or below the
    5-day prior rolling min (avoids buying into peaks).
    Also considers proximity to the 52-week low for context.
    """
    closes    = df['Close'].values
    dates     = df.index
    n         = len(closes)
    week52_low = float(np.min(closes))
    trades    = []
    i         = 5

    while i < n - 1:
        rolling_min = np.min(closes[max(0, i - 5):i])

        # Buy only at or near local lows
        if closes[i] > rolling_min * 1.005:
            i += 1
            continue

        buy_price = closes[i]
        target    = buy_price * (1.0 + profit_target)
        end       = min(i + max_hold_days, n)
        future    = closes[i + 1:end]
        hits      = np.where(future >= target)[0]

        if len(hits) > 0:
            sell_offset = int(hits[0])
            sell_idx    = i + 1 + sell_offset
            trades.append({
                'buy_date':       dates[i],
                'buy_price':      float(buy_price),
                'sell_date':      dates[sell_idx],
                'sell_price':     float(closes[sell_idx]),
                'hold_days':      sell_offset + 1,
                'profit_pct':     float((closes[sell_idx] - buy_price) / buy_price * 100),
                'vs_52wk_low':    float((buy_price - week52_low) / week52_low * 100),
            })
            i = sell_idx + 1
        else:
            i += 1

    return trades


# ==========================================
# STRATEGY-AWARE BUY RATE ENGINE
# Computes a recommended entry price range
# from 1yr historical data, strategy, and
# risk profile. No projections — purely
# derived from confirmed historical levels.
# ==========================================

def compute_recommended_buy_rate(df, risk, strategy, analyst_target, current_p):
    """
    Returns a dict with:
      buy_low       — lower bound of the entry zone
      buy_high      — upper bound of the entry zone
      ideal_entry   — single best entry price
      sell_target   — +20% above ideal_entry
      stop_loss     — protective stop below entry
      rationale     — one-line explanation of the logic
    """
    support_40  = float(df['SUPPORT'].iloc[-1])
    support_20  = float(df['SUPPORT_20'].iloc[-1])
    resist_20   = float(df['RESISTANCE_20'].iloc[-1])
    atr         = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else current_p * 0.02
    week52_low  = float(df['Close'].min())
    week52_high = float(df['Close'].max())
    vwap        = float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns and not pd.isna(df['VWAP'].iloc[-1]) else current_p

    # --- Risk multiplier: how deep below support to demand ---
    risk_discount = {
        "Conservative (Safe)":       0.97,   # want 3% below support
        "Moderate (Balanced)":       0.99,   # want 1% below support
        "Aggressive (High Growth)":  1.005,  # willing to buy just above support
    }
    r_mult = risk_discount.get(risk, 0.99)

    # --- Strategy logic ---
    if strategy == "Buying the Dip":
        # Anchor buy zone to the 20-bar support, discounted by risk
        ideal_entry = support_20 * r_mult
        buy_low     = ideal_entry - atr * 0.5
        buy_high    = support_20 + atr * 0.5
        rationale   = (
            f"Dip buyer strategy: entry zone anchored to 20-bar support ({support_20:.2f}) "
            f"with ±0.5 ATR buffer ({atr:.2f}). Risk discount applied: {r_mult:.1%}."
        )

    elif strategy == "Momentum Trading":
        # Buy on a mild 2–3% intraday pullback from current price, above VWAP
        pullback    = 0.97 if "Conservative" in risk else 0.985 if "Moderate" in risk else 0.995
        ideal_entry = max(current_p * pullback, vwap)
        buy_low     = ideal_entry - atr * 0.3
        buy_high    = ideal_entry + atr * 0.3
        rationale   = (
            f"Momentum strategy: buy on {(1 - pullback) * 100:.1f}% pullback from current "
            f"price ({current_p:.2f}), minimum floor at VWAP ({vwap:.2f})."
        )

    elif strategy == "Breakout Trading":
        # Buy only when price clears 20-bar resistance with buffer
        breakout_level = resist_20 * 1.005
        ideal_entry    = breakout_level
        buy_low        = resist_20
        buy_high       = resist_20 + atr
        rationale      = (
            f"Breakout strategy: entry ONLY above 20-bar resistance ({resist_20:.2f}) "
            f"confirmed with 0.5% breakout buffer. Do NOT buy below {resist_20:.2f}."
        )

    else:
        # Long Term Investing: accumulate in the lower third of the 52-week range
        lower_third = week52_low + (week52_high - week52_low) * 0.33
        ideal_entry = min(lower_third * r_mult, support_40)
        buy_low     = week52_low * 1.02          # 2% above 52-week floor
        buy_high    = lower_third
        rationale   = (
            f"Long-term accumulation zone: lower third of 52-week range "
            f"({week52_low:.2f}–{week52_high:.2f}). "
            f"Ideal accumulation band: {buy_low:.2f}–{buy_high:.2f}."
        )

    # Enforce: ideal_entry must be > 0 and ≤ current_p * 1.02
    ideal_entry = max(ideal_entry, week52_low)
    ideal_entry = min(ideal_entry, current_p * 1.02)
    buy_low     = max(buy_low, week52_low * 0.98)
    buy_high    = min(buy_high, current_p * 1.05)

    sell_target = round(ideal_entry * 1.20, 2)
    stop_loss   = round(ideal_entry - atr * 1.5, 2)

    # If analyst target is lower than sell_target, flag it
    analyst_note = ""
    if analyst_target and analyst_target > 0 and analyst_target < sell_target:
        analyst_note = (
            f" ⚠️ Analyst consensus target ({analyst_target:.2f}) is BELOW the +20% "
            f"sell target ({sell_target:.2f}) — 20% gain not confirmed by analyst coverage."
        )

    return {
        'buy_low':      round(buy_low,     2),
        'buy_high':     round(buy_high,    2),
        'ideal_entry':  round(ideal_entry, 2),
        'sell_target':  sell_target,
        'stop_loss':    stop_loss,
        'rationale':    rationale + analyst_note,
    }


# ==========================================
# FUNDAMENTALS DATA ENGINE
# Pulls key financial ratios from yfinance.
# No projection logic — raw facts only.
# ==========================================

def fetch_fundamentals(ticker_obj):
    """
    Returns a dict of key financial ratios and analyst targets.
    No 6M projections. Raw data — verified or marked N/A.
    """
    result = {
        'trailing_pe':     None,
        'forward_pe':      None,
        'trailing_eps':    None,
        'forward_eps':     None,
        'earnings_growth': None,
        'revenue_growth':  None,
        'analyst_target':  None,
        'analyst_low':     None,
        'analyst_high':    None,
        'debt_to_equity':  None,
        'roe':             None,
        'profit_margin':   None,
        'market_cap':      None,
        'sector':          'N/A',
        'industry':        'N/A',
        'data_note':       'No fundamental data available.',
    }

    if ticker_obj is None:
        return result

    try:
        info = ticker_obj.info

        result['trailing_pe']     = info.get('trailingPE')
        result['forward_pe']      = info.get('forwardPE')
        result['trailing_eps']    = info.get('trailingEps')
        result['forward_eps']     = info.get('forwardEps')
        result['earnings_growth'] = info.get('earningsGrowth')
        result['revenue_growth']  = info.get('revenueGrowth')
        result['analyst_target']  = info.get('targetMeanPrice')
        result['analyst_low']     = info.get('targetLowPrice')
        result['analyst_high']    = info.get('targetHighPrice')
        result['debt_to_equity']  = info.get('debtToEquity')
        result['roe']             = info.get('returnOnEquity')
        result['profit_margin']   = info.get('profitMargins')
        result['market_cap']      = info.get('marketCap')
        result['sector']          = info.get('sector', 'N/A')
        result['industry']        = info.get('industry', 'N/A')
        result['data_note']       = 'Live data from yFinance.'

    except Exception as ex:
        result['data_note'] = f"Fetch error: {ex}"

    return result


# ==========================================
# 2. CORE DATA ENGINE
# ==========================================

class StockAgentEngine:
    def __init__(self, profile, indicators, api_key):
        self.profile    = profile
        self.indicators = indicators
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    def _fetch_from_tradingview(self, ticker, market_type):
        try:
            exchange     = 'india' if market_type == "NSE (India)" else 'america'
            clean_ticker = ticker.replace(".NS", "")
            result = (
                Query().set_markets(exchange)
                .select('name', 'close', 'high', 'low', 'open', 'volume')
                .where(Column('name').equals(clean_ticker))
                .get_data()
            )
            if not result or len(result) < 2: return None
            df_tv = result[1]
            if df_tv is None or df_tv.empty: return None
            row = df_tv.iloc[0]
            return pd.DataFrame({
                'Open':   [row.get('open',   row['close'])],
                'High':   [row.get('high',   row['close'])],
                'Low':    [row.get('low',    row['close'])],
                'Close':  [row['close']],
                'Volume': [row.get('volume', 0)],
            }, index=[pd.Timestamp.now()])
        except Exception:
            return None

    def run(self, ticker, market_type):
        df               = pd.DataFrame()
        source_used      = "Yahoo Finance"
        news_headlines   = []
        inst_data_return = None
        ticker_obj       = None

        try:
            ticker_obj = yf.Ticker(ticker)
            # FIX: always pull 1 year of daily data — hardcoded, not user-selectable
            df = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)

            try:
                raw_news = ticker_obj.news or []
                news_headlines = [
                    n.get('content', {}).get('title', n.get('title', 'Unknown headline'))
                    for n in raw_news[:8]
                ] if raw_news else ["No recent news available."]
            except Exception:
                news_headlines = ["Unable to fetch news."]

            try:
                inst_holders = ticker_obj.institutional_holders
                if inst_holders is not None and not inst_holders.empty:
                    inst_data_return = inst_holders.head(10)
                else:
                    inst_data_return = None
            except Exception:
                inst_data_return = None

        except Exception:
            pass

        if df.empty:
            source_used    = "TradingView (Limited Data)"
            df             = self._fetch_from_tradingview(ticker, market_type)
            news_headlines = ["News engine requires yFinance."]

        if df is None or df.empty:
            return None, "Unable to load market data.", "None", [], None, None

        if len(df) < 50:
            return None, "Need at least 50 trading days of data.", source_used, [], None, None

        metrics = []
        for ind in self.indicators:
            df = ind.calculate(df)
            metrics.append(ind.get_summary(df))

        indicator_text = "\n".join(metrics)
        news_text      = "\n".join([f"- {n}" for n in news_headlines])

        # ==========================================
        # AI PROMPT — STRICT 8-10 BULLETS, NO FLUFF
        # ==========================================
        prompt = f"""
You are a ruthlessly honest institutional equity analyst. You analyze data and state conclusions directly.

Risk Profile: {self.profile['risk']}
Strategy: {self.profile['strategy']}

Technical Indicator Readings (verified from 1-year daily data):
{indicator_text}

Recent Market Headlines:
{news_text}

STRICT OUTPUT FORMAT:
- Write EXACTLY 8 to 10 numbered bullet points. No more. No less.
- Each bullet is ONE sentence only. No sub-bullets. No paragraphs.
- Use specific numbers from the indicator data above. No vague statements.
- Do NOT say "may", "could", "might", "should consider" — state facts and direct conclusions only.
- Do NOT repeat the indicator values already shown above — interpret them.
- The LAST bullet MUST follow this exact format:
  "Verdict: [BUY / HOLD / AVOID] — Reason: [one direct sentence stating why]."

Cover in order: trend, momentum, volatility, support/resistance proximity, news impact, volume/institutional signal, risk level, entry timing verdict.
"""

        response = self.llm.invoke(prompt)
        return df, response.content, source_used, news_headlines, inst_data_return, ticker_obj


# ==========================================
# 3. STREAMLIT APPLICATION INTERFACE
# ==========================================

st.set_page_config(page_title="AI Strategic Stock Analyst", layout="wide")
st.title("📈 AI Strategic Investment Desk")

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Configuration")

risk     = st.sidebar.selectbox(
    "Risk Profile",
    ["Conservative (Safe)", "Moderate (Balanced)", "Aggressive (High Growth)"]
)
strategy = st.sidebar.selectbox(
    "Strategy",
    ["Long Term Investing", "Buying the Dip", "Momentum Trading", "Breakout Trading"]
)

st.sidebar.subheader("Indicators")
use_rsi   = st.sidebar.checkbox("RSI (Wilder Smoothed)",       value=True)
use_macd  = st.sidebar.checkbox("MACD (12/26/9)",              value=True)
use_sma   = st.sidebar.checkbox("SMA (50-period)",             value=True)
use_bb    = st.sidebar.checkbox("Bollinger Bands",             value=True)
use_trend = st.sidebar.checkbox("EMA Trend (20/50/200)",       value=True)
use_atr   = st.sidebar.checkbox("ATR Volatility (14)",         value=True)
use_vwap  = st.sidebar.checkbox("VWAP",                        value=True)

st.sidebar.caption("📅 Historical window: 1 Year (fixed)")

market_type  = st.selectbox("Exchange", ["US Markets", "NSE (India)"])
ticker_raw   = st.text_input("Ticker Symbol", "AAPL")
ticker_input = ticker_raw.strip().upper()
if market_type == "NSE (India)" and not ticker_input.endswith(".NS"):
    ticker_input += ".NS"


# ==========================================
# RUN ENGINE
# ==========================================

if st.button("Run Analysis"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY environment variable is not set.")
    else:
        # S/R is always forced — all other indicator logic depends on it
        indicators = [SupportResistanceIndicator()]
        if use_rsi:   indicators.append(RSIIndicator())
        if use_macd:  indicators.append(MACDIndicator())
        if use_sma:   indicators.append(SMAIndicator())
        if use_bb:    indicators.append(BollingerBandsIndicator())
        if use_trend: indicators.append(PriceTrendIndicator())
        if use_atr:   indicators.append(ATRIndicator())
        if use_vwap:  indicators.append(VWAPIndicator())

        profile_dict = {"risk": risk, "strategy": strategy}

        with st.spinner("Pulling 1-year data, running indicators, verifying technicals and fundamentals..."):
            try:
                df, report, source, news_headlines, inst_data, ticker_obj = (
                    StockAgentEngine(profile_dict, indicators, api_key).run(
                        ticker=ticker_input,
                        market_type=market_type,
                    )
                )

                if df is None:
                    st.error(report)
                else:
                    st.success(f"Data source: {source} | 1-year daily bars loaded: {len(df)}")

                    current_p       = float(df['Close'].iloc[-1])
                    day_high        = float(df['High'].iloc[-1])
                    day_low         = float(df['Low'].iloc[-1])
                    curr_support    = float(df['SUPPORT'].iloc[-1])
                    curr_res        = float(df['RESISTANCE'].iloc[-1])
                    curr_support_20 = float(df['SUPPORT_20'].iloc[-1])
                    curr_res_20     = float(df['RESISTANCE_20'].iloc[-1])
                    week52_low      = float(df['Close'].min())
                    week52_high     = float(df['Close'].max())

                    log_returns    = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                    annualized_vol = log_returns.std() * np.sqrt(252)

                    # Fundamentals
                    fund = fetch_fundamentals(ticker_obj)

                    # Strategy-aware buy rate
                    buy_rec = compute_recommended_buy_rate(
                        df, risk, strategy, fund['analyst_target'], current_p
                    )

                    # ==========================================
                    # TOP METRICS
                    # ==========================================
                    st.markdown("### 📊 Market Snapshot")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Current Price",     f"{current_p:.2f}")
                    c2.metric("52-Week Low / High", f"{week52_low:.2f} / {week52_high:.2f}")
                    c3.metric("Support (40-bar)",  f"{curr_support:.2f}")
                    c4.metric("Resistance (40-bar)",f"{curr_res:.2f}")
                    c5.metric("Annualised Vol",     f"{annualized_vol:.2%}")

                    st.markdown("---")

                    # ==========================================
                    # RECOMMENDED BUY RATE — STRATEGY AWARE
                    # ==========================================
                    st.markdown("## 🎯 Recommended Buy Rate")

                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Ideal Entry",    f"{buy_rec['ideal_entry']:.2f}")
                    b2.metric("Entry Zone",     f"{buy_rec['buy_low']:.2f} – {buy_rec['buy_high']:.2f}")
                    b3.metric("+20% Sell Target",f"{buy_rec['sell_target']:.2f}")
                    b4.metric("Stop Loss",      f"{buy_rec['stop_loss']:.2f}")

                    st.info(f"**Strategy Logic:** {buy_rec['rationale']}")

                    dist_from_current = ((buy_rec['ideal_entry'] - current_p) / current_p) * 100
                    if dist_from_current < -15:
                        st.warning(
                            f"⚠️ Ideal entry ({buy_rec['ideal_entry']:.2f}) is "
                            f"{abs(dist_from_current):.1f}% below current price. "
                            f"This level may not be reached without a significant pullback."
                        )
                    elif dist_from_current >= 0:
                        st.warning(
                            f"⚠️ Breakout strategy entry ({buy_rec['ideal_entry']:.2f}) is "
                            f"{dist_from_current:.1f}% above current price. "
                            f"Wait for confirmed breakout — do not chase."
                        )
                    else:
                        st.success(
                            f"✅ Entry zone is {abs(dist_from_current):.1f}% below current price "
                            f"— within a normal pullback range for this strategy."
                        )

                    st.markdown("---")

                    # ==========================================
                    # INDICATOR SUMMARY TABLE
                    # ==========================================
                    st.markdown("## 🧮 Technical Indicator Readings")
                    ind_rows = []
                    for ind in indicators:
                        ind_rows.append({
                            "Indicator": type(ind).__name__.replace("Indicator", ""),
                            "Reading":   ind.get_summary(df)
                        })
                    st.table(pd.DataFrame(ind_rows))

                    st.markdown("---")

                    # ==========================================
                    # AI ANALYST REPORT — 8-10 BULLETS ONLY
                    # ==========================================
                    st.markdown("## 📋 Analyst Verdict (8–10 Points)")
                    st.write(report)

                    st.markdown("---")

                    # ==========================================
                    # HISTORICAL CHART — 1 YEAR WITH ALL LAYERS
                    # ==========================================
                    st.markdown("## 📈 1-Year Historical Price Structure")

                    hist_trades = find_historical_trade_zones(df, profit_target=0.20, max_hold_days=126)

                    fig_hist = go.Figure()

                    fig_hist.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='OHLC',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ))

                    if 'EMA_20' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['EMA_20'],  name='EMA 20',  line=dict(color='orange', width=1.5, dash='dash')))
                    if 'EMA_50' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['EMA_50'],  name='EMA 50',  line=dict(color='yellow', width=1.5, dash='dot')))
                    if 'EMA_200' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name='EMA 200', line=dict(color='red',    width=1.5, dash='dot')))
                    if 'SMA_50' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],  name='SMA 50',  line=dict(color='cyan',   width=1.2, dash='dash')))
                    if 'BB_UPPER' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB Upper', line=dict(color='rgba(150,150,255,0.5)', width=1)))
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB Lower', line=dict(color='rgba(150,150,255,0.5)', width=1), fill='tonexty', fillcolor='rgba(150,150,255,0.04)'))
                    if 'VWAP' in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='magenta', width=1.2, dash='dot')))

                    # S/R levels
                    fig_hist.add_hline(y=curr_support,    line_dash="dash", line_color="rgba(0,255,0,0.4)",   annotation_text=f"Support 40: {curr_support:.2f}")
                    fig_hist.add_hline(y=curr_res,        line_dash="dash", line_color="rgba(255,80,80,0.4)", annotation_text=f"Resistance 40: {curr_res:.2f}")
                    fig_hist.add_hline(y=curr_support_20, line_dash="dot",  line_color="rgba(0,200,0,0.3)",   annotation_text=f"Support 20: {curr_support_20:.2f}")
                    fig_hist.add_hline(y=curr_res_20,     line_dash="dot",  line_color="rgba(200,60,60,0.3)", annotation_text=f"Resistance 20: {curr_res_20:.2f}")

                    # Recommended buy zone band
                    fig_hist.add_hrect(
                        y0=buy_rec['buy_low'], y1=buy_rec['buy_high'],
                        fillcolor="rgba(0,255,100,0.07)",
                        line=dict(color="rgba(0,255,100,0.3)", width=1, dash="dot"),
                        annotation_text=f"Buy Zone {buy_rec['buy_low']:.2f}–{buy_rec['buy_high']:.2f}",
                        annotation_position="top left",
                        annotation_font_color="lightgreen"
                    )

                    # Historical trades where ≥20% profit was achieved
                    if hist_trades:
                        fig_hist.add_trace(go.Scatter(
                            x=[t['buy_date']  for t in hist_trades],
                            y=[t['buy_price'] for t in hist_trades],
                            name='Historical BUY (≥20% achieved)', mode='markers',
                            marker=dict(color='lime', size=10, symbol='triangle-up', line=dict(color='darkgreen', width=1.2)),
                            text=[f"BUY {t['buy_price']:.2f}" for t in hist_trades],
                            hoverinfo='text+x'
                        ))
                        fig_hist.add_trace(go.Scatter(
                            x=[t['sell_date']  for t in hist_trades],
                            y=[t['sell_price'] for t in hist_trades],
                            name='Historical SELL (≥20% achieved)', mode='markers',
                            marker=dict(color='crimson', size=10, symbol='triangle-down', line=dict(color='darkred', width=1.2)),
                            text=[f"SELL {t['sell_price']:.2f} (+{t['profit_pct']:.0f}%)" for t in hist_trades],
                            hoverinfo='text+x'
                        ))

                    fig_hist.update_layout(
                        title=f"{ticker_input} — 1-Year Structure | Strategy: {strategy} | Profile: {risk}",
                        hovermode='x unified', xaxis_rangeslider_visible=False,
                        plot_bgcolor='rgba(15,15,15,1)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(50,50,50,0.2)'),
                        yaxis=dict(gridcolor='rgba(50,50,50,0.2)')
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Historical ≥20% trade table
                    if hist_trades:
                        st.markdown(f"**{len(hist_trades)} confirmed ≥20% profit window(s) in the past 1 year:**")
                        hist_df = pd.DataFrame(hist_trades)
                        hist_df['buy_date']     = pd.to_datetime(hist_df['buy_date']).dt.strftime('%Y-%m-%d')
                        hist_df['sell_date']    = pd.to_datetime(hist_df['sell_date']).dt.strftime('%Y-%m-%d')
                        hist_df['buy_price']    = hist_df['buy_price'].apply(lambda x: f"{x:.2f}")
                        hist_df['sell_price']   = hist_df['sell_price'].apply(lambda x: f"{x:.2f}")
                        hist_df['profit_pct']   = hist_df['profit_pct'].apply(lambda x: f"{x:.1f}%")
                        hist_df['vs_52wk_low']  = hist_df['vs_52wk_low'].apply(lambda x: f"+{x:.1f}% above 52wk low")
                        hist_df.columns         = ['Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Hold Days', 'Profit', '% Above 52Wk Low']
                        st.dataframe(hist_df, use_container_width=True)
                        avg_hold = np.mean([t['hold_days'] for t in hist_trades])
                        avg_profit = np.mean([t['profit_pct'] for t in hist_trades])
                        st.caption(
                            f"Average hold period to reach +20%: {avg_hold:.0f} days | "
                            f"Average actual profit at exit: {avg_profit:.1f}%"
                        )
                    else:
                        st.warning("⚠️ No ≥20% profit windows found in the past 1 year. "
                                   "This stock did not produce clean long entries that reached +20% within 6 months.")

                    st.markdown("---")

                    # ==========================================
                    # NEWS SENTIMENT
                    # ==========================================
                    st.markdown("## 📰 News Sentiment")
                    if news_headlines:
                        for h in news_headlines:
                            hl = h.lower()
                            if any(w in hl for w in ['surge', 'rally', 'beat', 'record', 'growth', 'upgrade', 'profit', 'rise', 'gain', 'strong']):
                                badge = "🟢 Bullish"
                            elif any(w in hl for w in ['fall', 'drop', 'miss', 'loss', 'risk', 'decline', 'downgrade', 'concern', 'cut', 'sell-off', 'tariff', 'fine', 'lawsuit']):
                                badge = "🔴 Bearish"
                            else:
                                badge = "🟡 Neutral"
                            st.markdown(f"- {badge} — {h}")
                    else:
                        st.info("No news available.")

                    st.markdown("---")

                    # ==========================================
                    # INSTITUTIONAL HOLDERS
                    # ==========================================
                    st.markdown("## 🏦 Institutional Holders")
                    if inst_data is not None and isinstance(inst_data, pd.DataFrame):
                        st.dataframe(inst_data, use_container_width=True)
                    else:
                        st.info("Institutional holder data not available for this asset.")

                    st.markdown("---")

                    # ==========================================
                    # FUNDAMENTALS RATIOS TABLE
                    # No projections. Raw verified data only.
                    # ==========================================
                    st.markdown("## 🏗️ Fundamental Ratios & Analyst Targets")
                    st.caption(f"Source: {fund['data_note']}")

                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Analyst Target (12M)", f"{fund['analyst_target']:.2f}" if fund['analyst_target'] else "N/A")
                    f2.metric("Analyst Low",           f"{fund['analyst_low']:.2f}"   if fund['analyst_low']    else "N/A")
                    f3.metric("Analyst High",          f"{fund['analyst_high']:.2f}"  if fund['analyst_high']   else "N/A")
                    f4.metric("Forward P/E",           f"{fund['forward_pe']:.1f}x"   if fund['forward_pe']     else "N/A")

                    ratio_rows = {
                        "Metric": [
                            "Sector", "Industry",
                            "Trailing P/E", "Forward P/E",
                            "Trailing EPS", "Forward EPS",
                            "Earnings Growth (YoY)", "Revenue Growth (YoY)",
                            "Return on Equity", "Profit Margin",
                            "Debt / Equity", "Market Cap"
                        ],
                        "Value": [
                            fund['sector'], fund['industry'],
                            f"{fund['trailing_pe']:.1f}x"    if fund['trailing_pe']     else "N/A",
                            f"{fund['forward_pe']:.1f}x"     if fund['forward_pe']      else "N/A",
                            f"{fund['trailing_eps']:.2f}"    if fund['trailing_eps']     else "N/A",
                            f"{fund['forward_eps']:.2f}"     if fund['forward_eps']      else "N/A",
                            f"{fund['earnings_growth']:.1%}" if fund['earnings_growth']  else "N/A",
                            f"{fund['revenue_growth']:.1%}"  if fund['revenue_growth']   else "N/A",
                            f"{fund['roe']:.1%}"             if fund['roe']              else "N/A",
                            f"{fund['profit_margin']:.1%}"   if fund['profit_margin']    else "N/A",
                            f"{fund['debt_to_equity']:.1f}"  if fund['debt_to_equity']   else "N/A",
                            f"{fund['market_cap']:,.0f}"     if fund['market_cap']       else "N/A",
                        ]
                    }
                    st.table(pd.DataFrame(ratio_rows))

                    st.markdown("---")

                    # ==========================================
                    # FINANCIAL PERFORMANCE CHARTS
                    # Quarterly (last 8Q) + Annual (last 5Y)
                    # Tabs: Total Revenue | Net Income | Op Income
                    # ==========================================
                    st.markdown("## 📊 Financial Performance — Revenue & Profit")

                    fin = fetch_financial_statements(ticker_obj)

                    if fin['quarterly'] is None and fin['annual'] is None:
                        st.warning(
                            f"Financial statement data unavailable for {ticker_input}. "
                            f"{fin.get('error', '')}"
                        )
                    else:
                        fin_period    = st.radio(
                            "View Period",
                            ["Quarterly (Last 8Q)", "Annual (Last 5Y)"],
                            horizontal=True, key="fin_period_toggle"
                        )
                        use_quarterly = fin_period.startswith("Quarterly")
                        fin_data      = fin['quarterly'] if use_quarterly else fin['annual']
                        period_lbl    = "QoQ" if use_quarterly else "YoY"

                        if fin_data is None:
                            st.warning(f"{'Quarterly' if use_quarterly else 'Annual'} data not available.")
                        else:
                            labels = fin_data['labels']
                            tab_rev, tab_ni, tab_oi = st.tabs(
                                ["📈 Total Revenue", "💰 Net Income", "⚙️ Operating Income"]
                            )

                            def _styled_table(tbl, period_col):
                                def _c(val):
                                    if val == "—": return "color: #888"
                                    return "color: #26a69a" if str(val).startswith("+") else "color: #ef5350"
                                return tbl.style.applymap(_c, subset=[period_col])

                            with tab_rev:
                                rev_v = fin_data['revenue']
                                rev_g = fin_data['rev_growth']
                                if any(v is not None for v in rev_v):
                                    st.plotly_chart(
                                        build_financial_bar_chart(labels, rev_v, rev_g, "Total Revenue (USD B)", "#29b6f6", period_lbl),
                                        use_container_width=True
                                    )
                                    tbl = build_financial_table(labels, rev_v, rev_g, "Revenue", period_lbl)
                                    st.dataframe(_styled_table(tbl, f'{period_lbl} Growth'), use_container_width=True, hide_index=True)
                                else:
                                    st.info("Revenue data not available.")

                            with tab_ni:
                                ni_v = fin_data['net_income']
                                ni_g = fin_data['ni_growth']
                                if any(v is not None for v in ni_v):
                                    st.plotly_chart(
                                        build_financial_bar_chart(labels, ni_v, ni_g, "Net Income (USD B)", "#ab47bc", period_lbl),
                                        use_container_width=True
                                    )
                                    tbl = build_financial_table(labels, ni_v, ni_g, "Net Income", period_lbl)
                                    st.dataframe(_styled_table(tbl, f'{period_lbl} Growth'), use_container_width=True, hide_index=True)
                                else:
                                    st.info("Net income data not available.")

                            with tab_oi:
                                oi_v = fin_data['op_income']
                                oi_g = fin_data['oi_growth']
                                if any(v is not None for v in oi_v):
                                    st.plotly_chart(
                                        build_financial_bar_chart(labels, oi_v, oi_g, "Operating Income (USD B)", "#26a69a", period_lbl),
                                        use_container_width=True
                                    )
                                    tbl = build_financial_table(labels, oi_v, oi_g, "Op Income", period_lbl)
                                    st.dataframe(_styled_table(tbl, f'{period_lbl} Growth'), use_container_width=True, hide_index=True)
                                else:
                                    st.info("Operating income data not available.")

                    st.markdown("---")

                    # ==========================================
                    # FINANCIAL RATIOS DASHBOARD
                    # D/E, ROE, ROA + solvency + valuation
                    # ==========================================
                    st.markdown("## 📐 Financial Ratios Dashboard")

                    ratios = fetch_financial_ratios(ticker_obj, fund)

                    st.markdown("### Core Efficiency & Leverage")
                    r1, r2, r3 = st.columns(3)

                    de      = ratios['de_ratio']
                    de_str  = f"{de:.2f}" if de is not None else "N/A"
                    de_sig  = ("⚠️ High Leverage" if de and de > 200
                               else "✅ Moderate"  if de and de <= 100
                               else "🔵 Low"       if de is not None else "—")
                    r1.metric("Debt / Equity (D/E)", de_str, delta=de_sig)

                    roe     = ratios['roe']
                    roe_str = f"{roe:.2%}" if roe is not None else "N/A"
                    roe_sig = ("✅ Strong"      if roe and roe > 0.15
                               else "⚠️ Weak"   if roe and roe < 0.05
                               else "🔵 Acceptable" if roe is not None else "—")
                    r2.metric("Return on Equity (ROE)", roe_str, delta=roe_sig)

                    roa     = ratios['roa']
                    roa_str = f"{roa:.2%}" if roa is not None else "N/A"
                    roa_sig = ("✅ Efficient"  if roa and roa > 0.05
                               else "⚠️ Poor"  if roa and roa < 0.01
                               else "🔵 Average" if roa is not None else "—")
                    r3.metric("Return on Assets (ROA)", roa_str, delta=roa_sig)

                    st.markdown("### Solvency & Liquidity")
                    s1, s2, s3 = st.columns(3)

                    cr     = ratios['current_ratio']
                    cr_str = f"{cr:.2f}" if cr is not None else "N/A"
                    cr_sig = ("✅ Healthy"   if cr and cr >= 1.5
                              else "⚠️ Tight" if cr and cr < 1.0
                              else "🔵 Adequate" if cr is not None else "—")
                    s1.metric("Current Ratio", cr_str, delta=cr_sig)

                    qr     = ratios['quick_ratio']
                    qr_str = f"{qr:.2f}" if qr is not None else "N/A"
                    qr_sig = ("✅ Strong"     if qr and qr >= 1.0
                              else "⚠️ At Risk" if qr and qr < 0.75
                              else "🔵 OK"      if qr is not None else "—")
                    s2.metric("Quick Ratio", qr_str, delta=qr_sig)

                    ic     = ratios['interest_coverage']
                    ic_str = f"{ic:.2f}x" if ic is not None else "N/A"
                    ic_sig = ("✅ Safe"      if ic and ic > 5
                              else "⚠️ Risky"  if ic and ic < 2
                              else "🔵 Moderate" if ic is not None else "—")
                    s3.metric("Interest Coverage", ic_str, delta=ic_sig)

                    st.markdown("### Profitability Margins")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Gross Margin",      f"{ratios['gross_margin']:.2%}" if ratios['gross_margin'] else "N/A")
                    m2.metric("Operating Margin",  f"{ratios['op_margin']:.2%}"    if ratios['op_margin']    else "N/A")
                    m3.metric("Net Profit Margin", f"{fund.get('profit_margin'):.2%}" if fund.get('profit_margin') else "N/A")

                    st.markdown("### Valuation Multiples")
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("P/B Ratio",      f"{ratios['pb_ratio']:.2f}x"       if ratios['pb_ratio']       else "N/A")
                    v2.metric("P/S Ratio",      f"{ratios['ps_ratio']:.2f}x"       if ratios['ps_ratio']       else "N/A")
                    v3.metric("EV / EBITDA",    f"{ratios['ev_ebitda']:.2f}x"      if ratios['ev_ebitda']      else "N/A")
                    v4.metric("Asset Turnover", f"{ratios['asset_turnover']:.3f}x" if ratios['asset_turnover'] else "N/A")

                    st.markdown("### Ratio Reference & Signal")
                    st.table(pd.DataFrame({
                        "Ratio":             ["D/E",   "ROE",    "ROA",    "Current Ratio", "Quick Ratio", "Interest Coverage", "Gross Margin",              "Op Margin"],
                        "Your Value":        [de_str,  roe_str,  roa_str,  cr_str,          qr_str,        ic_str,
                                              f"{ratios['gross_margin']:.2%}" if ratios['gross_margin'] else "N/A",
                                              f"{ratios['op_margin']:.2%}"    if ratios['op_margin']    else "N/A"],
                        "Healthy Benchmark": ["< 100 (equity)", "> 15%", "> 5%", "> 1.5", "> 1.0", "> 5x", "> 40% (sector)", "> 15%"],
                        "Signal":            [de_sig,  roe_sig,  roa_sig,  cr_sig,          qr_sig,        ic_sig,
                                              ("✅" if ratios['gross_margin'] and ratios['gross_margin'] > 0.4  else "🔵" if ratios['gross_margin'] else "—"),
                                              ("✅" if ratios['op_margin']    and ratios['op_margin']    > 0.15 else "🔵" if ratios['op_margin']    else "—")],
                    }))

                    st.markdown("---")

                    # Flag: analyst target vs +20% required
                    if fund['analyst_target'] and fund['analyst_target'] > 0:
                        pct_to_target = (fund['analyst_target'] - current_p) / current_p * 100
                        pct_to_20     = (buy_rec['sell_target'] - current_p) / current_p * 100
                        if pct_to_target < 20:
                            st.error(
                                f"🚨 Analyst 12M consensus ({fund['analyst_target']:.2f}) implies only "
                                f"{pct_to_target:.1f}% upside from current price. "
                                f"A +20% gain requires the stock to reach {buy_rec['sell_target']:.2f}. "
                                f"Analyst coverage does NOT support a +20% gain target."
                            )
                        else:
                            st.success(
                                f"✅ Analyst 12M consensus ({fund['analyst_target']:.2f}) implies "
                                f"{pct_to_target:.1f}% upside — above the +20% threshold. "
                                f"Fundamental coverage supports the profit target."
                            )

            except Exception as e:
                st.error(f"Execution Error: {e}")