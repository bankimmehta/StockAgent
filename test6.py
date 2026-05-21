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
    All values in billions (USD/INR) for display.
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
        g = []
        for i in range(len(series)):
            if i == 0 or pd.isna(series[i]) or pd.isna(series[i-1]) or series[i-1] == 0:
                g.append(None)
            else:
                g.append(((series[i] - series[i-1]) / abs(series[i-1])) * 100)
        return g

    try:
        # 1. Quarterly Data
        q_inc = ticker_obj.quarterly_income_stmt
        if q_inc is not None and not q_inc.empty:
            q_inc_sorted = q_inc.iloc[:, ::-1].tail(8)
            q_cols = q_inc_sorted.columns
            rev_q = q_inc_sorted.loc['Total Revenue'] if 'Total Revenue' in q_inc_sorted.index else pd.Series(index=q_cols)
            net_q = q_inc_sorted.loc['Net Income'] if 'Net Income' in q_inc_sorted.index else pd.Series(index=q_cols)
            op_q  = q_inc_sorted.loc['Operating Income'] if 'Operating Income' in q_inc_sorted.index else pd.Series(index=q_cols)

            result['quarterly'] = {
                'labels':     [_format_label_q(d) for d in q_cols],
                'revenue':    _to_billions(rev_q),
                'rev_growth': _growth(rev_q),
                'net_income': _to_billions(net_q),
                'net_growth': _growth(net_q),
                'op_income':  _to_billions(op_q),
            }

        # 2. Annual Data
        y_inc = ticker_obj.income_stmt
        if y_inc is not None and not y_inc.empty:
            y_inc_sorted = y_inc.iloc[:, ::-1].tail(5)
            y_cols = y_inc_sorted.columns
            rev_y = y_inc_sorted.loc['Total Revenue'] if 'Total Revenue' in y_inc_sorted.index else pd.Series(index=y_cols)
            net_y = y_inc_sorted.loc['Net Income'] if 'Net Income' in y_inc_sorted.index else pd.Series(index=y_cols)
            op_y  = y_inc_sorted.loc['Operating Income'] if 'Operating Income' in y_inc_sorted.index else pd.Series(index=y_cols)

            result['annual'] = {
                'labels':     [_format_label_y(d) for d in y_cols],
                'revenue':    _to_billions(rev_y),
                'rev_growth': _growth(rev_y),
                'net_income': _to_billions(net_y),
                'net_growth': _growth(net_y),
                'op_income':  _to_billions(op_y),
            }

    except Exception as e:
        result['error'] = str(e)

    return result


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
    def calculate(self, df):
        df['SMA_50']  = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        return df

    def get_summary(self, df):
        p   = df['Close'].iloc[-1]
        s50 = df['SMA_50'].iloc[-1]
        s200= df['SMA_200'].iloc[-1]
        if pd.isna(s50) or pd.isna(s200): return "MAs unavailable."
        t1  = "Above 50MA" if p > s50 else "Below 50MA"
        t2  = "Above 200MA" if p > s200 else "Below 200MA"
        t3  = "Golden Cross" if s50 > s200 else "Death Cross"
        return f"Price: {t1}, {t2} | Structural Trend: {t3}"


class ATRIndicator:
    def calculate(self, df):
        h_l = df['High'] - df['Low']
        h_pc = (df['High'] - df['Close'].shift(1)).abs()
        l_pc = (df['Low'] - df['Close'].shift(1)).abs()
        tr  = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        return df

    def get_summary(self, df):
        val = df['ATR'].iloc[-1]
        if pd.isna(val): return "ATR unavailable."
        pct = (val / df['Close'].iloc[-1]) * 100
        return f"ATR (14D): {val:.2f} ({pct:.1f}% of Price)"


# ==========================================
# 2. RISK / EXECUTION MATRIX
# ==========================================

class BreakoutStrategyEngine:
    def compute_setup(self, df):
        if len(df) < 50: return None
        h50 = df['High'].iloc[-51:-1].max()
        l50 = df['Low'].iloc[-51:-1].min()
        cp  = df['Close'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        triggered = cp > h50
        risk_dist = cp - l50
        r_r_ratio = (cp * 0.20) / risk_dist if risk_dist > 0 else 0

        # Adjust tracking flags for stop targets
        sl_atr = cp - (2 * atr)

        return {
            'h50': h50, 'l50': l50, 'cp': cp,
            'triggered': triggered,
            'risk_dist_pct': (risk_dist / cp) * 100,
            'r_r_ratio': r_r_ratio,
            'sell_target': cp * 1.20,
            'stop_loss_hard': l50,
            'stop_loss_atr':  sl_atr,
        }


# ==========================================
# 3. GEN-AI INTEGRATION
# ==========================================

class GoogleGenAIClient:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
        else:
            self.llm = None

    def generate_report(self, ticker, summaries, buy_rec, financials):
        if not self.llm:
            return "⚠️ Gemini API key is missing. Set 'GEMINI_API_KEY' in your Environment/Secrets configurations."

        prompt = f"""
        You are an elite institutional risk manager and technical analyst.
        Analyze the following data for asset '{ticker}' and output an institutional grade summary.
        
        TECHNICAL STANDINGS:
        - {summaries.get('rsi','')}
        - {summaries.get('sma','')}
        - {summaries.get('atr','')}
        
        EXECUTION ALIGNMENT:
        - Is 50-Day Breakout Active?: {buy_rec['triggered']}
        - Current Close: {buy_rec['cp']:.2f} (Target Entry)
        - Structural 50D Peak (Trigger Line): {buy_rec['h50']:.2f}
        - Profit Target (+20% required): {buy_rec['sell_target']:.2f}
        - Primary Hard Stop (50D Low): {buy_rec['stop_loss_hard']:.2f}
        - Volatility Stop (2-ATR trailing context): {buy_rec['stop_loss_atr']:.2f}
        - Risk-to-Reward Ratio vs +20% Target: {buy_rec['r_r_ratio']:.2f}
        
        FINANCIAL STATEMENTS BRIEF (Billions Context):
        - Quarterly Profiles: {financials.get('quarterly', 'No Q data')}
        - Annual Profiles: {financials.get('annual', 'No Y data')}

        Provide an objective evaluation. Separate your response into distinct sections:
        1. Executive Risk/Reward Assessment
        2. Structural Trend Context
        3. Fundamental Health Alignment (Verify if financial metrics match technical actions)
        4. Strict Go/No-Go Verdict
        Keep it direct, professional, and completely devoid of fluff.
        """
        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"Failed to generate AI executive summary: {e}"


# ==========================================
# 4. INDIA EXCHANGES / SCREENER DATA ENGINE
# ==========================================

class ScreenerIndiaEngine:
    """
    Simulates or pulls standard fundamental screening blocks native to Screener.in layouts 
    via public TradingView standard database mappings for Indian Exchanges (NSE/BSE).
    """
    def fetch_fundamental_ratios(self, ticker_str):
        # Default fallback structure
        data = {
            'pe_ratio': None, 'pb_ratio': None, 'roe': None, 'roce': None,
            'debt_to_equity': None, 'market_cap': None, 'analyst_target': None,
            'op_margin': None, 'div_yield': None, 'source': 'Fallback Data Extraction'
        }
        
        clean_ticker = ticker_str.split('.')[0].upper()
        
        try:
            # Query Indian Equity Space across NSE and BSE via the screener provider mapping
            q = (Query()
                 .set_markets('india')
                 .select('name', 'close', 'earnings_per_share_basic_ttm', 'price_earnings_ttm', 
                         'price_book_ttm', 'return_on_equity_fy', 'return_on_capital_employed_by_operating_items_fy',
                         'debt_to_equity_fy', 'market_cap_basic', 'target_price_analyst_mean', 
                         'operating_margin_ttm', 'dividend_yield_recent_dividend_payment')
                 .where(Column('name').has(clean_ticker))
                 .limit(1))
            
            res = q.get_scanner_data()
            if len(res) > 1 and not res[1].empty:
                row = res[1].iloc[0]
                data['pe_ratio']        = row.get('price_earnings_ttm')
                data['pb_ratio']        = row.get('price_book_ttm')
                data['roe']             = row.get('return_on_equity_fy')
                data['roce']            = row.get('return_on_capital_employed_by_operating_items_fy')
                data['debt_to_equity']  = row.get('debt_to_equity_fy')
                data['market_cap']      = row.get('market_cap_basic')
                data['analyst_target']  = row.get('target_price_analyst_mean')
                data['op_margin']       = row.get('operating_margin_ttm')
                data['div_yield']       = row.get('dividend_yield_recent_dividend_payment')
                data['source']          = 'TradingView Indian Matrix (Screener.in Proxy)'
                return data
        except Exception:
            pass

        # Secondary native fallback parsing method from standard attributes
        try:
            t = yf.Ticker(ticker_str)
            inf = t.info
            data['pe_ratio']       = inf.get('trailingPE') or inf.get('forwardPE')
            data['pb_ratio']       = inf.get('priceToBook')
            data['roe']            = (inf.get('returnOnEquity') * 100) if inf.get('returnOnEquity') else None
            data['debt_to_equity'] = inf.get('debtToEquity')
            data['market_cap']     = inf.get('marketCap')
            data['analyst_target'] = inf.get('targetMeanPrice')
            data['op_margin']      = inf.get('operatingMargins')
            data['div_yield']      = (inf.get('dividendYield') * 100) if inf.get('dividendYield') else None
        except Exception:
            pass
            
        return data


# ==========================================
# 5. STREAMLIT APPLICATION INTERFACE
# ==========================================

st.set_page_config(page_title="Institutional Edge Dashboard", layout="wide")
st.title("📊 Institutional Risk & Breakout Engine")

# App input configuration
symbol = st.text_input("Enter Asset Ticker (e.g., RELIANCE.NS, TCS.NS, AAPL, GC=F):", "RELIANCE.NS")

if st.button("Run Comprehensive Analysis"):
    with st.spinner("Processing technical indicators and extracting fundamental tracking tables..."):
        
        # Pull core pricing array
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="1y")

        if hist.empty:
            st.error("No historical data returned. Please verify spelling or data provider routing matrix.")
        elif len(hist) < 50:
            st.error(f"Insufficient historical timeline. Found {len(hist)} periods; engine rules require at least 50.")
        else:
            # Run analytics sequence
            rsi_eng = RSIIndicator()
            sma_eng = SMAIndicator()
            atr_eng = ATRIndicator()

            hist = rsi_eng.calculate(hist)
            hist = sma_eng.calculate(hist)
            hist = atr_eng.calculate(hist)

            sums = {
                'rsi': rsi_eng.get_summary(hist),
                'sma': sma_eng.get_summary(hist),
                'atr': atr_eng.get_summary(hist)
            }

            strat = BreakoutStrategyEngine()
            buy_rec = strat.compute_setup(hist)

            fin_data = fetch_financial_statements(ticker_obj)
            
            # Fetch Fundamental Screener Engine mapping
            scr_eng = ScreenerIndiaEngine()
            fund = scr_eng.fetch_fundamental_ratios(symbol)

            # =========================================================================
            # DISPLAY SCREEN 1: TECHNICAL ANALYSIS & CHARTING
            # =========================================================================
            st.subheader("📈 Technical Posture & Breakout Vectors")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1: st.metric("RSI (14D)", f"{hist['RSI'].iloc[-1]:.2f}", sums['rsi'].split('(')[-1].replace(')', ''))
            with col_t2: st.metric("Current Stop Context", f"{buy_rec['stop_loss_atr']:.2f}", "2-ATR Trailing Bound")
            with col_t3: st.metric("50D Peak Resistance", f"{buy_rec['h50']:.2f}", "Trigger Baseline")

            # Chart Construction
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price', line=dict(color='#1f77b4', width=2)))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='50 SMA', line=dict(color='#ff7f0e', width=1.5, dash='dash')))
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='200 SMA', line=dict(color='#2ca02c', width=1.5, dash='dot')))
            
            # Trigger overlay line
            fig.add_shape(type="line", x0=hist.index[0], y0=buy_rec['h50'], x1=hist.index[-1], y1=buy_rec['h50'],
                          line=dict(color="Red", width=1, dash="dashdot"))

            fig.update_layout(title=f"{symbol} Tactical Price Mapping", template="plotly_dark", 
                              xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Execution Alerts
            if buy_rec['triggered']:
                st.success(f"🚀 **Active 50-Day Breakout Detected!** Price closed above physical resistance layer at {buy_rec['h50']:.2f}")
            else:
                st.warning(f"🔒 **Consolidation Bound.** Current close sits below the structural trigger line of {buy_rec['h50']:.2f}")

            # =========================================================================
            # DISPLAY SCREEN 2: SYSTEM GEN-AI RISK ASSESSMENT
            # =========================================================================
            st.subheader("🤖 Institutional AI Intelligence Directive")
            ai_eng = GoogleGenAIClient()
            report = ai_eng.generate_report(symbol, sums, buy_rec, fin_data)
            st.markdown(report)

            # =========================================================================
            # DISPLAY SCREEN 3: HIGHLY READABLE & PRESENTABLE FINANCIAL DASHBOARD
            # =========================================================================
            st.markdown("---")
            st.subheader("📋 Screener.in Corporate Fundamental Dashboard")
            st.caption(f"Data Source Pipeline: {fund['source']}")

            # Formatting functions for the presentable card widgets
            def fmt_val(val, suffix="", prefix="", digits=2):
                return f"{prefix}{val:,.{digits}f}{suffix}" if val is not None and not pd.isna(val) else "N/A"

            # Block A: Key Financial Ratios displayed as key metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(label="📊 Trailing P/E", value=fmt_val(fund['pe_ratio']))
            with m_col2:
                st.metric(label="📖 Price to Book (P/B)", value=fmt_val(fund['pb_ratio']))
            with m_col3:
                roe_val = fund['roe']
                roe_str = fmt_val(roe_val, suffix="%")
                st.metric(label="🎯 Return on Equity (ROE)", value=roe_str)
            with m_col4:
                debt_str = fmt_val(fund['debt_to_equity'], digits=2)
                st.metric(label="⚖️ Debt to Equity", value=debt_str)

            m_col5, m_col6, m_col7, m_col8 = st.columns(4)
            with m_col5:
                st.metric(label="⚙️ Operating Margin", value=fmt_val(fund['op_margin'] * 100 if fund['op_margin'] and fund['op_margin'] < 1 else fund['op_margin'], suffix="%"))
            with m_col6:
                st.metric(label="💸 Dividend Yield", value=fmt_val(fund['div_yield'], suffix="%"))
            with m_col7:
                # Market Cap formatting
                mc = fund['market_cap']
                if mc and mc > 1e11:
                    mc_str = f"₹{mc/1e11:.2f} L Cr" if ".NS" in symbol or ".BO" in symbol else f"${mc/1e9:.2f} B"
                elif mc and mc > 1e7:
                    mc_str = f"₹{mc/1e7:.2f} Cr" if ".NS" in symbol or ".BO" in symbol else f"${mc/1e6:.2f} M"
                else:
                    mc_str = fmt_val(mc, prefix="₹" if ".NS" in symbol or ".BO" in symbol else "$")
                st.metric(label="🏦 Market Capitalization", value=mc_str)
            with m_col8:
                st.metric(label="🎯 Consensus Target", value=fmt_val(fund['analyst_target'], prefix="₹" if ".NS" in symbol or ".BO" in symbol else "$"))

            # Block B: Clean Historical Statements Tables
            tab_q, tab_y = st.tabs(["🕒 Quarterly Performance (Last 8 Quarters)", "📅 Annual Performance (Last 5 Years)"])
            
            with tab_q:
                if fin_data['quarterly']:
                    q_df = pd.DataFrame({
                        "Quarter": fin_data['quarterly']['labels'],
                        "Revenue (Billion)": [fmt_val(v, digits=3) for v in fin_data['quarterly']['revenue']],
                        "Revenue QoQ Growth": [fmt_val(v, suffix="%") for v in fin_data['quarterly']['rev_growth']],
                        "Net Income (Billion)": [fmt_val(v, digits=3) for v in fin_data['quarterly']['net_income']],
                        "Net Income QoQ Growth": [fmt_val(v, suffix="%") for v in fin_data['quarterly']['net_growth']],
                        "Operating Income (Billion)": [fmt_val(v, digits=3) for v in fin_data['quarterly']['op_income']]
                    })
                    st.dataframe(q_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Quarterly income statements unavailable or missing for this asset architecture.")

            with tab_y:
                if fin_data['annual']:
                    y_df = pd.DataFrame({
                        "Year": fin_data['annual']['labels'],
                        "Revenue (Billion)": [fmt_val(v, digits=3) for v in fin_data['annual']['revenue']],
                        "Revenue YoY Growth": [fmt_val(v, suffix="%") for v in fin_data['annual']['rev_growth']],
                        "Net Income (Billion)": [fmt_val(v, digits=3) for v in fin_data['annual']['net_income']],
                        "Net Income YoY Growth": [fmt_val(v, suffix="%") for v in fin_data['annual']['net_growth']],
                        "Operating Income (Billion)": [fmt_val(v, digits=3) for v in fin_data['annual']['op_income']]
                    })
                    st.dataframe(y_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Annual financial history data block unavailable for this asset architecture.")

            # Block C: Risk / Target Target Coverage Warnings
            st.markdown("### 🎯 Target Validation Check")
            current_p = buy_rec['cp']
            if fund['analyst_target'] and fund['analyst_target'] > 0:
                pct_to_target = (fund['analyst_target'] - current_p) / current_p * 100
                pct_to_20     = (buy_rec['sell_target'] - current_p) / current_p * 100
                
                if pct_to_target < 20:
                    st.error(
                        f"🚨 **Analyst Consensus Warning:** The 12M consensus target ({fund['analyst_target']:.2f}) implies only "
                        f"**{pct_to_target:.1f}% upside** from the current close. A structural +20% gain target requires "
                        f"reaching **{buy_rec['sell_target']:.2f}**. Fundamental analyst consensus coverage does **NOT** support your profit projection window."
                    )
                else:
                    st.success(
                        f"✅ **Analyst Consensus Confirmed:** The 12M consensus target ({fund['analyst_target']:.2f}) implies an upside of "
                        f"**{pct_to_target:.1f}%**—comfortably exceeding your +20% strategy milestone ({buy_rec['sell_target']:.2f}). Fundamentals match the breakout view."
                    )
            else:
                st.info("💡 Consensus target boundary conditions not active or tracked for this particular asset ticker index.")