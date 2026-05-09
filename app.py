import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PSX AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(140deg,#dbeafe 0%,#ede9fe 40%,#fce7f3 70%,#d1fae5 100%);
    background-attachment: fixed;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e3a5f 0%,#2d1b69 100%) !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.15) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
.main-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(90deg,#1e3a5f,#7c3aed,#be185d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.sub-title {
    text-align: center; color: #475569;
    font-size: 0.95rem; margin-bottom: 20px; font-weight: 500;
}
.sec-hdr {
    font-size: 1.1rem; font-weight: 800; color: #1e3a5f;
    border-left: 4px solid #7c3aed;
    padding-left: 10px; margin: 22px 0 14px 0;
}
.stat-card {
    background: #ffffff;
    border-radius: 16px; padding: 18px 12px; text-align: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.15); margin-bottom: 10px;
}
.stat-card .s-num { font-size: 2rem; font-weight: 900; color: #1e3a5f; }
.stat-card .s-lbl { font-size: 0.75rem; color: #64748b; margin-top: 3px; font-weight: 600; }

.ccard {
    background: #ffffff;
    border-radius: 16px; padding: 16px 10px; text-align: center;
    box-shadow: 0 4px 16px rgba(30,58,95,0.10);
    border: 1px solid rgba(30,58,95,0.10); margin-bottom: 10px;
}
.ccard .c-name  { color: #64748b; font-size: 0.72rem; font-weight: 700;
                  margin-bottom: 6px; text-transform: uppercase; }
.ccard .c-price { color: #1e3a5f; font-size: 1.25rem; font-weight: 900; }
.ccard .c-chg   { font-size: 0.8rem; font-weight: 700; margin-top: 4px; }

.b-buy  { background:#dcfce7; color:#166534; border:1.5px solid #16a34a;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-sell { background:#fee2e2; color:#991b1b; border:1.5px solid #dc2626;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-hold { background:#fef3c7; color:#92400e; border:1.5px solid #d97706;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }

.mini-card {
    background: #ffffff; border-radius: 12px; padding: 12px 10px;
    text-align: center; box-shadow: 0 2px 10px rgba(30,58,95,0.08);
    border: 1px solid rgba(30,58,95,0.08); margin-bottom: 8px;
}
.mini-card .m-lbl { color: #94a3b8; font-size: 0.68rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.mini-card .m-val { color: #1e293b; font-size: 1rem; font-weight: 800; }

.model-card {
    background: linear-gradient(135deg,#1e3a5f 0%,#7c3aed 100%);
    border-radius: 16px; padding: 20px; text-align: center;
    box-shadow: 0 6px 24px rgba(124,58,237,0.30); margin-bottom: 10px;
}
.model-card .mc-name { color: rgba(255,255,255,0.8); font-size: 0.82rem;
                       font-weight: 600; margin-bottom: 6px; }
.model-card .mc-val  { color: #ffffff; font-size: 2rem; font-weight: 900; }
.model-card .mc-lbl  { color: rgba(255,255,255,0.65); font-size: 0.72rem; margin-top: 3px; }

.rec-card {
    background: #ffffff; border-radius: 18px; padding: 22px 14px;
    text-align: center; box-shadow: 0 6px 24px rgba(30,58,95,0.12);
    border: 2px solid rgba(124,58,237,0.12); margin-bottom: 8px;
}
.rec-card .r-name   { color: #475569; font-size: 0.8rem; font-weight: 700; }
.rec-card .r-sector { color: #94a3b8; font-size: 0.68rem; margin-bottom: 8px; }
.rec-card .r-return { color: #7c3aed; font-size: 1.8rem; font-weight: 900; }
.rec-card .r-price  { color: #64748b; font-size: 0.8rem; margin-top: 4px; font-weight: 600; }

h1,h2,h3 { color: #1e3a5f !important; }
p, label, .stMarkdown p { color: #334155 !important; }
div[data-testid="stMetricValue"] { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

TICKERS = [
    "OGDC.KA","PSO.KA","HBL.KA","MCB.KA","LUCK.KA",
    "ENGRO.KA","FFC.KA","UBL.KA","NESTLE.KA","SYS.KA"
]
TICKER_NAMES = {
    "OGDC.KA":  "Oil & Gas Dev Co",
    "PSO.KA":   "Pakistan State Oil",
    "HBL.KA":   "Habib Bank",
    "MCB.KA":   "MCB Bank",
    "LUCK.KA":  "Lucky Cement",
    "ENGRO.KA": "Engro Corporation",
    "FFC.KA":   "Fauji Fertilizer",
    "UBL.KA":   "United Bank",
    "NESTLE.KA":"Nestle Pakistan",
    "SYS.KA":   "Systems Limited",
}
TICKER_SECTORS = {
    "OGDC.KA":"Energy",    "PSO.KA":"Energy",
    "HBL.KA":"Banking",    "MCB.KA":"Banking",
    "LUCK.KA":"Cement",    "ENGRO.KA":"Chemicals",
    "FFC.KA":"Fertilizer", "UBL.KA":"Banking",
    "NESTLE.KA":"FMCG",    "SYS.KA":"Technology",
}
FEATURES = [
    'SMA_20','SMA_50','RSI','MACD','BB_upper','BB_lower',
    'Lag_1','Lag_2','Lag_3','Lag_5',
    'Return_1d','Return_5','Volatility','SMA_ratio','BB_pos'
]
PAPER_BG = 'rgba(255,255,255,0)'
PLOT_BG  = 'rgba(255,255,255,0.85)'
GRID_CLR = 'rgba(30,58,95,0.07)'
TICK_CLR = '#64748b'
FONT_CLR = '#1e3a5f'

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📡 Loading PSX market data...")
def load_data():
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(
                t, start="2019-01-01", end="2024-12-31",
                auto_adjust=True, progress=False
            )
            if df is None or len(df) == 0:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cols = [c for c in ['Open','High','Low','Close','Volume']
                    if c in df.columns]
            df = df[cols].copy()
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > 60:
                data[t] = df
        except Exception:
            pass
    return data

# ── Feature engineering ───────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df['SMA_20']     = df['Close'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    delta            = df['Close'].diff()
    gain             = delta.clip(lower=0).rolling(14).mean()
    loss             = -delta.clip(upper=0).rolling(14).mean()
    df['RSI']        = 100 - (100 / (1 + gain / loss))
    ema12            = df['Close'].ewm(span=12).mean()
    ema26            = df['Close'].ewm(span=26).mean()
    df['MACD']       = ema12 - ema26
    df['BB_upper']   = df['SMA_20'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower']   = df['SMA_20'] - 2 * df['Close'].rolling(20).std()
    df['Return_1d']  = df['Close'].pct_change() * 100
    df['Volatility'] = df['Return_1d'].rolling(20).std()
    df['Lag_1']      = df['Close'].shift(1)
    df['Lag_2']      = df['Close'].shift(2)
    df['Lag_3']      = df['Close'].shift(3)
    df['Lag_5']      = df['Close'].shift(5)
    df['Return_5']   = df['Close'].pct_change(5)
    df['SMA_ratio']  = df['SMA_20'] / (df['SMA_50'] + 1e-9)
    bb_range         = (df['BB_upper'] - df['BB_lower']).replace(0, 1e-9)
    df['BB_pos']     = (df['Close'] - df['BB_lower']) / bb_range
    return df.dropna()

# ── Accuracy metric ───────────────────────────────────────────────────────────
def mape_accuracy(y_true, y_pred):
    mape = np.mean(np.abs(
        (np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-9)
    )) * 100
    return round(max(0.0, 100.0 - mape), 2)

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training AI models per stock...")
def train_models(_data):
    processed_dfs = []
    for ticker, raw_df in _data.items():
        try:
            feat_df = add_features(raw_df)
            feat_df['Ticker'] = ticker
            processed_dfs.append(feat_df)
        except Exception:
            pass

    if not processed_dfs:
        return pd.DataFrame(), {}, {}

    df_proc = pd.concat(processed_dfs)
    df_proc['Target'] = df_proc.groupby('Ticker')['Close'].shift(-1)
    df_proc.dropna(inplace=True)

    model_defs = {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_split=5, n_jobs=-1, random_state=SEED
        ),
        "XGBoost": lambda: GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=SEED
        ),
    }

    per_models = {n: {} for n in model_defs}
    results    = {}

    for mname, make_model in model_defs.items():
        accs, maes = [], []
        per_stock  = {}

        for ticker in df_proc['Ticker'].unique():
            try:
                tdf   = df_proc[df_proc['Ticker'] == ticker].copy()
                split = int(len(tdf) * 0.8)
                if split < 10 or (len(tdf) - split) < 5:
                    continue
                X_tr = tdf[FEATURES].iloc[:split]
                X_te = tdf[FEATURES].iloc[split:]
                y_tr = tdf['Target'].iloc[:split]
                y_te = tdf['Target'].iloc[split:]
                m    = make_model()
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                acc   = mape_accuracy(y_te.values, preds)
                mae   = mean_absolute_error(y_te, preds)
                per_models[mname][ticker] = m
                per_stock[ticker]         = acc
                accs.append(acc)
                maes.append(mae)
            except Exception:
                pass

        results[mname] = {
            "Acc":       round(float(np.mean(accs)),  1) if accs else 0.0,
            "MAE":       round(float(np.mean(maes)),  2) if maes else 0.0,
            "per_stock": per_stock,
        }

    return df_proc, per_models, results

# ── Signal helper (FIXED — balanced thresholds) ───────────────────────────────
def get_signal(ticker, model_name, df_processed, per_models):
    """
    Returns 'buy', 'sell', or 'hold' based on predicted return.
    Uses ±1% threshold for a balanced distribution of signals.
    """
    try:
        sd = df_processed[df_processed['Ticker'] == ticker]
        if len(sd) == 0:
            return "hold"

        # Try the selected model first, fall back to the other one
        m = per_models.get(model_name, {}).get(ticker)
        if m is None:
            other = "Random Forest" if model_name == "XGBoost" else "XGBoost"
            m = per_models.get(other, {}).get(ticker)
        if m is None:
            return "hold"

        latest = sd[FEATURES].iloc[-1:]
        if latest.isnull().any().any():
            return "hold"

        pp  = float(m.predict(latest.values)[0])
        lc  = float(sd['Close'].iloc[-1])
        if lc == 0:
            return "hold"

        ret = ((pp - lc) / lc) * 100

        # ±1% gives a realistic spread of buy/hold/sell
        if ret > 1.0:
            return "buy"
        elif ret < -1.0:
            return "sell"
        else:
            return "hold"
    except Exception:
        return "hold"

# ── Predicted return helper ───────────────────────────────────────────────────
def get_predicted_return(ticker, model_name, df_processed, per_models):
    """Returns the raw predicted % return for a ticker."""
    try:
        sd = df_processed[df_processed['Ticker'] == ticker]
        if len(sd) == 0:
            return 0.0
        m = per_models.get(model_name, {}).get(ticker)
        if m is None:
            return 0.0
        latest = sd[FEATURES].iloc[-1:]
        if latest.isnull().any().any():
            return 0.0
        pp  = float(m.predict(latest.values)[0])
        lc  = float(sd['Close'].iloc[-1])
        if lc == 0:
            return 0.0
        return round(((pp - lc) / lc) * 100, 2)
    except Exception:
        return 0.0

# ── Recommendation engine ─────────────────────────────────────────────────────
def recommend_stocks(watchlist, model_name, data, df_processed,
                     per_models, top_n=5):
    recs = []
    for ticker in data.keys():
        if ticker in watchlist:
            continue
        try:
            sd = df_processed[df_processed['Ticker'] == ticker]
            if len(sd) == 0:
                continue
            m = per_models.get(model_name, {}).get(ticker)
            if m is None:
                continue
            latest = sd[FEATURES].iloc[-1:]
            if latest.isnull().any().any():
                continue
            pp  = float(m.predict(latest.values)[0])
            lc  = float(sd['Close'].iloc[-1])
            if lc == 0:
                continue
            ret       = ((pp - lc) / lc) * 100
            rsi       = float(sd['RSI'].iloc[-1])
            rsi_score = 1.0 if rsi < 40 else (-1.0 if rsi > 60 else 0.0)
            vol       = sd['Close'].pct_change().std()
            vol       = 0.0 if pd.isna(vol) else float(vol)
            signal    = "Buy" if ret > 1.0 else ("Sell" if ret < -1.0 else "Hold")
            score     = ret * 0.6 + rsi_score * 0.3 - vol * 0.1
            recs.append({
                "ticker":           ticker,
                "name":             TICKER_NAMES.get(ticker, ticker),
                "sector":           TICKER_SECTORS.get(ticker, ""),
                "predicted_return": round(float(ret), 2),
                "signal":           signal,
                "rsi":              round(float(rsi), 1),
                "last_close":       round(float(lc), 2),
                "score":            round(float(score), 4),
            })
        except Exception:
            pass
    recs.sort(key=lambda x: x['score'], reverse=True)
    return recs[:top_n]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
data = load_data()

if len(data) == 0:
    st.error("❌ Could not load any stock data. Please refresh the page.")
    st.stop()

df_processed, per_models, results = train_models(data)

if df_processed is None or len(df_processed) == 0:
    st.error("❌ Feature engineering failed. Please refresh the page.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    available_tickers = [t for t in TICKERS if t in data]

    if not available_tickers:
        st.error("No tickers available.")
        st.stop()

    selected_stock = st.selectbox(
        "Select Company",
        options=available_tickers,
        format_func=lambda x: TICKER_NAMES.get(x, x)
    )
    selected_model = st.radio(
        "AI Model",
        options=["Random Forest", "XGBoost"],
        index=1
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.8rem;color:#cbd5e1;line-height:1.9'>
    📌 <b>PSX AI Stock Predictor</b><br>
    Final Year Project<br>
    Data: Yahoo Finance<br>
    Models: RF + XGBoost<br>
    Stocks: {len(data)} companies<br>
    Period: 2019–2024
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 PSX AI Stock Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Prediction & Recommendation System'
    ' for Top Pakistan Stock Exchange Companies</div>',
    unsafe_allow_html=True)

# ── Stats bar (FIXED — signals computed after models are loaded) ───────────────
signals = {t: get_signal(t, selected_model, df_processed, per_models)
           for t in data.keys()}

buy_c  = sum(1 for s in signals.values() if s == "buy")
sell_c = sum(1 for s in signals.values() if s == "sell")
hold_c = sum(1 for s in signals.values() if s == "hold")
best   = max(results, key=lambda x: results[x]['Acc']) if results else "XGBoost"
best_acc = results.get(best, {}).get('Acc', '—')

s_cols = st.columns(5)
for col, num, lbl in zip(
    s_cols,
    [len(data), buy_c, sell_c, hold_c, f"{best_acc}%"],
    ["🏢 Companies","🟢 Buy Signals","🔴 Sell Signals",
     "🟡 Hold Signals","⭐ Best Accuracy"]
):
    with col:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='s-num'>{num}</div>"
            f"<div class='s-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Company cards (FIXED — shows real signal per ticker) ──────────────────────
st.markdown('<div class="sec-hdr">🏢 PSX Companies — Live Snapshot</div>',
            unsafe_allow_html=True)

cols5 = st.columns(5)
for i, tick in enumerate(available_tickers):
    with cols5[i % 5]:
        sd = df_processed[df_processed['Ticker'] == tick]
        if len(sd) > 0:
            try:
                lc    = float(sd['Close'].iloc[-1])
                rsi   = float(sd['RSI'].iloc[-1])
                ret1d = float(sd['Return_1d'].iloc[-1])

                # Use AI predicted signal (not just RSI)
                sig   = signals.get(tick, "hold")
                arrow = "▲" if ret1d > 0 else "▼"
                clr   = "#15803d" if ret1d > 0 else "#dc2626"

                # Show predicted return too
                pred_ret = get_predicted_return(
                    tick, selected_model, df_processed, per_models)

                st.markdown(f"""
                <div class='ccard'>
                    <div class='c-name'>{TICKER_NAMES.get(tick, tick)}</div>
                    <div class='c-price'>PKR {lc:,.0f}</div>
                    <div class='c-chg' style='color:{clr}'>
                        {arrow} {abs(ret1d):.2f}%
                    </div>
                    <div style='color:#7c3aed;font-size:0.72rem;
                         font-weight:700;margin-top:3px;'>
                        AI: {pred_ret:+.2f}%
                    </div>
                    <div style='margin-top:5px'>
                        <span class='b-{sig}'>{sig.upper()}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                pass

st.markdown("<br>", unsafe_allow_html=True)

# ── Price chart ───────────────────────────────────────────────────────────────
left, right = st.columns([2, 1])

if selected_stock not in data:
    st.warning(f"⚠️ Data not available for "
               f"{TICKER_NAMES.get(selected_stock, selected_stock)}. "
               f"Please select another company.")
    st.stop()

sd  = df_processed[df_processed['Ticker'] == selected_stock].copy()
raw = data[selected_stock]

with left:
    st.markdown(
        f'<div class="sec-hdr">📊 '
        f'{TICKER_NAMES.get(selected_stock, selected_stock)} — Price Chart</div>',
        unsafe_allow_html=True)

    chart_type = st.radio("", ["Line", "Candlestick"], horizontal=True)
    fig = go.Figure()

    if chart_type == "Candlestick" and all(
            c in raw.columns for c in ['Open','High','Low','Close']):
        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw['Open'], high=raw['High'],
            low=raw['Low'], close=raw['Close'],
            increasing_line_color='#15803d',
            decreasing_line_color='#dc2626'))
    else:
        fig.add_trace(go.Scatter(
            x=sd.index, y=sd['Close'], name="Close Price",
            line=dict(color='#4f46e5', width=2.5),
            fill='tozeroy', fillcolor='rgba(79,70,229,0.07)'))

    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['SMA_20'], name="SMA 20",
        line=dict(dash='dash', color='#ea580c', width=1.5)))
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['SMA_50'], name="SMA 50",
        line=dict(dash='dot', color='#0891b2', width=1.5)))
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['BB_upper'], name="BB Upper",
        line=dict(color='rgba(220,38,38,0.4)', width=1)))
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['BB_lower'], name="BB Lower",
        line=dict(color='rgba(220,38,38,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(220,38,38,0.04)'))

    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=380,
        xaxis=dict(showgrid=False, color=TICK_CLR, linecolor=GRID_CLR),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        legend=dict(orientation="h", y=-0.22, font_color='#334155',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Volume chart
    if 'Volume' in raw.columns and 'Open' in raw.columns:
        vol_colors = [
            'rgba(21,128,61,0.6)' if c >= o else 'rgba(220,38,38,0.5)'
            for c, o in zip(raw['Close'], raw['Open'])
        ]
        fig_v = go.Figure(go.Bar(
            x=raw.index, y=raw['Volume'],
            marker_color=vol_colors, name="Volume"))
        fig_v.update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font_color=FONT_CLR, height=120,
            margin=dict(l=0, r=0, t=5, b=0), showlegend=False,
            xaxis=dict(showgrid=False, color=TICK_CLR),
            yaxis=dict(showgrid=False, color=TICK_CLR))
        st.plotly_chart(fig_v, use_container_width=True)

with right:
    # RSI gauge
    st.markdown('<div class="sec-hdr">📉 RSI Gauge</div>',
                unsafe_allow_html=True)
    rsi_now = float(sd['RSI'].iloc[-1]) if len(sd) > 0 else 50.0
    gc = "#15803d" if rsi_now < 40 else ("#dc2626" if rsi_now > 60 else "#d97706")

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_now,
        number={'font': {'color': '#1e3a5f', 'size': 36}},
        gauge={
            'axis':       {'range': [0,100], 'tickcolor':'#64748b',
                           'tickfont': {'color':'#64748b'}},
            'bar':        {'color': gc, 'thickness': 0.28},
            'bgcolor':    'rgba(255,255,255,0.9)',
            'bordercolor':'rgba(30,58,95,0.1)',
            'steps': [
                {'range':[0,  30],  'color':'rgba(21,128,61,0.12)'},
                {'range':[30, 70],  'color':'rgba(217,119,6,0.07)'},
                {'range':[70, 100], 'color':'rgba(220,38,38,0.12)'},
            ],
        },
        title={'text':"RSI Index", 'font':{'color':'#475569','size':14}}
    ))
    fig_g.update_layout(
        paper_bgcolor=PAPER_BG, font_color=FONT_CLR,
        height=240, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    # Key stats
    st.markdown('<div class="sec-hdr">📌 Key Stats</div>',
                unsafe_allow_html=True)
    try:
        lc     = float(sd['Close'].iloc[-1])
        raw_1y = raw.tail(252)
        hi52   = float(raw_1y['High'].max()) if 'High' in raw_1y else 0
        lo52   = float(raw_1y['Low'].min())  if 'Low'  in raw_1y else 0
        macd   = float(sd['MACD'].iloc[-1])
        vol20  = float(sd['Volatility'].iloc[-1])
        pred_r = get_predicted_return(
            selected_stock, selected_model, df_processed, per_models)
        ai_sig = signals.get(selected_stock, "hold").upper()

        stats = [
            ("Last Close",    f"PKR {lc:,.0f}"),
            ("AI Prediction", f"{pred_r:+.2f}%"),
            ("AI Signal",     ai_sig),
            ("52W High",      f"PKR {hi52:,.0f}"),
            ("52W Low",       f"PKR {lo52:,.0f}"),
            ("MACD",          f"{macd:.2f}"),
            ("Volatility",    f"{vol20:.2f}%"),
            ("Sector",        TICKER_SECTORS.get(selected_stock,'')),
        ]
        gc1, gc2 = st.columns(2)
        for j, (lb, vl) in enumerate(stats):
            with (gc1 if j % 2 == 0 else gc2):
                st.markdown(
                    f"<div class='mini-card'>"
                    f"<div class='m-lbl'>{lb}</div>"
                    f"<div class='m-val'>{vl}</div></div>",
                    unsafe_allow_html=True)
    except Exception:
        st.info("Stats unavailable for this ticker.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Model comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🤖 AI Model Performance Comparison</div>',
            unsafe_allow_html=True)

rf_res  = results.get("Random Forest", {"Acc":0,"MAE":0,"per_stock":{}})
xgb_res = results.get("XGBoost",       {"Acc":0,"MAE":0,"per_stock":{}})
winner  = ("XGBoost" if xgb_res.get('Acc',0) >= rf_res.get('Acc',0)
           else "Random Forest")
loser   = "Random Forest" if winner == "XGBoost" else "XGBoost"

w_acc = results.get(winner,{}).get('Acc',0)
l_acc = results.get(loser, {}).get('Acc',0)

# Winner banner
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#7c3aed);
     border-radius:16px; padding:18px 24px;
     display:flex; align-items:center; justify-content:space-between;
     margin-bottom:16px; box-shadow:0 6px 24px rgba(124,58,237,0.3);'>
  <div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.78rem;font-weight:600;
         text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;'>
         🏆 Best Performing Model</div>
    <div style='color:#fff;font-size:1.6rem;font-weight:900;'>{winner}</div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.85rem;margin-top:2px;'>
      Accuracy: <b style='color:#a5f3fc'>{w_acc}%</b>
      &nbsp;|&nbsp;
      MAE: <b style='color:#a5f3fc'>
        PKR {results.get(winner,{}).get('MAE',0):.0f}</b>
    </div>
  </div>
  <div style='text-align:right'>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;margin-bottom:4px;'>
        vs {loser}</div>
    <div style='color:#fde68a;font-size:1.2rem;font-weight:800;'>
      +{abs(w_acc - l_acc):.1f}% better
    </div>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;'>
        accuracy advantage</div>
  </div>
</div>""", unsafe_allow_html=True)

# 4 metric cards
mc = st.columns(4)
metrics_list = [
    ("Random Forest","Accuracy",
     f"{rf_res.get('Acc','—')}%",        "Higher is better"),
    ("Random Forest","MAE (Avg Error)",
     f"PKR {rf_res.get('MAE',0):.0f}",   "Lower is better"),
    ("XGBoost",      "Accuracy",
     f"{xgb_res.get('Acc','—')}%",       "Higher is better"),
    ("XGBoost",      "MAE (Avg Error)",
     f"PKR {xgb_res.get('MAE',0):.0f}",  "Lower is better"),
]
for i, (mname, lbl, val, hint) in enumerate(metrics_list):
    bdr = "border:2px solid #a5f3fc;" if mname == winner else ""
    with mc[i]:
        st.markdown(f"""<div class='model-card' style='{bdr}'>
            <div class='mc-name'>{mname}</div>
            <div class='mc-val'>{val}</div>
            <div class='mc-lbl'>{lbl} — {hint}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Per-stock accuracy chart
st.markdown('<div class="sec-hdr">📋 Per-Company Accuracy — RF vs XGBoost</div>',
            unsafe_allow_html=True)

rf_ps   = rf_res.get('per_stock',  {})
xgb_ps  = xgb_res.get('per_stock', {})
all_t   = sorted(set(list(rf_ps.keys()) + list(xgb_ps.keys())))
names_ps = [TICKER_NAMES.get(t, t) for t in all_t]

fig_per = go.Figure()
fig_per.add_trace(go.Bar(
    name='Random Forest', x=names_ps,
    y=[rf_ps.get(t, 0) for t in all_t],
    marker_color='#7c3aed', opacity=0.85,
    text=[f"{rf_ps.get(t,0):.1f}%" for t in all_t],
    textposition='outside', textfont=dict(size=10, color='#1e3a5f')))
fig_per.add_trace(go.Bar(
    name='XGBoost', x=names_ps,
    y=[xgb_ps.get(t, 0) for t in all_t],
    marker_color='#0891b2', opacity=0.85,
    text=[f"{xgb_ps.get(t,0):.1f}%" for t in all_t],
    textposition='outside', textfont=dict(size=10, color='#1e3a5f')))
fig_per.add_hline(
    y=90, line_dash="dash", line_color="#dc2626", line_width=1.5,
    annotation_text="Target: 90%",
    annotation_font_color="#dc2626", annotation_font_size=11)
fig_per.update_layout(
    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
    font_color=FONT_CLR, height=360, barmode='group',
    title=dict(text="Accuracy per Company — Random Forest vs XGBoost",
               font_color='#1e3a5f', font_size=13),
    yaxis=dict(range=[0,115], showgrid=True,
               gridcolor=GRID_CLR, color=TICK_CLR, title="Accuracy (%)"),
    xaxis=dict(showgrid=False, color=TICK_CLR, tickangle=-25,
               tickfont=dict(size=10, color='#1e3a5f')),
    legend=dict(font_color='#334155', bgcolor='rgba(255,255,255,0.8)',
                orientation='h', y=1.12),
    margin=dict(l=10, r=10, t=60, b=90))
st.plotly_chart(fig_per, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Head-to-head bar charts
st.markdown('<div class="sec-hdr">📊 Head-to-Head Charts</div>',
            unsafe_allow_html=True)
cc1, cc2 = st.columns(2)

with cc1:
    fig_cmp = go.Figure()
    for mname, clr, val in [
        ("Random Forest","#7c3aed", rf_res.get('Acc',0)),
        ("XGBoost",      "#0891b2", xgb_res.get('Acc',0))
    ]:
        fig_cmp.add_trace(go.Bar(
            name=mname, x=['Accuracy (%)'], y=[val],
            marker_color=clr, text=[f"{val}%"],
            textposition='outside',
            textfont=dict(color='#1e3a5f', size=13), width=0.3))
    fig_cmp.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=280, barmode='group',
        title=dict(text="Accuracy Comparison (%)",
                   font_color='#1e3a5f', font_size=13),
        yaxis=dict(range=[0,100], showgrid=True,
                   gridcolor=GRID_CLR, color=TICK_CLR),
        xaxis=dict(showgrid=False, color=TICK_CLR),
        legend=dict(font_color='#334155',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=40, b=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

with cc2:
    fig_mae = go.Figure()
    for mname, clr, val in [
        ("Random Forest","#7c3aed", rf_res.get('MAE',0)),
        ("XGBoost",      "#0891b2", xgb_res.get('MAE',0))
    ]:
        fig_mae.add_trace(go.Bar(
            name=mname, x=['MAE (PKR)'], y=[val],
            marker_color=clr, text=[f"PKR {val:.0f}"],
            textposition='outside',
            textfont=dict(color='#1e3a5f', size=13), width=0.3))
    fig_mae.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=280, barmode='group',
        title=dict(text="MAE — lower is better",
                   font_color='#1e3a5f', font_size=13),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        xaxis=dict(showgrid=False, color=TICK_CLR),
        legend=dict(font_color='#334155',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=40, b=10))
    st.plotly_chart(fig_mae, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-hdr">💡 Top 5 Stock Recommendations — {selected_model}</div>',
    unsafe_allow_html=True)
st.markdown(
    f"<span style='color:#64748b;font-size:0.84rem;font-weight:600'>"
    f"Excluding: {TICKER_NAMES.get(selected_stock, selected_stock)}</span>",
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

recs = recommend_stocks(
    [selected_stock], selected_model,
    data, df_processed, per_models
)

if recs:
    rcols = st.columns(min(5, len(recs)))
    for i, r in enumerate(recs):
        with rcols[i % len(rcols)]:
            st.markdown(f"""
            <div class='rec-card'>
                <div class='r-name'>{r['name']}</div>
                <div class='r-sector'>{r['sector']}</div>
                <div class='r-return'>{r['predicted_return']}%</div>
                <div class='r-price'>PKR {r['last_close']:,.0f}</div>
                <div style='margin-top:10px'>
                    <span class='b-{r["signal"].lower()}'>{r['signal']}</span>
                </div>
                <div style='color:#94a3b8;font-size:0.72rem;
                     margin-top:6px;font-weight:600'>
                    RSI {r['rsi']}
                </div>
            </div>""", unsafe_allow_html=True)
else:
    st.info("No recommendations available. Try selecting a different company.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;
     padding:10px 0 20px 0;font-weight:500'>
📌 PSX AI Stock Predictor &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp;
Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; Data: Yahoo Finance<br>
⚠️ For educational purposes only. Not financial advice.
</div>""", unsafe_allow_html=True)
