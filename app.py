import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}
.main-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(90deg,#1e3a5f,#7c3aed,#be185d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.sub-title {
    text-align: center; color: #475569;
    font-size: 0.95rem; margin-bottom: 24px; font-weight: 500;
}
.sec-hdr {
    font-size: 1.1rem; font-weight: 800; color: #1e3a5f;
    border-left: 4px solid #7c3aed;
    padding-left: 10px; margin: 24px 0 14px 0;
}

/* ── STAT CARDS ── */
.stat-card {
    background: #ffffff;
    border-radius: 18px; padding: 20px 10px; text-align: center;
    box-shadow: 0 4px 24px rgba(124,58,237,0.13);
    border: 1.5px solid rgba(124,58,237,0.13); margin-bottom: 10px;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-card .s-icon { font-size: 1.6rem; margin-bottom: 6px; }
.stat-card .s-num  { font-size: 2.1rem; font-weight: 900; color: #1e3a5f; line-height:1; }
.stat-card .s-lbl  { font-size: 0.73rem; color: #64748b; margin-top: 5px;
                     font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-buy  { border-color: #16a34a !important; }
.stat-sell { border-color: #dc2626 !important; }
.stat-hold { border-color: #d97706 !important; }
.stat-acc  { border-color: #7c3aed !important;
             background: linear-gradient(135deg,#f5f3ff,#ede9fe) !important; }
.stat-acc .s-num { color: #7c3aed !important; }

/* ── COMPANY CARDS ── */
.ccard {
    background: #ffffff; border-radius: 16px; padding: 16px 10px;
    text-align: center; box-shadow: 0 4px 16px rgba(30,58,95,0.10);
    border: 1px solid rgba(30,58,95,0.10); margin-bottom: 10px;
}
.ccard .c-name  { color: #64748b; font-size: 0.70rem; font-weight: 700;
                  margin-bottom: 6px; text-transform: uppercase; letter-spacing:0.3px; }
.ccard .c-price { color: #1e3a5f; font-size: 1.2rem; font-weight: 900; }
.ccard .c-chg   { font-size: 0.78rem; font-weight: 700; margin-top: 3px; }
.ccard .c-ai    { color: #7c3aed; font-size: 0.72rem; font-weight: 700; margin-top: 3px; }

/* ── SIGNAL BADGES ── */
.b-buy  { background:#dcfce7; color:#166534; border:1.5px solid #16a34a;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-sell { background:#fee2e2; color:#991b1b; border:1.5px solid #dc2626;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-hold { background:#fef3c7; color:#92400e; border:1.5px solid #d97706;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }

/* ── MINI STATS ── */
.mini-card {
    background: #ffffff; border-radius: 12px; padding: 11px 8px;
    text-align: center; box-shadow: 0 2px 10px rgba(30,58,95,0.08);
    border: 1px solid rgba(30,58,95,0.08); margin-bottom: 8px;
}
.mini-card .m-lbl { color: #94a3b8; font-size: 0.65rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }
.mini-card .m-val { color: #1e293b; font-size: 0.95rem; font-weight: 800; }

/* ── MODEL CARDS ── */
.model-card {
    background: linear-gradient(135deg,#1e3a5f 0%,#7c3aed 100%);
    border-radius: 16px; padding: 20px; text-align: center;
    box-shadow: 0 6px 24px rgba(124,58,237,0.30); margin-bottom: 10px;
}
.model-card .mc-name { color:rgba(255,255,255,0.8); font-size:0.82rem;
                       font-weight:600; margin-bottom:6px; }
.model-card .mc-val  { color:#ffffff; font-size:2rem; font-weight:900; }
.model-card .mc-lbl  { color:rgba(255,255,255,0.65); font-size:0.72rem; margin-top:3px; }

/* ── REC CARDS ── */
.rec-card {
    background: #ffffff; border-radius: 18px; padding: 22px 14px;
    text-align: center; box-shadow: 0 6px 24px rgba(30,58,95,0.12);
    border: 2px solid rgba(124,58,237,0.12); margin-bottom: 8px;
}
.rec-card .r-name   { color:#475569; font-size:0.8rem; font-weight:700; }
.rec-card .r-sector { color:#94a3b8; font-size:0.68rem; margin-bottom:8px; }
.rec-card .r-return { color:#7c3aed; font-size:1.8rem; font-weight:900; }
.rec-card .r-price  { color:#64748b; font-size:0.8rem; margin-top:4px; font-weight:600; }

h1,h2,h3 { color:#1e3a5f !important; }
p, label, .stMarkdown p { color:#334155 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)

TICKERS = [
    "OGDC.KA","PSO.KA","HBL.KA","MCB.KA","LUCK.KA",
    "ENGRO.KA","FFC.KA","UBL.KA","NESTLE.KA","SYS.KA"
]
TICKER_NAMES = {
    "OGDC.KA":   "Oil & Gas Dev Co",
    "PSO.KA":    "Pakistan State Oil",
    "HBL.KA":    "Habib Bank",
    "MCB.KA":    "MCB Bank",
    "LUCK.KA":   "Lucky Cement",
    "ENGRO.KA":  "Engro Corporation",
    "FFC.KA":    "Fauji Fertilizer",
    "UBL.KA":    "United Bank",
    "NESTLE.KA": "Nestle Pakistan",
    "SYS.KA":    "Systems Limited",
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

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def mape_accuracy(y_true, y_pred):
    mape = np.mean(np.abs(
        (np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-9)
    )) * 100
    return round(max(0.0, 100.0 - mape), 2)

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

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL & PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_predicted_return(ticker, model_name, df_processed, per_models):
    """Returns predicted % change for next day."""
    try:
        sd = df_processed[df_processed['Ticker'] == ticker]
        if len(sd) == 0:
            return 0.0
        m = per_models.get(model_name, {}).get(ticker)
        if m is None:
            other = "Random Forest" if model_name == "XGBoost" else "XGBoost"
            m = per_models.get(other, {}).get(ticker)
        if m is None:
            return 0.0
        latest = sd[FEATURES].iloc[-1:]
        if latest.isnull().any().any():
            return 0.0
        pp = float(m.predict(latest.values)[0])
        lc = float(sd['Close'].iloc[-1])
        if lc == 0:
            return 0.0
        return round(((pp - lc) / lc) * 100, 2)
    except Exception:
        return 0.0

def classify_signal(ret: float) -> str:
    """Convert predicted return % to buy/sell/hold string."""
    if ret > 1.0:
        return "buy"
    elif ret < -1.0:
        return "sell"
    else:
        return "hold"

def build_signals(data, model_name, df_processed, per_models):
    """
    Build a dict  {ticker: {"ret": float, "signal": str}}
    for ALL loaded tickers at once.
    This is called ONCE after training and reused everywhere —
    so stats bar, company cards and recommendations are always consistent.
    """
    out = {}
    for ticker in data.keys():
        ret    = get_predicted_return(ticker, model_name, df_processed, per_models)
        signal = classify_signal(ret)
        out[ticker] = {"ret": ret, "signal": signal}
    return out

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
def recommend_stocks(watchlist, signals_dict, df_processed, top_n=5):
    recs = []
    for ticker, info in signals_dict.items():
        if ticker in watchlist:
            continue
        try:
            sd = df_processed[df_processed['Ticker'] == ticker]
            if len(sd) == 0:
                continue
            lc  = float(sd['Close'].iloc[-1])
            rsi = float(sd['RSI'].iloc[-1])
            ret = info["ret"]
            rsi_score = 1.0 if rsi < 40 else (-1.0 if rsi > 60 else 0.0)
            vol = sd['Close'].pct_change().std()
            vol = 0.0 if pd.isna(vol) else float(vol)
            score = ret * 0.6 + rsi_score * 0.3 - vol * 0.1
            recs.append({
                "ticker":           ticker,
                "name":             TICKER_NAMES.get(ticker, ticker),
                "sector":           TICKER_SECTORS.get(ticker, ""),
                "predicted_return": round(float(ret), 2),
                "signal":           info["signal"].capitalize(),
                "rsi":              round(float(rsi), 1),
                "last_close":       round(float(lc), 2),
                "score":            round(float(score), 4),
            })
        except Exception:
            pass
    recs.sort(key=lambda x: x['score'], reverse=True)
    return recs[:top_n]

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
data = load_data()
if len(data) == 0:
    st.error("❌ Could not load any stock data. Please refresh the page.")
    st.stop()

df_processed, per_models, results = train_models(data)
if df_processed is None or len(df_processed) == 0:
    st.error("❌ Training failed. Please refresh the page.")
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
    <div style='font-size:0.8rem;color:#cbd5e1;line-height:1.9;'>
    📌 <b>PSX AI Stock Predictor</b><br>
    Final Year Project<br>
    Data: Yahoo Finance<br>
    Models: RF + XGBoost<br>
    Stocks: {len(data)} companies<br>
    Period: 2019–2024
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD SIGNALS — single source of truth, used everywhere below
# ══════════════════════════════════════════════════════════════════════════════
all_signals = build_signals(data, selected_model, df_processed, per_models)

buy_tickers  = [t for t,v in all_signals.items() if v["signal"] == "buy"]
sell_tickers = [t for t,v in all_signals.items() if v["signal"] == "sell"]
hold_tickers = [t for t,v in all_signals.items() if v["signal"] == "hold"]

buy_c  = len(buy_tickers)
sell_c = len(sell_tickers)
hold_c = len(hold_tickers)

best_model = max(results, key=lambda x: results[x]['Acc']) if results else "XGBoost"
best_acc   = results.get(best_model, {}).get('Acc', 0)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 PSX AI Stock Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Prediction & Recommendation System'
    ' for Top Pakistan Stock Exchange Companies</div>',
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STATS BAR  ←  THE FIXED SECTION
# Shows: Companies | Buy | Sell | Hold | Best Accuracy
# All numbers come from all_signals dict built after training completes
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class='stat-card'>
        <div class='s-icon'>🏢</div>
        <div class='s-num'>{len(data)}</div>
        <div class='s-lbl'>Companies</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='stat-card stat-buy'>
        <div class='s-icon'>🟢</div>
        <div class='s-num' style='color:#16a34a'>{buy_c}</div>
        <div class='s-lbl'>Buy Signals</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='stat-card stat-sell'>
        <div class='s-icon'>🔴</div>
        <div class='s-num' style='color:#dc2626'>{sell_c}</div>
        <div class='s-lbl'>Sell Signals</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='stat-card stat-hold'>
        <div class='s-icon'>🟡</div>
        <div class='s-num' style='color:#d97706'>{hold_c}</div>
        <div class='s-lbl'>Hold Signals</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class='stat-card stat-acc'>
        <div class='s-icon'>⭐</div>
        <div class='s-num'>{best_acc}%</div>
        <div class='s-lbl'>Best Accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPANY CARDS — each card shows its own real AI signal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">🏢 PSX Companies — Live Snapshot</div>',
            unsafe_allow_html=True)

cols5 = st.columns(5)
for i, tick in enumerate(available_tickers):
    with cols5[i % 5]:
        sd = df_processed[df_processed['Ticker'] == tick]
        if len(sd) == 0:
            continue
        try:
            lc      = float(sd['Close'].iloc[-1])
            ret1d   = float(sd['Return_1d'].iloc[-1])
            sig_info = all_signals.get(tick, {"ret": 0.0, "signal": "hold"})
            sig      = sig_info["signal"]          # buy / sell / hold
            pred_ret = sig_info["ret"]             # predicted % return
            arrow    = "▲" if ret1d >= 0 else "▼"
            clr      = "#15803d" if ret1d >= 0 else "#dc2626"
            pr_clr   = "#15803d" if pred_ret >= 0 else "#dc2626"

            st.markdown(f"""
            <div class='ccard'>
                <div class='c-name'>{TICKER_NAMES.get(tick, tick)}</div>
                <div class='c-price'>PKR {lc:,.0f}</div>
                <div class='c-chg' style='color:{clr}'>
                    {arrow} {abs(ret1d):.2f}%
                </div>
                <div class='c-ai' style='color:{pr_clr}'>
                    AI: {pred_ret:+.2f}%
                </div>
                <div style='margin-top:6px'>
                    <span class='b-{sig}'>{sig.upper()}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRICE CHART + KEY STATS
# ══════════════════════════════════════════════════════════════════════════════
if selected_stock not in data:
    st.warning(f"⚠️ Data not available for "
               f"{TICKER_NAMES.get(selected_stock, selected_stock)}.")
    st.stop()

sd  = df_processed[df_processed['Ticker'] == selected_stock].copy()
raw = data[selected_stock]

left, right = st.columns([2, 1])

with left:
    st.markdown(
        f'<div class="sec-hdr">📊 '
        f'{TICKER_NAMES.get(selected_stock)} — Price Chart</div>',
        unsafe_allow_html=True)

    chart_type = st.radio("", ["Line","Candlestick"], horizontal=True)
    fig = go.Figure()

    if chart_type == "Candlestick" and all(
            c in raw.columns for c in ['Open','High','Low','Close']):
        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw['Open'], high=raw['High'],
            low=raw['Low'],  close=raw['Close'],
            increasing_line_color='#15803d',
            decreasing_line_color='#dc2626'))
    else:
        fig.add_trace(go.Scatter(
            x=sd.index, y=sd['Close'], name="Close Price",
            line=dict(color='#4f46e5', width=2.5),
            fill='tozeroy', fillcolor='rgba(79,70,229,0.07)'))

    fig.add_trace(go.Scatter(x=sd.index, y=sd['SMA_20'], name="SMA 20",
        line=dict(dash='dash', color='#ea580c', width=1.5)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['SMA_50'], name="SMA 50",
        line=dict(dash='dot', color='#0891b2', width=1.5)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['BB_upper'], name="BB Upper",
        line=dict(color='rgba(220,38,38,0.4)', width=1)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['BB_lower'], name="BB Lower",
        line=dict(color='rgba(220,38,38,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(220,38,38,0.04)'))

    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=380,
        xaxis=dict(showgrid=False, color=TICK_CLR),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        legend=dict(orientation="h", y=-0.22, font_color='#334155',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    if 'Volume' in raw.columns and 'Open' in raw.columns:
        vol_colors = [
            'rgba(21,128,61,0.6)' if c >= o else 'rgba(220,38,38,0.5)'
            for c, o in zip(raw['Close'], raw['Open'])
        ]
        fig_v = go.Figure(go.Bar(
            x=raw.index, y=raw['Volume'], marker_color=vol_colors))
        fig_v.update_layout(
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font_color=FONT_CLR, height=120, showlegend=False,
            margin=dict(l=0, r=0, t=5, b=0),
            xaxis=dict(showgrid=False, color=TICK_CLR),
            yaxis=dict(showgrid=False, color=TICK_CLR))
        st.plotly_chart(fig_v, use_container_width=True)

with right:
    # RSI gauge
    st.markdown('<div class="sec-hdr">📉 RSI Gauge</div>',
                unsafe_allow_html=True)
    rsi_now = float(sd['RSI'].iloc[-1]) if len(sd) > 0 else 50.0
    gc = "#15803d" if rsi_now < 40 else ("#dc2626" if rsi_now > 60 else "#d97706")
    rsi_label = "Oversold 🟢" if rsi_now < 40 else ("Overbought 🔴" if rsi_now > 60 else "Neutral 🟡")

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_now,
        number={'font': {'color':'#1e3a5f','size':36}},
        gauge={
            'axis':        {'range':[0,100],'tickcolor':'#64748b',
                            'tickfont':{'color':'#64748b'}},
            'bar':         {'color':gc,'thickness':0.28},
            'bgcolor':     'rgba(255,255,255,0.9)',
            'bordercolor': 'rgba(30,58,95,0.1)',
            'steps': [
                {'range':[0,  40],  'color':'rgba(21,128,61,0.12)'},
                {'range':[40, 60],  'color':'rgba(217,119,6,0.07)'},
                {'range':[60, 100], 'color':'rgba(220,38,38,0.12)'},
            ],
        },
        title={'text': f"RSI — {rsi_label}",
               'font':{'color':'#475569','size':13}}
    ))
    fig_g.update_layout(
        paper_bgcolor=PAPER_BG, font_color=FONT_CLR,
        height=250, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

    # Key stats
    st.markdown('<div class="sec-hdr">📌 Key Stats</div>',
                unsafe_allow_html=True)
    try:
        lc      = float(sd['Close'].iloc[-1])
        raw_1y  = raw.tail(252)
        hi52    = float(raw_1y['High'].max()) if 'High' in raw_1y else 0
        lo52    = float(raw_1y['Low'].min())  if 'Low'  in raw_1y else 0
        macd_v  = float(sd['MACD'].iloc[-1])
        vol20   = float(sd['Volatility'].iloc[-1])
        sig_now = all_signals.get(selected_stock, {"ret":0,"signal":"hold"})
        pred_r  = sig_now["ret"]
        ai_sig  = sig_now["signal"].upper()

        stats = [
            ("Last Close",    f"PKR {lc:,.0f}"),
            ("AI Prediction", f"{pred_r:+.2f}%"),
            ("AI Signal",     ai_sig),
            ("52W High",      f"PKR {hi52:,.0f}"),
            ("52W Low",       f"PKR {lo52:,.0f}"),
            ("MACD",          f"{macd_v:.2f}"),
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
        st.info("Stats unavailable.")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">🤖 AI Model Performance Comparison</div>',
            unsafe_allow_html=True)

rf_res  = results.get("Random Forest", {"Acc":0,"MAE":0,"per_stock":{}})
xgb_res = results.get("XGBoost",       {"Acc":0,"MAE":0,"per_stock":{}})
winner  = "XGBoost" if xgb_res.get('Acc',0) >= rf_res.get('Acc',0) \
          else "Random Forest"
loser   = "Random Forest" if winner == "XGBoost" else "XGBoost"
w_acc   = results.get(winner,{}).get('Acc',0)
l_acc   = results.get(loser, {}).get('Acc',0)

# Winner banner
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#7c3aed);
     border-radius:16px;padding:18px 24px;
     display:flex;align-items:center;justify-content:space-between;
     margin-bottom:16px;box-shadow:0 6px 24px rgba(124,58,237,0.3);'>
  <div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.78rem;font-weight:600;
         text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;'>
         🏆 Best Performing Model</div>
    <div style='color:#fff;font-size:1.6rem;font-weight:900;'>{winner}</div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.85rem;margin-top:2px;'>
      Accuracy: <b style='color:#a5f3fc'>{w_acc}%</b>
      &nbsp;|&nbsp;
      MAE: <b style='color:#a5f3fc'>PKR {results.get(winner,{}).get('MAE',0):.0f}</b>
    </div>
  </div>
  <div style='text-align:right;'>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;margin-bottom:4px;'>
        vs {loser}</div>
    <div style='color:#fde68a;font-size:1.2rem;font-weight:800;'>
        +{abs(w_acc - l_acc):.1f}% better</div>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;'>
        accuracy advantage</div>
  </div>
</div>""", unsafe_allow_html=True)

# 4 metric cards
mc = st.columns(4)
for i, (mname, lbl, val, hint) in enumerate([
    ("Random Forest","Accuracy",        f"{rf_res.get('Acc','—')}%",        "Higher is better"),
    ("Random Forest","MAE (Avg Error)", f"PKR {rf_res.get('MAE',0):.0f}",   "Lower is better"),
    ("XGBoost",      "Accuracy",        f"{xgb_res.get('Acc','—')}%",       "Higher is better"),
    ("XGBoost",      "MAE (Avg Error)", f"PKR {xgb_res.get('MAE',0):.0f}",  "Lower is better"),
]):
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

rf_ps  = rf_res.get('per_stock',  {})
xgb_ps = xgb_res.get('per_stock', {})
all_t  = sorted(set(list(rf_ps.keys()) + list(xgb_ps.keys())))
names_ = [TICKER_NAMES.get(t, t) for t in all_t]

fig_per = go.Figure()
fig_per.add_trace(go.Bar(
    name='Random Forest', x=names_,
    y=[rf_ps.get(t,0) for t in all_t],
    marker_color='#7c3aed', opacity=0.85,
    text=[f"{rf_ps.get(t,0):.1f}%" for t in all_t],
    textposition='outside', textfont=dict(size=10,color='#1e3a5f')))
fig_per.add_trace(go.Bar(
    name='XGBoost', x=names_,
    y=[xgb_ps.get(t,0) for t in all_t],
    marker_color='#0891b2', opacity=0.85,
    text=[f"{xgb_ps.get(t,0):.1f}%" for t in all_t],
    textposition='outside', textfont=dict(size=10,color='#1e3a5f')))
fig_per.add_hline(y=90, line_dash="dash", line_color="#dc2626",
    line_width=1.5, annotation_text="Target 90%",
    annotation_font_color="#dc2626", annotation_font_size=11)
fig_per.update_layout(
    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
    font_color=FONT_CLR, height=360, barmode='group',
    yaxis=dict(range=[0,115], showgrid=True,
               gridcolor=GRID_CLR, color=TICK_CLR, title="Accuracy (%)"),
    xaxis=dict(showgrid=False, color=TICK_CLR,
               tickangle=-25, tickfont=dict(size=10)),
    legend=dict(font_color='#334155',bgcolor='rgba(255,255,255,0.8)',
                orientation='h',y=1.1),
    margin=dict(l=10,r=10,t=50,b=80))
st.plotly_chart(fig_per, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="sec-hdr">💡 Top 5 Stock Recommendations — {selected_model}</div>',
    unsafe_allow_html=True)
st.markdown(
    f"<span style='color:#64748b;font-size:0.84rem;font-weight:600;'>"
    f"Excluding: {TICKER_NAMES.get(selected_stock, selected_stock)}</span>",
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

recs = recommend_stocks([selected_stock], all_signals, df_processed)

if recs:
    rcols = st.columns(min(5, len(recs)))
    for i, r in enumerate(recs):
        with rcols[i % len(rcols)]:
            st.markdown(f"""
            <div class='rec-card'>
                <div class='r-name'>{r['name']}</div>
                <div class='r-sector'>{r['sector']}</div>
                <div class='r-return'>{r['predicted_return']:+.2f}%</div>
                <div class='r-price'>PKR {r['last_close']:,.0f}</div>
                <div style='margin-top:10px'>
                    <span class='b-{r["signal"].lower()}'>{r['signal']}</span>
                </div>
                <div style='color:#94a3b8;font-size:0.72rem;
                     margin-top:6px;font-weight:600;'>
                    RSI {r['rsi']}
                </div>
            </div>""", unsafe_allow_html=True)
else:
    st.info("No recommendations available. Try selecting a different company.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;
     padding:10px 0 20px 0;font-weight:500;'>
📌 PSX AI Stock Predictor &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp;
Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; Data: Yahoo Finance<br>
⚠️ For educational purposes only. Not financial advice.
</div>""", unsafe_allow_html=True)
