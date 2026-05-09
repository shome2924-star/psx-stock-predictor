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
from datetime import datetime

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
section[data-testid="stSidebar"] * { color: #ffffff !important; }
.main-title {
    font-size:2.8rem; font-weight:900;
    background:linear-gradient(90deg,#1e3a5f,#7c3aed,#be185d);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; margin-bottom:4px;
}
.sub-title { text-align:center; color:#475569; font-size:0.95rem;
             margin-bottom:20px; font-weight:500; }
.sec-hdr { font-size:1.1rem; font-weight:800; color:#1e3a5f;
           border-left:4px solid #7c3aed; padding-left:10px;
           margin:22px 0 14px 0; }
.stat-card { background:#ffffff; border-radius:16px; padding:18px 12px;
             text-align:center; box-shadow:0 4px 20px rgba(124,58,237,0.12);
             border:1px solid rgba(124,58,237,0.15); margin-bottom:10px; }
.stat-card .s-num { font-size:2rem; font-weight:900; color:#1e3a5f; }
.stat-card .s-lbl { font-size:0.75rem; color:#64748b; margin-top:3px;
                    font-weight:600; }
.ccard { background:#ffffff; border-radius:16px; padding:16px 10px;
         text-align:center; box-shadow:0 4px 16px rgba(30,58,95,0.10);
         border:1px solid rgba(30,58,95,0.10); margin-bottom:10px; }
.ccard .c-name  { color:#64748b; font-size:0.72rem; font-weight:700;
                  margin-bottom:6px; text-transform:uppercase; }
.ccard .c-price { color:#1e3a5f; font-size:1.25rem; font-weight:900; }
.ccard .c-chg   { font-size:0.8rem; font-weight:700; margin-top:4px; }
.b-buy  { background:#dcfce7; color:#166534; border:1.5px solid #16a34a;
          padding:3px 12px; border-radius:20px; font-size:0.72rem;
          font-weight:800; }
.b-sell { background:#fee2e2; color:#991b1b; border:1.5px solid #dc2626;
          padding:3px 12px; border-radius:20px; font-size:0.72rem;
          font-weight:800; }
.b-hold { background:#fef3c7; color:#92400e; border:1.5px solid #d97706;
          padding:3px 12px; border-radius:20px; font-size:0.72rem;
          font-weight:800; }
.rec-card { background:#ffffff; border-radius:18px; padding:22px 14px;
            text-align:center; box-shadow:0 6px 24px rgba(30,58,95,0.12);
            border:2px solid rgba(124,58,237,0.12); margin-bottom:8px; }
.rec-card .r-name   { color:#475569; font-size:0.8rem; font-weight:700; }
.rec-card .r-return { color:#7c3aed; font-size:1.8rem; font-weight:900; }
.rec-card .r-price  { color:#64748b; font-size:0.8rem; margin-top:4px;
                      font-weight:600; }
.model-card { background:linear-gradient(135deg,#1e3a5f 0%,#7c3aed 100%);
              border-radius:16px; padding:20px; text-align:center;
              box-shadow:0 6px 24px rgba(124,58,237,0.30); margin-bottom:10px; }
.model-card .mc-name { color:rgba(255,255,255,0.8); font-size:0.82rem;
                       font-weight:600; margin-bottom:6px; }
.model-card .mc-val  { color:#ffffff; font-size:2rem; font-weight:900; }
.model-card .mc-lbl  { color:rgba(255,255,255,0.65); font-size:0.72rem;
                       margin-top:3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

TICKERS = ["OGDC.KA","PSO.KA","HBL.KA","MCB.KA","LUCK.KA",
           "ENGRO.KA","FFC.KA","UBL.KA","NESTLE.KA","SYS.KA"]

TICKER_NAMES = {
    "OGDC.KA":"Oil & Gas Dev Co", "PSO.KA":"Pakistan State Oil",
    "HBL.KA":"Habib Bank",        "MCB.KA":"MCB Bank",
    "LUCK.KA":"Lucky Cement",     "ENGRO.KA":"Engro Corporation",
    "FFC.KA":"Fauji Fertilizer",  "UBL.KA":"United Bank",
    "NESTLE.KA":"Nestle Pakistan","SYS.KA":"Systems Limited",
}

TICKER_SECTORS = {
    "OGDC.KA":"Energy",    "PSO.KA":"Energy",    "HBL.KA":"Banking",
    "MCB.KA":"Banking",    "LUCK.KA":"Cement",   "ENGRO.KA":"Chemicals",
    "FFC.KA":"Fertilizer", "UBL.KA":"Banking",   "NESTLE.KA":"FMCG",
    "SYS.KA":"Technology",
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
            df = yf.download(t, start="2019-01-01", end="2024-12-31",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open','High','Low','Close','Volume']].copy()
            if len(df) > 60:
                data[t] = df
        except:
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
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                          / (np.array(y_true) + 1e-9))) * 100
    return round(max(0.0, 100.0 - mape), 2)

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training AI models...")
def train_models(data):
    processed_dfs = []
    for ticker, raw_df in data.items():
        feat_df = add_features(raw_df)
        feat_df['Ticker'] = ticker
        processed_dfs.append(feat_df)

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
        for ticker in df_proc['Ticker'].unique():
            tdf   = df_proc[df_proc['Ticker'] == ticker].copy()
            split = int(len(tdf) * 0.8)
            X_tr  = tdf[FEATURES].iloc[:split]
            X_te  = tdf[FEATURES].iloc[split:]
            y_tr  = tdf['Target'].iloc[:split]
            y_te  = tdf['Target'].iloc[split:]
            if len(X_tr) < 10 or len(X_te) < 5:
                continue
            m = make_model()
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            acc   = mape_accuracy(y_te.values, preds)
            mae   = mean_absolute_error(y_te, preds)
            per_models[mname][ticker] = m
            accs.append(acc)
            maes.append(mae)
            results.setdefault(mname, {"per_stock": {}})
            results[mname]["per_stock"][ticker] = acc

        results[mname]["Acc"] = round(float(np.mean(accs)), 1)
        results[mname]["MAE"] = round(float(np.mean(maes)), 2)

    return df_proc, per_models, results

# ── Recommendation engine ─────────────────────────────────────────────────────
def recommend_stocks(watchlist, model_name, data,
                     df_processed, per_models, top_n=5):
    recs = []
    for ticker in data.keys():
        if ticker in watchlist:
            continue
        sd = df_processed[df_processed['Ticker'] == ticker]
        if len(sd) == 0:
            continue
        m = per_models.get(model_name, {}).get(ticker)
        if m is None:
            continue
        try:
            pp  = m.predict(sd[FEATURES].iloc[-1:].values)[0]
            lc  = sd['Close'].iloc[-1]
            ret = ((pp - lc) / lc) * 100
        except:
            continue
        rsi       = sd['RSI'].iloc[-1]
        rsi_score = 1.0 if rsi < 40 else (-1.0 if rsi > 60 else 0.0)
        vol       = sd['Close'].pct_change().std()
        vol       = 0.0 if pd.isna(vol) else vol
        signal    = "Buy" if ret > 0.5 else ("Sell" if ret < -0.5 else "Hold")
        score     = ret * 0.6 + rsi_score * 0.3 - vol * 0.1
        recs.append({
            "ticker": ticker, "name": TICKER_NAMES.get(ticker, ticker),
            "sector": TICKER_SECTORS.get(ticker, ""),
            "predicted_return": round(ret, 2), "signal": signal,
            "rsi": round(rsi, 1), "last_close": round(lc, 2),
            "score": round(score, 4),
        })
    recs.sort(key=lambda x: x['score'], reverse=True)
    return recs[:top_n]

# ── Signal helper ─────────────────────────────────────────────────────────────
def get_signal(ticker, model_name, df_processed, per_models):
    sd = df_processed[df_processed['Ticker'] == ticker]
    if len(sd) == 0:
        return "hold"
    try:
        m = per_models.get(model_name, {}).get(ticker)
        if m is None:
            return "hold"
        pp  = m.predict(sd[FEATURES].iloc[-1:].values)[0]
        lc  = sd['Close'].iloc[-1]
        ret = ((pp - lc) / lc) * 100
        return "buy" if ret > 0.5 else ("sell" if ret < -0.5 else "hold")
    except:
        return "hold"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# Load & train
data = load_data()
df_processed, per_models, results = train_models(data)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    selected_stock = st.selectbox(
        "Select Company",
        options=list(TICKER_NAMES.keys()),
        format_func=lambda x: TICKER_NAMES[x]
    )
    selected_model = st.radio(
        "AI Model",
        options=["Random Forest", "XGBoost"],
        index=1
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.8rem;color:#cbd5e1;line-height:1.8'>
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
    ' for Pakistan Stock Exchange</div>',
    unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
buy_c  = sum(1 for t in data if get_signal(t, selected_model,
                                            df_processed, per_models) == "buy")
sell_c = sum(1 for t in data if get_signal(t, selected_model,
                                            df_processed, per_models) == "sell")
hold_c = len(data) - buy_c - sell_c
best   = max(results, key=lambda x: results[x]['Acc'])

cols = st.columns(5)
for col, num, lbl in zip(cols,
    [len(data), buy_c, sell_c, hold_c, f"{results[best]['Acc']}%"],
    ["🏢 Companies","🟢 Buy","🔴 Sell","🟡 Hold","⭐ Best Accuracy"]):
    with col:
        st.markdown(
            f"<div class='stat-card'><div class='s-num'>{num}</div>"
            f"<div class='s-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Company cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🏢 PSX Companies — Live Snapshot</div>',
            unsafe_allow_html=True)
cols5 = st.columns(5)
for i, tick in enumerate(TICKER_NAMES.keys()):
    with cols5[i % 5]:
        if tick in df_processed['Ticker'].values:
            sd    = df_processed[df_processed['Ticker'] == tick]
            lc    = sd['Close'].iloc[-1]
            rsi   = sd['RSI'].iloc[-1]
            ret1d = sd['Return_1d'].iloc[-1]
            sig   = "buy" if rsi < 40 else ("sell" if rsi > 60 else "hold")
            arrow = "▲" if ret1d > 0 else "▼"
            clr   = "#15803d" if ret1d > 0 else "#dc2626"
            st.markdown(f"""
            <div class='ccard'>
                <div class='c-name'>{TICKER_NAMES[tick]}</div>
                <div class='c-price'>PKR {lc:,.0f}</div>
                <div class='c-chg' style='color:{clr}'>{arrow} {abs(ret1d):.2f}%</div>
                <div style='margin-top:5px'>
                    <span class='b-{sig}'>{sig.upper()}</span>
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Price chart ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-hdr">📊 {TICKER_NAMES.get(selected_stock)} — Price Chart</div>',
    unsafe_allow_html=True)

sd  = df_processed[df_processed['Ticker'] == selected_stock].copy()
raw = data[selected_stock]

chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True)
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=raw.index, open=raw['Open'], high=raw['High'],
        low=raw['Low'], close=raw['Close'],
        increasing_line_color='#15803d',
        decreasing_line_color='#dc2626'))
else:
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['Close'], name="Close",
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
    font_color=FONT_CLR, height=400,
    xaxis=dict(showgrid=False, color=TICK_CLR),
    yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
    legend=dict(orientation="h", y=-0.2, bgcolor='rgba(255,255,255,0.8)'),
    margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model comparison ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🤖 Model Performance</div>',
            unsafe_allow_html=True)

rf_res  = results.get("Random Forest", {})
xgb_res = results.get("XGBoost", {})
winner  = "XGBoost" if xgb_res.get('Acc',0) >= rf_res.get('Acc',0) \
          else "Random Forest"

mc = st.columns(4)
metrics_list = [
    ("Random Forest", "Accuracy",  f"{rf_res.get('Acc','—')}%",        "Higher is better"),
    ("Random Forest", "Avg Error", f"PKR {rf_res.get('MAE',0):.0f}",   "Lower is better"),
    ("XGBoost",       "Accuracy",  f"{xgb_res.get('Acc','—')}%",       "Higher is better"),
    ("XGBoost",       "Avg Error", f"PKR {xgb_res.get('MAE',0):.0f}",  "Lower is better"),
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

# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-hdr">💡 Top Stock Recommendations — {selected_model}</div>',
    unsafe_allow_html=True)

recs = recommend_stocks(
    [selected_stock], selected_model,
    data, df_processed, per_models
)

if recs:
    rcols = st.columns(min(5, len(recs)))
    for i, r in enumerate(recs):
        with rcols[i]:
            st.markdown(f"""
            <div class='rec-card'>
                <div class='r-name'>{r['name']}</div>
                <div style='color:#94a3b8;font-size:0.68rem'>{r['sector']}</div>
                <div class='r-return'>{r['predicted_return']}%</div>
                <div class='r-price'>PKR {r['last_close']:,.0f}</div>
                <div style='margin-top:10px'>
                    <span class='b-{r["signal"].lower()}'>{r['signal']}</span>
                </div>
                <div style='color:#94a3b8;font-size:0.72rem;margin-top:6px'>
                    RSI {r['rsi']}
                </div>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:10px 0 20px'>
📌 PSX AI Stock Predictor &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp;
Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; Data: Yahoo Finance<br>
⚠️ For educational purposes only. Not financial advice.
</div>""", unsafe_allow_html=True)
