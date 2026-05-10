import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

st.set_page_config(page_title="PSX AI Stock Predictor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Light sky-blue background ── */
.stApp {
    background: linear-gradient(140deg, #dbeafe 0%, #ede9fe 40%, #fce7f3 70%, #d1fae5 100%);
    background-attachment: fixed;
}

/* ── Sidebar — deep navy ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #2d1b69 100%) !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption { 
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
/* Live line fix */
.live-line { 
    color: #ffffff !important; 
    font-size: 0.85rem; 
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 6px 0 10px 0;
}
.sidebar-note {
    color: #cbd5e1 !important;
    font-size: 0.82rem;
    line-height: 1.7;
}

/* ── Main title ── */
.main-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(90deg, #1e3a5f, #7c3aed, #be185d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.sub-title {
    text-align: center; color: #475569;
    font-size: 0.95rem; margin-bottom: 20px; font-weight: 500;
}

/* ── Section header ── */
.sec-hdr {
    font-size: 1.1rem; font-weight: 800; color: #1e3a5f;
    border-left: 4px solid #7c3aed;
    padding-left: 10px; margin: 22px 0 14px 0;
}

/* ── Stat card ── */
.stat-card {
    background: #ffffff;
    border-radius: 16px; padding: 18px 12px; text-align: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.15);
    margin-bottom: 10px;
}
.stat-card .s-num { font-size: 2rem; font-weight: 900; color: #1e3a5f; }
.stat-card .s-lbl { font-size: 0.75rem; color: #64748b; margin-top: 3px; font-weight: 600; }

/* ── Company card ── */
.ccard {
    background: #ffffff;
    border-radius: 16px; padding: 16px 10px; text-align: center;
    box-shadow: 0 4px 16px rgba(30,58,95,0.10);
    border: 1px solid rgba(30,58,95,0.10);
    margin-bottom: 10px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.ccard:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(124,58,237,0.18); }
.ccard .c-name  { color: #64748b; font-size: 0.72rem; font-weight: 700;
                  margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.ccard .c-price { color: #1e3a5f; font-size: 1.25rem; font-weight: 900; }
.ccard .c-chg   { font-size: 0.8rem; font-weight: 700; margin-top: 4px; }
.ccard .c-rsi   { font-size: 0.7rem; color: #94a3b8; margin-top: 5px; }

/* ── Badges ── */
.b-buy  { background:#dcfce7; color:#166534; border:1.5px solid #16a34a;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-sell { background:#fee2e2; color:#991b1b; border:1.5px solid #dc2626;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }
.b-hold { background:#fef3c7; color:#92400e; border:1.5px solid #d97706;
          padding:3px 12px; border-radius:20px; font-size:0.72rem; font-weight:800; }

/* ── Mini stat card ── */
.mini-card {
    background: #ffffff;
    border-radius: 12px; padding: 12px 10px; text-align: center;
    box-shadow: 0 2px 10px rgba(30,58,95,0.08);
    border: 1px solid rgba(30,58,95,0.08);
    margin-bottom: 8px;
}
.mini-card .m-lbl { color: #94a3b8; font-size: 0.68rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.mini-card .m-val { color: #1e293b; font-size: 1rem; font-weight: 800; }

/* ── Model card ── */
.model-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #7c3aed 100%);
    border-radius: 16px; padding: 20px; text-align: center;
    box-shadow: 0 6px 24px rgba(124,58,237,0.30);
    margin-bottom: 10px;
}
.model-card .mc-name { color: rgba(255,255,255,0.8); font-size: 0.82rem;
                        font-weight: 600; margin-bottom: 6px; }
.model-card .mc-val  { color: #ffffff; font-size: 2rem; font-weight: 900; }
.model-card .mc-lbl  { color: rgba(255,255,255,0.65); font-size: 0.72rem; margin-top: 3px; }

/* ── Recommendation card ── */
.rec-card {
    background: #ffffff;
    border-radius: 18px; padding: 22px 14px; text-align: center;
    box-shadow: 0 6px 24px rgba(30,58,95,0.12);
    border: 2px solid rgba(124,58,237,0.12);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 8px;
}
.rec-card:hover { transform: translateY(-5px); box-shadow: 0 14px 40px rgba(124,58,237,0.22); }
.rec-card .r-name   { color: #475569; font-size: 0.8rem; font-weight: 700; }
.rec-card .r-sector { color: #94a3b8; font-size: 0.68rem; margin-bottom: 8px; }
.rec-card .r-return { color: #7c3aed; font-size: 1.8rem; font-weight: 900; }
.rec-card .r-price  { color: #64748b; font-size: 0.8rem; margin-top: 4px; font-weight: 600; }

/* ── Live dot ── */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #22c55e; border-radius: 50%;
    animation: blink 1.2s infinite; margin-right: 5px; vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Streamlit overrides ── */
h1,h2,h3  { color: #1e3a5f !important; }
p, label, .stMarkdown p { color: #334155 !important; }
.stRadio label  { color: #334155 !important; }
.stCaption      { color: #64748b !important; }
.stDivider      { border-color: rgba(30,58,95,0.12) !important; }
div[data-testid="stMetricValue"] { color: #1e3a5f !important; }
div[data-testid="stMetricLabel"] { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
tickers = ["OGDC.KA","PSO.KA","HBL.KA","MCB.KA","LUCK.KA",
           "ENGRO.KA","FFC.KA","UBL.KA","NESTLE.KA","SYS.KA"]
ticker_names = {
    "OGDC.KA":"Oil & Gas Dev Co", "PSO.KA":"Pakistan State Oil",
    "HBL.KA":"Habib Bank",        "MCB.KA":"MCB Bank",
    "LUCK.KA":"Lucky Cement",     "ENGRO.KA":"Engro Corporation",
    "FFC.KA":"Fauji Fertilizer",  "UBL.KA":"United Bank",
    "NESTLE.KA":"Nestle Pakistan","SYS.KA":"Systems Limited",
}
ticker_sectors = {
    "OGDC.KA":"Energy",    "PSO.KA":"Energy",    "HBL.KA":"Banking",
    "MCB.KA":"Banking",    "LUCK.KA":"Cement",   "ENGRO.KA":"Chemicals",
    "FFC.KA":"Fertilizer", "UBL.KA":"Banking",   "NESTLE.KA":"FMCG",
    "SYS.KA":"Technology",
}

@st.cache_data(show_spinner="📡 Loading PSX market data...")
def load_data():
    data = {}
    for t in tickers:
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

data = load_data()

def add_features(df):
    df = df.copy()
    df['SMA_20']    = df['Close'].rolling(20).mean()
    df['SMA_50']    = df['Close'].rolling(50).mean()
    delta           = df['Close'].diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = -delta.clip(upper=0).rolling(14).mean()
    df['RSI']       = 100 - (100 / (1 + gain / loss))
    ema12           = df['Close'].ewm(span=12).mean()
    ema26           = df['Close'].ewm(span=26).mean()
    df['MACD']      = ema12 - ema26
    df['BB_upper']  = df['SMA_20'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower']  = df['SMA_20'] - 2 * df['Close'].rolling(20).std()
    df['Return_1d'] = df['Close'].pct_change() * 100
    df['Volatility']= df['Return_1d'].rolling(20).std()
    return df.dropna()

processed_dfs = []
for ticker, raw_df in data.items():
    feat_df = add_features(raw_df)
    feat_df['Ticker'] = ticker
    processed_dfs.append(feat_df)
df_processed = pd.concat(processed_dfs)

features = ['SMA_20','SMA_50','RSI','MACD','BB_upper','BB_lower']
df_processed['Target'] = df_processed.groupby('Ticker')['Close'].shift(-1)
df_model = df_processed.dropna()
X = df_model[features]; y = df_model['Target']
split = int(len(X)*0.8)
X_train,X_test = X[:split],X[split:]
y_train,y_test = y[:split],y[split:]

@st.cache_resource(show_spinner="🤖 Training AI models...")
def train_models():
    mdls = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":       GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    res = {}
    for name, m in mdls.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        # Per-stock accuracy — correct method
        test_df = df_model.iloc[split:].copy()
        test_df['Pred'] = p
        per_acc, per_mae, per_rmse = [], [], []
        for ticker in test_df['Ticker'].unique():
            t = test_df[test_df['Ticker']==ticker]
            if len(t) < 5: continue
            t_mae  = mean_absolute_error(t['Target'], t['Pred'])
            t_rmse = np.sqrt(mean_squared_error(t['Target'], t['Pred']))
            t_acc  = max(0, 100 - (t_mae / t['Target'].mean() * 100))
            per_acc.append(t_acc); per_mae.append(t_mae); per_rmse.append(t_rmse)
        res[name] = {
            "MAE":       round(float(np.mean(per_mae)),  2),
            "RMSE":      round(float(np.mean(per_rmse)), 2),
            "Acc":       round(float(np.mean(per_acc)),  1),
            "per_stock": {ticker: round(max(0,100-(mean_absolute_error(
                            test_df[test_df['Ticker']==ticker]['Target'],
                            test_df[test_df['Ticker']==ticker]['Pred']
                          ) / test_df[test_df['Ticker']==ticker]['Target'].mean()*100)),1)
                          for ticker in test_df['Ticker'].unique() if len(test_df[test_df['Ticker']==ticker])>=5}
        }
    return mdls, res

models, results = train_models()

def recommend_stocks(watchlist, model, top_n=5):
    recs = []
    for ticker in data.keys():
        if ticker in watchlist: continue
        sd = df_processed[df_processed['Ticker']==ticker]
        if len(sd)==0: continue
        try:
            pp  = model.predict(sd[features].iloc[-1:].values)[0]
            lc  = sd['Close'].iloc[-1]
            ret = ((pp - lc) / lc) * 100
        except: continue
        rsi = sd['RSI'].iloc[-1]
        rs  = 1.0 if rsi<40 else (-1.0 if rsi>60 else 0.0)
        vol = sd['Close'].pct_change().std() or 0.0
        sig = "Buy" if ret>0.5 else ("Sell" if ret<-0.5 else "Hold")
        recs.append({
            'ticker': ticker, 'name': ticker_names.get(ticker,ticker),
            'sector': ticker_sectors.get(ticker,''),
            'predicted_return': round(ret,2), 'signal': sig,
            'rsi': round(rsi,1), 'score': ret*0.6 + rs*0.3 - vol*0.1,
            'last_close': round(lc, 0)
        })
    return sorted(recs, key=lambda x: x['score'], reverse=True)[:top_n]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#ffffff !important; font-size:1.3rem; font-weight:900; margin-bottom:6px;'>📈 PSX AI Predictor</h2>", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:rgba(255,255,255,0.12);border-radius:10px;padding:8px 12px;margin-bottom:14px;'>
        <span style='display:inline-block;width:9px;height:9px;background:#22c55e;border-radius:50%;margin-right:7px;vertical-align:middle;'></span>
        <span style='color:#ffffff;font-size:0.85rem;font-weight:700;vertical-align:middle;'>LIVE</span>
        <span style='color:#cbd5e1;font-size:0.82rem;margin-left:8px;'>{datetime.now().strftime('%d %b %Y')}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);margin:4px 0 14px 0;'>", unsafe_allow_html=True)
    selected_stock = st.selectbox("🏢 Company", list(data.keys()),
                                   format_func=lambda x: ticker_names.get(x,x))
    selected_model = st.selectbox("🤖 Model", list(models.keys()))
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style='background:rgba(255,255,255,0.08);border-radius:10px;padding:10px 12px;'>
        <div style='color:#ffffff;font-size:0.82rem;font-weight:700;margin-bottom:4px;'>📌 Final Year Project</div>
        <div style='color:#cbd5e1;font-size:0.78rem;line-height:1.8;'>
            PSX Stock Prediction<br>&amp; Recommendation System<br>Data: Yahoo Finance
        </div>
    </div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📈 PSX AI Stock Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Prediction & Recommendation System for Top 10 Pakistan Stock Exchange Companies</div>', unsafe_allow_html=True)

# ── STATS BAR ─────────────────────────────────────────────────────────────────
def get_signal(ticker):
    sd = df_processed[df_processed['Ticker']==ticker]
    if len(sd)==0: return "hold"
    try:
        pp  = models[selected_model].predict(sd[features].iloc[-1:].values)[0]
        lc  = sd['Close'].iloc[-1]
        ret = ((pp - lc) / lc) * 100
        return "buy" if ret>0.5 else ("sell" if ret<-0.5 else "hold")
    except:
        return "hold"

buy_c  = sum(1 for t in data if get_signal(t)=="buy")
sell_c = sum(1 for t in data if get_signal(t)=="sell")
hold_c = len(data) - buy_c - sell_c
best   = max(results, key=lambda x: results[x]['Acc'])

s1,s2,s3,s4,s5 = st.columns(5)
for col, num, lbl in zip(
    [s1,s2,s3,s4,s5],
    [len(data), buy_c, sell_c, hold_c, f"{results[best]['Acc']}%"],
    ["🏢 Companies", "🟢 Buy Signals", "🔴 Sell Signals", "🟡 Hold Signals", "⭐ Best Accuracy"]
):
    with col:
        st.markdown(f"<div class='stat-card'><div class='s-num'>{num}</div><div class='s-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── COMPANY CARDS ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🏢 All 10 PSX Companies — Live Snapshot</div>', unsafe_allow_html=True)
cols5 = st.columns(5)
for i, tick in enumerate(ticker_names.keys()):
    with cols5[i % 5]:
        if tick in df_processed['Ticker'].values:
            sd    = df_processed[df_processed['Ticker']==tick]
            lc    = sd['Close'].iloc[-1]
            rsi   = sd['RSI'].iloc[-1]
            ret1d = sd['Return_1d'].iloc[-1]
            sig   = "buy" if rsi<40 else ("sell" if rsi>60 else "hold")
            arrow = "▲" if ret1d>0 else "▼"
            clr   = "#15803d" if ret1d>0 else "#dc2626"
            st.markdown(f"""
            <div class='ccard'>
                <div class='c-name'>{ticker_names[tick]}</div>
                <div class='c-price'>PKR {lc:,.0f}</div>
                <div class='c-chg' style='color:{clr}'>{arrow} {abs(ret1d):.2f}%</div>
                <div class='c-rsi'>RSI {rsi:.0f} &nbsp;
                    <span class='b-{sig}'>{sig.upper()}</span>
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CHART SECTION ─────────────────────────────────────────────────────────────
left, right = st.columns([2,1])
sd  = df_processed[df_processed['Ticker']==selected_stock].copy()
raw = data[selected_stock]

PLOT_BG   = 'rgba(255,255,255,0.85)'
PAPER_BG  = 'rgba(255,255,255,0)'
GRID_CLR  = 'rgba(30,58,95,0.07)'
TICK_CLR  = '#64748b'
FONT_CLR  = '#1e3a5f'

with left:
    st.markdown(f'<div class="sec-hdr">📊 {ticker_names.get(selected_stock)} — Price Chart</div>', unsafe_allow_html=True)
    chart_type = st.radio("", ["Line","Candlestick"], horizontal=True)

    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw['Open'], high=raw['High'],
            low=raw['Low'], close=raw['Close'], name="OHLC",
            increasing_line_color='#15803d', decreasing_line_color='#dc2626'))
    else:
        fig.add_trace(go.Scatter(x=sd.index, y=sd['Close'],
                                 name="Close Price",
                                 line=dict(color='#4f46e5', width=2.5),
                                 fill='tozeroy', fillcolor='rgba(79,70,229,0.07)'))

    fig.add_trace(go.Scatter(x=sd.index, y=sd['SMA_20'], name="SMA 20",
                             line=dict(dash='dash', color='#ea580c', width=1.5)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['SMA_50'], name="SMA 50",
                             line=dict(dash='dot',  color='#0891b2', width=1.5)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['BB_upper'], name="BB Upper",
                             line=dict(color='rgba(220,38,38,0.4)', width=1)))
    fig.add_trace(go.Scatter(x=sd.index, y=sd['BB_lower'], name="BB Lower",
                             line=dict(color='rgba(220,38,38,0.4)', width=1),
                             fill='tonexty', fillcolor='rgba(220,38,38,0.04)'))
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=380,
        xaxis=dict(showgrid=False, color=TICK_CLR, linecolor=GRID_CLR),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        legend=dict(orientation="h", y=-0.22, font_color='#334155',
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.05)'),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume chart
    vol_colors = ['rgba(21,128,61,0.6)' if c >= o else 'rgba(220,38,38,0.5)'
                  for c,o in zip(raw['Close'], raw['Open'])]
    fig_v = go.Figure(go.Bar(x=raw.index, y=raw['Volume'],
                              marker_color=vol_colors, name="Volume"))
    fig_v.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=120,
        margin=dict(l=0,r=0,t=5,b=0), showlegend=False,
        xaxis=dict(showgrid=False, color=TICK_CLR),
        yaxis=dict(showgrid=False, color=TICK_CLR)
    )
    st.plotly_chart(fig_v, use_container_width=True)

with right:
    st.markdown('<div class="sec-hdr">📉 RSI Gauge</div>', unsafe_allow_html=True)
    rsi_now = sd['RSI'].iloc[-1]
    gc = "#15803d" if rsi_now<40 else ("#dc2626" if rsi_now>60 else "#d97706")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_now,
        number={'font': {'color': '#1e3a5f', 'size': 36}},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'#64748b', 'tickfont':{'color':'#64748b'}},
            'bar':  {'color': gc, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': 'rgba(30,58,95,0.1)',
            'steps': [
                {'range':[0,30],  'color':'rgba(21,128,61,0.12)'},
                {'range':[30,70], 'color':'rgba(217,119,6,0.07)'},
                {'range':[70,100],'color':'rgba(220,38,38,0.12)'},
            ],
        },
        title={'text':"RSI Index", 'font':{'color':'#475569','size':14}}
    ))
    fig_g.update_layout(
        paper_bgcolor=PAPER_BG, font_color=FONT_CLR,
        height=240, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown('<div class="sec-hdr">📌 Key Stats</div>', unsafe_allow_html=True)
    lc    = sd['Close'].iloc[-1]
    raw_1y = raw.tail(252)
    hi52  = raw_1y['High'].max()
    lo52  = raw_1y['Low'].min()
    macd  = sd['MACD'].iloc[-1]
    vol20 = sd['Volatility'].iloc[-1]
    stats = [("Last Close",f"PKR {lc:,.0f}"),("52W High",f"PKR {hi52:,.0f}"),
             ("52W Low",f"PKR {lo52:,.0f}"),("MACD",f"{macd:.2f}"),
             ("Volatility",f"{vol20:.2f}%"),("Sector",ticker_sectors.get(selected_stock,''))]
    gc1,gc2 = st.columns(2)
    for j,(lb,vl) in enumerate(stats):
        with (gc1 if j%2==0 else gc2):
            st.markdown(f"<div class='mini-card'><div class='m-lbl'>{lb}</div><div class='m-val'>{vl}</div></div>",
                        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🤖 AI Model Performance Comparison</div>', unsafe_allow_html=True)

rf_res  = results["Random Forest"]
xgb_res = results["XGBoost"]
winner  = "XGBoost" if xgb_res['Acc'] >= rf_res['Acc'] else "Random Forest"
loser   = "Random Forest" if winner == "XGBoost" else "XGBoost"

# ── Winner banner ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#7c3aed);border-radius:16px;
     padding:18px 24px;display:flex;align-items:center;justify-content:space-between;
     margin-bottom:16px;box-shadow:0 6px 24px rgba(124,58,237,0.3);'>
  <div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.78rem;font-weight:600;
         text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;'>
      🏆 Best Performing Model
    </div>
    <div style='color:#fff;font-size:1.6rem;font-weight:900;'>{winner}</div>
    <div style='color:rgba(255,255,255,0.7);font-size:0.85rem;margin-top:2px;'>
      Accuracy: <b style='color:#a5f3fc'>{results[winner]['Acc']}%</b> &nbsp;|&nbsp;
      MAE: <b style='color:#a5f3fc'>PKR {results[winner]['MAE']:.0f}</b>
    </div>
  </div>
  <div style='text-align:right'>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;margin-bottom:4px;'>vs {loser}</div>
    <div style='color:#fde68a;font-size:1.2rem;font-weight:800;'>
      +{abs(results[winner]['Acc']-results[loser]['Acc']):.1f}% better
    </div>
    <div style='color:rgba(255,255,255,0.5);font-size:0.75rem;'>accuracy advantage</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 4 metric cards ────────────────────────────────────────────────────────────
mc = st.columns(4)
metrics = [
    ("Random Forest", "Accuracy",         f"{rf_res['Acc']}%",         "Higher is better"),
    ("Random Forest", "MAE (Avg Error)",  f"PKR {rf_res['MAE']:.0f}",  "Lower is better"),
    ("XGBoost",       "Accuracy",         f"{xgb_res['Acc']}%",        "Higher is better"),
    ("XGBoost",       "MAE (Avg Error)",  f"PKR {xgb_res['MAE']:.0f}", "Lower is better"),
]
for i,(mname,lbl,val,hint) in enumerate(metrics):
    border = "border:2px solid #a5f3fc;" if mname==winner else ""
    with mc[i]:
        st.markdown(f"""<div class='model-card' style='{border}'>
            <div class='mc-name'>{mname}</div>
            <div class='mc-val'>{val}</div>
            <div class='mc-lbl'>{lbl} — {hint}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Side-by-side bar chart comparison ────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 Head-to-Head Model Comparison Charts</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Accuracy comparison bar chart
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name='Random Forest', x=['Accuracy (%)'],
        y=[rf_res['Acc']], marker_color='#7c3aed',
        text=[f"{rf_res['Acc']}%"], textposition='outside',
        textfont=dict(color='#1e3a5f', size=13, family='Arial'),
        width=0.3
    ))
    fig_cmp.add_trace(go.Bar(
        name='XGBoost', x=['Accuracy (%)'],
        y=[xgb_res['Acc']], marker_color='#0891b2',
        text=[f"{xgb_res['Acc']}%"], textposition='outside',
        textfont=dict(color='#1e3a5f', size=13, family='Arial'),
        width=0.3
    ))
    fig_cmp.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=280, barmode='group',
        title=dict(text="Accuracy Comparison (%)", font_color='#1e3a5f', font_size=13),
        yaxis=dict(range=[0,100], showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        xaxis=dict(showgrid=False, color=TICK_CLR),
        legend=dict(font_color='#334155', bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0,r=0,t=40,b=10)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

with chart_col2:
    # MAE comparison bar chart
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Bar(
        name='Random Forest', x=['MAE (PKR)'],
        y=[rf_res['MAE']], marker_color='#7c3aed',
        text=[f"PKR {rf_res['MAE']:.0f}"], textposition='outside',
        textfont=dict(color='#1e3a5f', size=13, family='Arial'),
        width=0.3
    ))
    fig_mae.add_trace(go.Bar(
        name='XGBoost', x=['MAE (PKR)'],
        y=[xgb_res['MAE']], marker_color='#0891b2',
        text=[f"PKR {xgb_res['MAE']:.0f}"], textposition='outside',
        textfont=dict(color='#1e3a5f', size=13, family='Arial'),
        width=0.3
    ))
    fig_mae.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=280, barmode='group',
        title=dict(text="MAE Comparison — lower is better", font_color='#1e3a5f', font_size=13),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=TICK_CLR),
        xaxis=dict(showgrid=False, color=TICK_CLR),
        legend=dict(font_color='#334155', bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=0,r=0,t=40,b=10)
    )
    st.plotly_chart(fig_mae, use_container_width=True)

# ── Per-stock accuracy comparison table ───────────────────────────────────────
st.markdown('<div class="sec-hdr">📋 Per-Company Accuracy Breakdown</div>', unsafe_allow_html=True)

rf_ps  = rf_res.get('per_stock',  {})
xgb_ps = xgb_res.get('per_stock', {})
all_tickers_ps = sorted(set(list(rf_ps.keys()) + list(xgb_ps.keys())))

fig_per = go.Figure()
names_ps  = [ticker_names.get(t, t) for t in all_tickers_ps]
rf_vals   = [rf_ps.get(t,  0) for t in all_tickers_ps]
xgb_vals  = [xgb_ps.get(t, 0) for t in all_tickers_ps]

fig_per.add_trace(go.Bar(
    name='Random Forest', x=names_ps, y=rf_vals,
    marker_color='#7c3aed', opacity=0.85,
    text=[f"{v}%" for v in rf_vals], textposition='outside',
    textfont=dict(size=10, color='#1e3a5f')
))
fig_per.add_trace(go.Bar(
    name='XGBoost', x=names_ps, y=xgb_vals,
    marker_color='#0891b2', opacity=0.85,
    text=[f"{v}%" for v in xgb_vals], textposition='outside',
    textfont=dict(size=10, color='#1e3a5f')
))
# Benchmark reference line
fig_per.add_hline(
    y=85, line_dash="dash", line_color="#dc2626", line_width=1.5,
    annotation_text="Research benchmark (85%)",
    annotation_font_color="#dc2626", annotation_font_size=11
)
fig_per.update_layout(
    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
    font_color=FONT_CLR, height=360, barmode='group',
    title=dict(text="Accuracy per Company — Random Forest vs XGBoost",
               font_color='#1e3a5f', font_size=13),
    yaxis=dict(range=[0,115], showgrid=True, gridcolor=GRID_CLR,
               color=TICK_CLR, title="Accuracy (%)"),
    xaxis=dict(showgrid=False, color=TICK_CLR, tickangle=-25,
               tickfont=dict(size=10, color='#1e3a5f')),
    legend=dict(font_color='#334155', bgcolor='rgba(255,255,255,0.8)',
                orientation='h', y=1.12),
    margin=dict(l=10,r=10,t=60,b=80)
)
st.plotly_chart(fig_per, use_container_width=True)

# ── Summary comparison table ──────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📌 Model Summary Table</div>', unsafe_allow_html=True)
table_rows = ""
row_data = [
    ("Algorithm Type",   "Ensemble — 100 parallel trees",     "Boosting — 200 sequential trees"),
    ("Accuracy",         f"{rf_res['Acc']}%",                 f"{xgb_res['Acc']}%"),
    ("MAE (Avg Error)",  f"PKR {rf_res['MAE']:.0f}",          f"PKR {xgb_res['MAE']:.0f}"),
    ("RMSE",             f"PKR {rf_res['RMSE']:.0f}",         f"PKR {xgb_res['RMSE']:.0f}"),
    ("Speed",            "Faster to train",                   "Slightly slower"),
    ("Overfitting Risk", "Lower risk",                        "Medium risk"),
    ("Interpretability", "Feature importance available",      "Feature importance available"),
    ("Best For",         "Quick & reliable baseline",         "Higher accuracy final model"),
    ("Winner?",
     "✅ Yes" if winner=="Random Forest" else "❌ No",
     "✅ Yes" if winner=="XGBoost"       else "❌ No"),
]
for label, rf_val, xgb_val in row_data:
    is_winner_row = label == "Winner?"
    rf_bg  = "#dcfce7" if (is_winner_row and winner=="Random Forest") else "#f8fafc"
    xgb_bg = "#dcfce7" if (is_winner_row and winner=="XGBoost")       else "#f8fafc"
    table_rows += f"""
    <tr>
      <td style='padding:10px 14px;font-size:0.82rem;font-weight:700;color:#475569;
                 border-bottom:1px solid #e2e8f0;background:#f1f5f9;'>{label}</td>
      <td style='padding:10px 14px;font-size:0.82rem;color:#1e3a5f;
                 border-bottom:1px solid #e2e8f0;background:{rf_bg};'>{rf_val}</td>
      <td style='padding:10px 14px;font-size:0.82rem;color:#1e3a5f;
                 border-bottom:1px solid #e2e8f0;background:{xgb_bg};'>{xgb_val}</td>
    </tr>"""

st.markdown(f"""
<div style='background:#fff;border-radius:16px;overflow:hidden;
     box-shadow:0 4px 20px rgba(30,58,95,0.10);border:1px solid rgba(30,58,95,0.08);'>
  <table style='width:100%;border-collapse:collapse;'>
    <thead>
      <tr>
        <th style='padding:12px 14px;background:#1e3a5f;color:#fff;font-size:0.82rem;
                   text-align:left;width:30%'>Metric</th>
        <th style='padding:12px 14px;background:#7c3aed;color:#fff;font-size:0.82rem;
                   text-align:left;width:35%'>🌲 Random Forest</th>
        <th style='padding:12px 14px;background:#0891b2;color:#fff;font-size:0.82rem;
                   text-align:left;width:35%'>⚡ XGBoost</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SECTOR + RSI CHARTS ───────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🗂️ Market Overview</div>', unsafe_allow_html=True)
pc, bc = st.columns(2)

with pc:
    sec_df = pd.DataFrame({'Sector': list(ticker_sectors.values())})
    sc = sec_df['Sector'].value_counts().reset_index()
    sc.columns = ['Sector','Count']
    fig_pie = px.pie(sc, names='Sector', values='Count', hole=0.5,
                     color_discrete_sequence=['#4f46e5','#0891b2','#15803d','#ea580c','#dc2626','#7c3aed'])
    fig_pie.update_layout(
        paper_bgcolor=PAPER_BG, font_color=FONT_CLR, height=300,
        margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text="Sector Breakdown", font_color='#1e3a5f', font_size=14),
        legend=dict(font_color='#334155')
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with bc:
    rsi_rows = [{'Co': ticker_names[t][:12],
                 'RSI': round(df_processed[df_processed['Ticker']==t]['RSI'].iloc[-1],1)}
                for t in data]
    rsi_df = pd.DataFrame(rsi_rows)
    bar_clr = ['#15803d' if r<40 else ('#dc2626' if r>60 else '#d97706') for r in rsi_df['RSI']]
    fig_b = go.Figure(go.Bar(x=rsi_df['Co'], y=rsi_df['RSI'],
                              marker_color=bar_clr,
                              text=rsi_df['RSI'], textposition='outside',
                              textfont=dict(color='#1e3a5f', size=11)))
    fig_b.add_hline(y=70, line_dash="dash", line_color="rgba(220,38,38,0.6)",
                    annotation_text="Overbought", annotation_font_color="#dc2626")
    fig_b.add_hline(y=30, line_dash="dash", line_color="rgba(21,128,61,0.6)",
                    annotation_text="Oversold",   annotation_font_color="#15803d")
    fig_b.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_CLR, height=320,
        title=dict(text="RSI — All 10 Companies", font_color='#1e3a5f', font_size=14),
        margin=dict(l=0,r=60,t=40,b=90),
        xaxis=dict(tickangle=-35, showgrid=False, color=TICK_CLR,
                   tickfont=dict(size=10, color='#1e3a5f')),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, range=[0,115], color=TICK_CLR)
    )
    st.plotly_chart(fig_b, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">💡 Top 5 Stock Recommendations — {selected_model}</div>', unsafe_allow_html=True)
st.markdown(f"<span style='color:#64748b;font-size:0.84rem;font-weight:600'>Excluding: {ticker_names.get(selected_stock)}</span>",
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

recs = recommend_stocks([selected_stock], models[selected_model])
if recs:
    rcols = st.columns(5)
    for i, r in enumerate(recs):
        with rcols[i]:
            st.markdown(f"""
            <div class='rec-card'>
                <div class='r-name'>{r['name']}</div>
                <div class='r-sector'>{r['sector']}</div>
                <div class='r-return'>{r['predicted_return']}%</div>
                <div class='r-price'>PKR {r['last_close']:,.0f}</div>
                <div style='margin-top:10px'>
                    <span class='b-{r["signal"].lower()}'>{r['signal']}</span>
                </div>
                <div style='color:#94a3b8;font-size:0.72rem;margin-top:6px;font-weight:600'>
                    RSI {r['rsi']}
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.78rem;padding:10px 0 20px 0;font-weight:500'>
📌 PSX AI Stock Predictor &nbsp;|&nbsp; Final Year Project &nbsp;|&nbsp;
Streamlit · scikit-learn · Plotly &nbsp;|&nbsp; Data: Yahoo Finance<br>
⚠️ For educational purposes only. Not financial advice.
</div>""", unsafe_allow_html=True)

