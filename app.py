import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# ==============================
# 🎨 PREMIUM STYLING
# ==============================

st.markdown("""
<style>
.main-title {
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    color: #ffffff;  /* ✅ WHITE TEXT */
    text-shadow: 0px 0px 12px rgba(255,255,255,0.6); /* 🔥 glow */
}

.sub-text {
    text-align: center;
    color: #ffffff;  /* ✅ WHITE TEXT */
    margin-bottom: 20px;
    opacity: 0.85;
}

.card {
    background-color: #111;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 12px rgba(255,255,255,0.1);
    color: white;  /* ✅ IMPORTANT (fixes card text too) */
}

hr {
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.markdown("<div class='main-title'>📈 AI Stock Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Built by Kush 🚀 | Smart Market Insights</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# COMPANY MAP
# ==============================

company_map = {
    # Indian IT
    "tcs": "TCS.NS","infosys": "INFY.NS","wipro": "WIPRO.NS","hcl": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS","ltimindtree": "LTIM.NS",

    # Indian Banks & Finance
    "hdfc": "HDFCBANK.NS","icici": "ICICIBANK.NS","sbi": "SBIN.NS",
    "axis": "AXISBANK.NS","kotak": "KOTAKBANK.NS","bank of baroda": "BANKBARODA.NS",
    "bajaj finance": "BAJFINANCE.NS","bajaj finserv": "BAJAJFINSV.NS",

    # Indian Conglomerates
    "reliance": "RELIANCE.NS","adani": "ADANIENT.NS","adani ports": "ADANIPORTS.NS",
    "tata": "TATASTEEL.NS","l&t": "LT.NS",

    # Auto
    "tata motors": "TATAMOTORS.NS","maruti": "MARUTI.NS","mahindra": "M&M.NS",
    "eicher": "EICHERMOT.NS","hero": "HEROMOTOCO.NS","bajaj auto": "BAJAJ-AUTO.NS",

    # FMCG
    "itc": "ITC.NS","hul": "HINDUNILVR.NS","nestle india": "NESTLEIND.NS",
    "britannia": "BRITANNIA.NS","dabur": "DABUR.NS",

    # Global Tech
    "apple": "AAPL","tesla": "TSLA","microsoft": "MSFT",
    "google": "GOOGL","amazon": "AMZN","nvidia": "NVDA",
    "meta": "META","netflix": "NFLX",

    # Global Finance
    "jpmorgan": "JPM","goldman sachs": "GS","morgan stanley": "MS",
    "visa": "V","mastercard": "MA",

    # Consumer
    "nike": "NKE","adidas": "ADDYY","coca cola": "KO","pepsi": "PEP",
    "philip morris": "PM","starbucks": "SBUX","mcdonalds": "MCD"
}

brand_map = {
    # Meta
    "instagram": "meta","whatsapp": "meta","facebook": "meta","threads": "meta",

    # Google
    "youtube": "google","gmail": "google","android": "google","maps": "google","chrome": "google",

    # Apple
    "iphone": "apple","macbook": "apple","ipad": "apple","airpods": "apple","apple watch": "apple",

    # Amazon
    "aws": "amazon","prime": "amazon","alexa": "amazon","kindle": "amazon",

    # Tesla
    "model 3": "tesla","model s": "tesla","model x": "tesla","model y": "tesla",

    # Microsoft
    "windows": "microsoft","azure": "microsoft","xbox": "microsoft","office": "microsoft",

    # Indian Brands
    "jio": "reliance","airtel": "bharti airtel","vi": "vodafone idea",
    "paytm": "paytm","phonepe": "flipkart","gpay": "google",

    # Auto Brands
    "range rover": "tata motors","jaguar": "tata motors",
    "kia": "hyundai","hyundai": "hyundai",
    "royal enfield": "eicher",

    # FMCG Brands
    "surf excel": "hul","lifebuoy": "hul","lux": "hul",
    "maggi": "nestle india","kitkat": "nestle india",
    "good day": "britannia","real juice": "dabur",

    # Others
    "marlboro": "philip morris","gatorade": "pepsi",
    "fanta": "coca cola","sprite": "coca cola"
}

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📊 Dashboard")

user_input = st.sidebar.text_input("🔍 Search Stock")

ticker = None

if user_input:
    key = user_input.lower().strip()

    if key in company_map:
        ticker = company_map[key]
    elif key in brand_map:
        ticker = company_map.get(brand_map[key])
    else:
        ticker = user_input.upper()

    st.sidebar.success(f"{ticker}")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open','High','Low','Close','Volume']]
        df.reset_index(inplace=True)
        return df
    except:
        return None

# ==============================
# INDICATORS
# ==============================

def calculate_rsi(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    return macd, signal

def calculate_confidence(df):
    latest = df.iloc[-1]
    score = 0

    score += 30 if latest['Close'] > latest['MA50'] else 10

    if latest['RSI'] < 30:
        score += 30
    elif latest['RSI'] > 70:
        score += 10
    else:
        score += 20

    score += 30 if latest['MACD'] > latest['MACD_Signal'] else 10

    return score

def get_signal(df):
    latest = df.iloc[-1]
    return "BUY 📈" if latest['Close'] > latest['MA50'] else "SELL 📉"

# ==============================
# ENTRY / EXIT SIGNALS
# ==============================

def generate_signals(df):
    df['Entry'] = 0
    df['Exit'] = 0

    for i in range(1, len(df)):
        if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]) and \
           (df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]) and \
           (df['Close'].iloc[i] > df['MA50'].iloc[i]):
            df.at[i, 'Entry'] = df['Close'].iloc[i]

        if (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]) and \
           (df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]) and \
           (df['Close'].iloc[i] < df['MA50'].iloc[i]):
            df.at[i, 'Exit'] = df['Close'].iloc[i]

    return df

# ==============================
# 💱 CURRENCY DETECTION
# ==============================

def get_currency_symbol(ticker):
    if ticker and ticker.endswith(".NS"):
        return "₹"
    return "$"

# ==============================
# MAIN
# ==============================

if ticker:

    df = load_data(ticker)

    if df is None:
        st.error("❌ Invalid ticker or no data")
        st.stop()

    df['MA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df)

    # ✅ FIXED DROPNA
    df = df.dropna(subset=['Close'])

    # ✅ SAFETY CHECK
    if df.empty:
        st.error("❌ No valid data available")
        st.stop()

    # ✅ ADD SIGNALS
    df = generate_signals(df)

    current = df['Close'].iloc[-1]
    currency = get_currency_symbol(ticker)
    
    signal = get_signal(df)
    confidence = calculate_confidence(df)

    # ==============================
    # METRICS
    # ==============================

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'>💰 Price<br><h2>{currency}{current:.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>📈 Signal<br><h2>{signal}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>🤖 Confidence<br><h2>{confidence}%</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # MAIN CHART
    # ==============================

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="MA50"))

    # ENTRY
    fig.add_trace(go.Scatter(
        x=df[df['Entry'] > 0]['Date'],
        y=df[df['Entry'] > 0]['Entry'],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12),
        name="ENTRY"
    ))

    # EXIT
    fig.add_trace(go.Scatter(
        x=df[df['Exit'] > 0]['Date'],
        y=df[df['Exit'] > 0]['Exit'],
        mode='markers',
        marker=dict(symbol='triangle-down', size=12),
        name="EXIT"
    ))

    fig.update_layout(template="plotly_dark", height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # RSI + MACD
    # ==============================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 RSI")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI']))
        fig_rsi.add_hline(y=70)
        fig_rsi.add_hline(y=30)
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        st.subheader("📊 MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD']))
        fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal']))
        st.plotly_chart(fig_macd, use_container_width=True)

    # ==============================
    # AI INSIGHT
    # ==============================

    st.markdown("### 🤖 AI Insight")

    if confidence > 70:
        st.success("Strong bullish signal 🚀")
    elif confidence > 50:
        st.warning("Moderate confidence ⚠️")
    else:
        st.error("Weak / risky signal ❌")

else:
    st.info("👈 Search stock from sidebar")