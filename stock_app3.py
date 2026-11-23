import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- LOGIC START ----
POPULAR = {
    "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS", "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS", "Bajaj Finance": "BAJFINANCE.NS", "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS", "Sun Pharma": "SUNPHARMA.NS", "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS", "Asian Paints": "ASIANPAINT.NS", "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS", "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS",
    "Zomato (Eternal)": "ETERNAL.NS", "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Tata Motors Commercial Vehicles": "TMCV.NS", "Tata Steel": "TATASTEEL.NS",
    "Tata Power": "TATAPOWER.NS", "Tata Consumer Products": "TATACONSUM.NS", "Trent": "TRENT.NS",
}

RISK_FREE = 0.065  # approx. Indian 10-year government bond yield


def scale(x, lo, hi):
    """Map x to 0-100 (lo -> 0, hi -> 100), clipped."""
    return float(np.clip((x - lo) / (hi - lo) * 100, 0, 100))


def compute_metrics(df, market_close=None):
    close = df["Close"]
    returns = close.pct_change().dropna()

    years = max((close.index[-1] - close.index[0]).days / 365.25, 0.1)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    max_dd = (close / close.cummax() - 1).min()
    sharpe = (cagr - RISK_FREE) / vol if vol > 0 else 0.0

    sma200 = close.rolling(200).mean().iloc[-1]
    above_200 = bool(close.iloc[-1] > sma200) if not np.isnan(sma200) else None

    beta = None
    if market_close is not None:
        m = market_close.pct_change().dropna()
        joined = pd.concat([returns, m], axis=1, join="inner").dropna()
        if len(joined) > 30:
            joined.columns = ["s", "m"]
            beta = joined["s"].cov(joined["m"]) / joined["m"].var()

    return {
        "cagr": cagr, "vol": vol, "max_dd": max_dd, "sharpe": sharpe,
        "above_200": above_200, "beta": beta,
    }


def trust_score(m):
    """Combine metrics into a 0-100 score. Higher = more stable / trusted."""
    parts = {
        "Volatility (lower is better)": (scale(-m["vol"], -0.50, -0.15), 25),
        "Max drawdown (smaller fall is better)": (scale(m["max_dd"], -0.60, 0.0), 25),
        "Yearly return (CAGR)": (scale(m["cagr"], -0.10, 0.25), 15),
        "Sharpe ratio": (scale(m["sharpe"], -0.5, 1.5), 15),
        "Trend (price vs 200-day avg)": (
            50.0 if m["above_200"] is None else (100.0 if m["above_200"] else 30.0), 10),
        "Beta (sensitivity to the market)": (
            50.0 if m["beta"] is None else scale(-m["beta"], -1.6, -0.7), 10),
    }
    total = sum(s * w for s, w in parts.values()) / sum(w for _, w in parts.values())
    return total, parts


def verdict(score):
    if score >= 70:
        return "LOW RISK", "green", ("Based on past data, this stock has been fairly stable. "
                                      "It generally falls in the safer category for long-term investing.")
    if score >= 45:
        return "MEDIUM RISK", "orange", ("This stock shows both positive and risky signs. Investing in small "
                                          "amounts over time and diversifying would be wiser.")
    return "HIGH RISK", "red", ("This stock has had large swings or deep falls. "
                                 "Only invest money you can afford to lose.")
# ---- LOGIC END ----


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def search_companies(query):
    """Return {label: ticker}. Yahoo search finds ANY listed company; POPULAR is a fallback."""
    q = query.strip().lower()
    found = {}  # ticker -> label
    for name, t in POPULAR.items():
        if q in name.lower() or q in t.lower():
            found[t] = name
    try:
        for r in yf.Search(query, max_results=20, news_count=0).quotes:
            sym = r.get("symbol", "")
            if r.get("quoteType") == "EQUITY" and sym.endswith((".NS", ".BO")):
                found.setdefault(sym, r.get("longname") or r.get("shortname") or sym)
    except Exception:
        pass  # offline / Yahoo issue -> POPULAR matches still work
    ordered = sorted(found, key=lambda t: (not t.endswith(".NS"), t))  # NSE first
    return {f"{found[t]} [{t}]": t for t in ordered}


st.set_page_config(page_title="StockVision AI", page_icon="📈", layout="wide")
st.title("📈 StockVision AI")
st.caption("Enter any NSE/BSE company name → see its history, chart, trust score and risk level.")

with st.sidebar:
    st.header("Search")
    query = st.text_input("Company name", value="Reliance", help="Any listed company name: Tata, Zomato, Wipro, Pidilite...")
    period = st.selectbox("History", ["1y", "3y", "5y", "10y"], index=2)
    matches = search_companies(query) if query.strip() else {}
    ticker, name = None, None
    if matches:
        name = st.selectbox("Which company?", list(matches))
        ticker = matches[name]
    elif query.strip():
        st.warning("No company found. Check the spelling or enter the ticker below.")
    custom = st.text_input("Or enter a ticker directly (e.g. WIPRO.NS)")
    if custom.strip():
        ticker, name = custom.strip().upper(), custom.strip().upper()

if not ticker:
    st.info("Type a company name in the sidebar.")
    st.stop()

with st.spinner("Fetching data..."):
    df = load(ticker, period)
    market = load("^NSEI", period)

if df.empty or len(df) < 60:
    st.error(f"No data found for {ticker} (or too little of it). Check the ticker "
             "or your internet connection.")
    st.stop()

st.subheader(f"{name}  ({ticker})")

last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close (₹)", f"{last:,.2f}", f"{(last / prev - 1) * 100:.2f}%")
c2.metric(f"{period} High (₹)", f"{df['High'].max():,.2f}")
c3.metric(f"{period} Low (₹)", f"{df['Low'].min():,.2f}")
c4.metric("Avg daily volume", f"{df['Volume'].mean():,.0f}")

# ---- Chart ----
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                    vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(),
                         name="50-day avg", line=dict(width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(200).mean(),
                         name="200-day avg", line=dict(width=1.5)), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

# ---- Trust / risk ----
m = compute_metrics(df, market["Close"] if not market.empty else None)
score, parts = trust_score(m)
label, color, advice = verdict(score)

st.divider()
st.subheader("Trust score & Risk level")
g1, g2 = st.columns([1, 1.4])
with g1:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "steps": [{"range": [0, 45], "color": "rgba(255,0,0,0.12)"},
                         {"range": [45, 70], "color": "rgba(255,165,0,0.12)"},
                         {"range": [70, 100], "color": "rgba(0,160,0,0.12)"}]}))
    gauge.update_layout(height=280, margin=dict(t=30, b=10))
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(f"### :{color}[{label}]")
    st.write(advice)
with g2:
    beta_txt = "n/a" if m["beta"] is None else f"{m['beta']:.2f}"
    st.dataframe(pd.DataFrame({
        "Measure": ["Yearly return (CAGR)", "Volatility (annual)", "Max drawdown",
                    "Sharpe ratio", "Beta vs Nifty 50", "Above 200-day avg?"],
        "Value": [f"{m['cagr'] * 100:.1f}%", f"{m['vol'] * 100:.1f}%",
                  f"{m['max_dd'] * 100:.1f}%", f"{m['sharpe']:.2f}", beta_txt,
                  "n/a" if m["above_200"] is None else ("Yes" if m["above_200"] else "No")],
    }), hide_index=True, use_container_width=True)
    with st.expander("How is the score calculated?"):
        st.dataframe(pd.DataFrame(
            [(k, round(s), w) for k, (s, w) in parts.items()],
            columns=["Factor", "Score (0-100)", "Weight %"]),
            hide_index=True, use_container_width=True)

st.warning("⚠️ This is an educational tool based only on past data, not financial advice. "
           "Past performance is no guarantee of future results. Consult a "
           "SEBI-registered advisor before investing.")

# ---- History table ----
with st.expander("Price history table"):
    hist = df.sort_index(ascending=False)
    st.dataframe(hist, use_container_width=True)
    st.download_button("Download CSV", hist.to_csv().encode(), f"{ticker}_history.csv", "text/csv")
