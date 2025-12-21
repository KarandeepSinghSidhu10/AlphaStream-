import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# --- 2. HELPER FUNCTIONS ---

@st.cache_data
def load_data(symbol, period):
    """Fetches stock data and cleans it for plotting."""
    try:
        data = yf.download(symbol, period=period)
        
        # Flatten MultiIndex columns (e.g., 'Close' -> 'AAPL' becomes just 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

@st.cache_data
def get_company_info(symbol):
    """Fetches company profile."""
    try:
        company = yf.Ticker(symbol)
        return company.info
    except Exception as e:
        return None

@st.cache_data
def get_stock_news(symbol):
    """
    Fetches news from Yahoo Finance RSS feed.
    This is more robust than yfinance.news which often breaks.
    """
    try:
        # Standard Yahoo Finance RSS URL
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        
        # We need a browser-like header or Yahoo might block the request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        
        # Parse the XML response
        root = ET.fromstring(response.content)
        news_items = []
        
        # Loop through the first 5 news items
        for item in root.findall('.//item')[:5]:
            news_items.append({
                'title': item.find('title').text,
                'link': item.find('link').text,
                'pubDate': item.find('pubDate').text
            })
            
        return news_items
    except Exception as e:
        return []

# --- 3. SIDEBAR ---
st.sidebar.header("Dashboard Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
time_period = st.sidebar.selectbox("Time Range", ["1mo", "3mo", "6mo", "1y", "5y", "max"])

# Add a spinner for the sidebar info load
with st.sidebar:
    with st.spinner("Loading info..."):
        info = get_company_info(ticker)

st.sidebar.markdown("---")
st.sidebar.subheader("Company Profile")

if info and 'longName' in info:
    st.sidebar.markdown(f"**{info['longName']}**")
    st.sidebar.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.sidebar.write(f"**Industry:** {info.get('industry', 'N/A')}")
    
    market_cap = info.get('marketCap', 'N/A')
    if isinstance(market_cap, int):
        if market_cap > 1e12:
            mc_str = f"${market_cap/1e12:.2f} T"
        else:
            mc_str = f"${market_cap/1e9:.2f} B"
    else:
        mc_str = "N/A"
    st.sidebar.write(f"**Market Cap:** {mc_str}")
    
    with st.sidebar.expander("Business Summary"):
        st.sidebar.write(info.get('longBusinessSummary', 'No summary available.'))
else:
    st.sidebar.warning("Company info not found. Check ticker.")

# --- 4. MAIN DASHBOARD ---
st.title(f"📈 {ticker} Market Dashboard")

# Add spinner for main data loading
with st.spinner("Fetching market data..."):
    stock_data = load_data(ticker, time_period)
    benchmark_data = load_data("^GSPC", time_period)

if stock_data is not None and not stock_data.empty:
    
    # Metrics
    last_close = float(stock_data['Close'].iloc[-1])
    prev_close = float(stock_data['Close'].iloc[-2])
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${last_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    col2.metric("High", f"${float(stock_data['High'].max()):.2f}")
    col3.metric("Low", f"${float(stock_data['Low'].min()):.2f}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Price Chart", "Market Comparison", "Latest News"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], 
                                 mode='lines', name=ticker, line=dict(color='#1f77b4', width=2)))
        fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price", 
                          height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### vs. S&P 500 Index (Percentage Growth)")
        if benchmark_data is not None:
            # IMPROVEMENT: Merge dataframes to ensure dates align perfectly
            # 1. Prepare temp dataframes with clearer column names
            stock_df = stock_data[['Date', 'Close']].copy()
            stock_df.columns = ['Date', 'Stock_Close']
            
            bench_df = benchmark_data[['Date', 'Close']].copy()
            bench_df.columns = ['Date', 'Bench_Close']
            
            # 2. Merge on Date (inner join keeps only dates present in BOTH)
            merged_df = pd.merge(stock_df, bench_df, on='Date', how='inner')
            
            # 3. Calculate cumulative return starting from the first available date
            merged_df['Stock_Return'] = (merged_df['Stock_Close'] / merged_df['Stock_Close'].iloc[0] - 1) * 100
            merged_df['Bench_Return'] = (merged_df['Bench_Close'] / merged_df['Bench_Close'].iloc[0] - 1) * 100
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Stock_Return'], 
                                          name=ticker, line=dict(color='#1f77b4', width=2)))
            fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Bench_Return'], 
                                          name='S&P 500', line=dict(color='gray', width=2, dash='dot')))
            fig_comp.update_layout(xaxis_title="Date", yaxis_title="Growth (%)", height=500, template="plotly_dark")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Benchmark data could not be loaded.")

    with tab3:
        st.subheader(f"News about {ticker}")
        with st.spinner("Fetching news..."):
            news_items = get_stock_news(ticker)
        
        if news_items:
            for item in news_items:
                st.markdown(f"### [{item['title']}]({item['link']})")
                st.caption(f"Published: {item['pubDate']}")
                st.markdown("---")
        else:
            st.write("No news found (or connection to Yahoo RSS failed).")

else:
    st.error("Error loading data. Please check the ticker symbol.")