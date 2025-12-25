import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="AlphaStream Pro", layout="wide")

# --- MODULE 1: DATA LAYER (Scalability) ---
class StockDataManager:
    """Handles all data fetching and preprocessing."""
    
    @staticmethod
    @st.cache_data
    def fetch_stock_history(symbol, period):
        """Fetches stock data and cleans it."""
        try:
            # Check if valid symbol by attempting to fetch info first could be slow,
            # so we try/except the download.
            data = yf.download(symbol, period=period)
            
            if data.empty:
                return None

            # Flatten MultiIndex columns if necessary
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    @st.cache_data
    def fetch_company_info(symbol):
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception:
            return None

    @staticmethod
    @st.cache_data
    def fetch_news(symbol):
        """Fetches news from RSS feed."""
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            root = ET.fromstring(response.content)
            news_items = []
            for item in root.findall('.//item')[:5]:
                news_items.append({
                    'title': item.find('title').text,
                    'link': item.find('link').text,
                    'pubDate': item.find('pubDate').text
                })
            return news_items
        except Exception:
            return []

# --- MODULE 2: ANALYTICS LAYER (Probability & AI) ---
class MarketAnalyzer:
    """Handles technical analysis and generates insights."""
    
    @staticmethod
    def calculate_technical_indicators(df):
        """Adds RSI and SMA to the dataframe."""
        df = df.copy()
        # Simple Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def calculate_success_probability(df):
        """
        Calculates a 'Success Probability' score based on technical factors.
        Returns: (probability_float, reason_string)
        """
        if len(df) < 200:
            return 50.0, "Insufficient data for full analysis."

        current_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1]
        
        score = 50 # Base neutral score
        reasons = []

        # 1. Trend (Weight: 30%)
        if current_price > sma_50:
            score += 15
            reasons.append("Price is in a short-term uptrend (Above 50 SMA).")
        else:
            score -= 15
            reasons.append("Price is in a short-term downtrend (Below 50 SMA).")
            
        # 2. Golden Cross / Death Cross (Weight: 20%)
        if sma_50 > sma_200:
            score += 10
            reasons.append("Long-term bullish signal (Golden Cross).")
        elif sma_50 < sma_200:
            score -= 10
            reasons.append("Long-term bearish signal (Death Cross).")

        # 3. Momentum / RSI (Weight: 50%)
        # Buying opportunity if oversold (RSI < 30)
        if rsi < 30:
            score += 25
            reasons.append("Asset is Oversold (RSI < 30). Potential rebound.")
        # Selling risk if overbought (RSI > 70)
        elif rsi > 70:
            score -= 25
            reasons.append("Asset is Overbought (RSI > 70). Correction likely.")
        else:
            reasons.append("Momentum is neutral.")

        return max(0, min(100, score)), reasons

    @staticmethod
    def generate_ai_commentary(symbol, df, probability, reasons):
        """
        Simulates an AI 'Market Interpreter' that explains the chart in plain English.
        """
        current_price = df['Close'].iloc[-1]
        start_price = df['Close'].iloc[0]
        total_change = ((current_price - start_price) / start_price) * 100
        
        trend_word = "rising" if total_change > 0 else "falling"
        feeling = "positive" if probability > 60 else ("risky" if probability < 40 else "uncertain")
        
        narrative = f"""
        **Market Interpreter for {symbol}:**
        
        Right now, the stock is generally **{trend_word}**. Over the selected period, it has moved by **{total_change:.1f}%**.
        
        Based on our automated analysis, the market looks **{feeling}** (Confidence: {probability}%).
        
        **Why?**
        * {' '.join(reasons[:2])}
        
        *Simply put:* If you are a beginner, this indicates that {'buyers are currently in control' if probability > 50 else 'sellers are currently in control'}.
        ALWAYS do your own research.
        """
        return narrative

# --- MODULE 3: UI LAYER ---
class DashboardUI:
    """Manages the frontend layout and interaction."""
    
    def __init__(self):
        self.data_manager = StockDataManager()
        self.analyzer = MarketAnalyzer()
        
    def render_sidebar(self):
        st.sidebar.header("Dashboard Settings")
        ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
        period = st.sidebar.selectbox("Time Range", ["6mo", "1y", "5y", "max"])
        
        # Load Info
        with st.sidebar:
            with st.spinner("Loading info..."):
                info = self.data_manager.fetch_company_info(ticker)
        
        if info:
            st.sidebar.subheader(info.get('longName', ticker))
            st.sidebar.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.sidebar.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.sidebar.info(info.get('longBusinessSummary', 'No summary available.')[:300] + "...")
            
        return ticker, period

    def render_metrics(self, df):
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${last_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
        col2.metric("High (Period)", f"${df['High'].max():.2f}")
        col3.metric("Low (Period)", f"${df['Low'].min():.2f}")

    def render_charts(self, ticker, df, benchmark_df):
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Market Comparison", "Latest News"])
        
        with tab1:
            # Enhanced Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker,
                # Professional trading colors (Green for Up, Red for Down)
                increasing_line_color='#26a69a', 
                decreasing_line_color='#ef5350'
            ))
            
            fig.update_layout(
                title=f"{ticker} Price Action",
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                template="plotly_dark",
                height=600,  # Increased height
                xaxis_rangeslider_visible=False,  # Hides the slider to prevent squishing
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if benchmark_df is not None:
                # Normalize for comparison
                df['Norm'] = df['Close'] / df['Close'].iloc[0] * 100
                benchmark_df['Norm'] = benchmark_df['Close'] / benchmark_df['Close'].iloc[0] * 100
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=df['Date'], y=df['Norm'], name=ticker))
                fig_comp.add_trace(go.Scatter(x=benchmark_df['Date'], y=benchmark_df['Norm'], name="S&P 500", line=dict(dash='dot')))
                fig_comp.update_layout(title="Relative Performance (Base=100)", height=500, template="plotly_dark")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Benchmark data unavailable.")

        with tab3:
            news = self.data_manager.fetch_news(ticker)
            for item in news:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(item['pubDate'])
                st.divider()

    def render_ai_modal(self, symbol, df, prob, reasons):
        """Displays the 'AI' explanation in an expander or sidebar."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Market Interpreter")
        
        if st.sidebar.button("Explain this Chart"):
            explanation = self.analyzer.generate_ai_commentary(symbol, df, prob, reasons)
            st.sidebar.success(explanation)

    def render_probability_widget(self, probability):
        """Displays the success probability metric."""
        st.markdown("### 🎲 Trade Probability Engine")
        
        col1, col2 = st.columns([1, 3])
        
        color = "green" if probability > 60 else "red" if probability < 40 else "orange"
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid {color}; padding: 10px; border-radius: 10px;">
                <h2 style="color: {color}; margin:0;">{probability:.0f}%</h2>
                <p style="margin:0;">Probability of Upside</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.caption("This metric is calculated using a weighted analysis of RSI (Momentum), SMA-50 (Trend), and Golden Cross patterns. It is a technical heuristic, not financial advice.")
            st.button("Buy " + st.session_state.get('ticker', 'Stock'), type="primary")

    def run(self):
        ticker, period = self.render_sidebar()
        st.session_state['ticker'] = ticker # Save for button label
        
        st.title(f"AlphaStream : {ticker}")
        
        with st.spinner("Analyzing Market Data..."):
            stock_data = self.data_manager.fetch_stock_history(ticker, period)
            benchmark_data = self.data_manager.fetch_stock_history("^GSPC", period)
            
        if stock_data is not None:
            # Add indicators
            stock_data = self.analyzer.calculate_technical_indicators(stock_data)
            
            # Calculate Probability
            prob, reasons = self.analyzer.calculate_success_probability(stock_data)
            
            # Render UI Components
            self.render_probability_widget(prob)
            self.render_metrics(stock_data)
            self.render_charts(ticker, stock_data, benchmark_data)
            self.render_ai_modal(ticker, stock_data, prob, reasons)
        else:
            st.error("Could not load data. Please check the ticker.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    app = DashboardUI()
    app.run()