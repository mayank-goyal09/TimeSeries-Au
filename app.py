import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from utils import forecast_next_n_days_direct
import os
import base64
from daily_data_update import update_database
from train_model import train_model

# Page configuration
st.set_page_config(
    page_title="Gold Price Oracle | AI Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Gold & Glass CSS
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Poppins:wght@300;400;500;600&display=swap');
    
    /* Root variables */
    :root {
        --gold-primary: #D4AF37;
        --gold-light: #F5E6A3;
        --gold-dark: #996515;
        --gold-shine: linear-gradient(135deg, #D4AF37 0%, #F5E6A3 25%, #D4AF37 50%, #996515 75%, #D4AF37 100%);
        --glass-bg: rgba(20, 20, 30, 0.7);
        --glass-border: rgba(212, 175, 55, 0.3);
        --glass-shadow: 0 8px 32px rgba(212, 175, 55, 0.15);
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a15 0%, #1a1a2e 25%, #16213e 50%, #1a1a2e 75%, #0a0a15 100%);
        background-attachment: fixed;
    }
    
    /* Animated gold particles overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #D4AF37, transparent),
            radial-gradient(2px 2px at 40px 70px, #F5E6A3, transparent),
            radial-gradient(1px 1px at 90px 40px, #D4AF37, transparent),
            radial-gradient(2px 2px at 130px 80px, #F5E6A3, transparent),
            radial-gradient(1px 1px at 160px 120px, #D4AF37, transparent);
        background-size: 200px 200px;
        animation: sparkle 8s linear infinite;
        pointer-events: none;
        opacity: 0.3;
        z-index: 0;
    }
    
    @keyframes sparkle {
        0% { transform: translateY(0); }
        100% { transform: translateY(-200px); }
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(20, 20, 35, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(212, 175, 55, 0.1),
            0 0 60px rgba(212, 175, 55, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    /* Headers with gold gradient */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        background: linear-gradient(135deg, #D4AF37 0%, #F5E6A3 30%, #D4AF37 60%, #996515 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(212, 175, 55, 0.3);
    }
    
    /* Main title styling */
    .main-title {
        font-family: 'Cinzel', serif !important;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #D4AF37 0%, #F5E6A3 25%, #FFFFFF 50%, #F5E6A3 75%, #D4AF37 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: goldShimmer 3s linear infinite;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
    }
    
    @keyframes goldShimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(212, 175, 55, 0.7);
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 20, 0.95) 0%, rgba(20, 20, 35, 0.95) 100%) !important;
        border-right: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #D4AF37, #F5E6A3, #D4AF37);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #996515 100%) !important;
        color: #000 !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.5) !important;
        background: linear-gradient(135deg, #F5E6A3 0%, #D4AF37 100%) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(20, 20, 35, 0.6);
        border: 2px dashed rgba(212, 175, 55, 0.4);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #D4AF37;
        box-shadow: 0 0 30px rgba(212, 175, 55, 0.2);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #D4AF37, #F5E6A3) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #D4AF37 !important;
        border: 2px solid #F5E6A3 !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-family: 'Cinzel', serif !important;
    }
    
    /* Info/Success/Error boxes */
    .stAlert {
        background: rgba(20, 20, 35, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    
    /* Download button special styling */
    .download-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        background: linear-gradient(135deg, #D4AF37 0%, #996515 100%);
        color: #000;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-decoration: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
        margin: 0.5rem 0;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.5);
        color: #000;
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(30, 30, 45, 0.6);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(212, 175, 55, 0.4);
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        color: #D4AF37;
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: rgba(255, 255, 255, 0.6);
        font-family: 'Poppins', sans-serif;
        font-size: 0.85rem;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(20, 20, 35, 0.8);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #D4AF37, #996515);
        border-radius: 4px;
    }
    
    /* Text colors */
    p, span, label, .stMarkdown {
        color: rgba(255, 255, 255, 0.85) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(20, 20, 35, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.2) !important;
        border-radius: 12px !important;
        color: #D4AF37 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px 12px 0 0;
        color: rgba(255, 255, 255, 0.7);
        font-family: 'Poppins', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(212, 175, 55, 0.1));
        border-color: #D4AF37;
        color: #D4AF37 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🥇 GOLD PRICE ORACLE</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Gold Price Forecasting with LSTM Neural Networks</p>', unsafe_allow_html=True)

# Feature cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">LSTM Model</div>
        <div class="feature-desc">Deep learning neural network</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Multi-Output</div>
        <div class="feature-desc">Up to 30 days forecast</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Real-Time</div>
        <div class="feature-desc">Instant predictions</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Accurate</div>
        <div class="feature-desc">Data-driven insights</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gold_lstm_multioutput.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("price_scaler.pkl")

model = load_model()
scaler = load_scaler()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="font-size: 1.8rem;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Update Data Section
    col_update, col_train = st.columns(2)
    with col_update:
        if st.button("🔄 Refresh Data", help="Fetch latest prices from Yahoo Finance"):
            with st.spinner("Fetching latest market data..."):
                try:
                    updated_df = update_database()
                    if updated_df is not None:
                          st.success(f"Updated! {updated_df.iloc[-1]['Date'].date()}")
                          st.rerun()
                    else:
                           st.warning("No new data.")
                except Exception as e:
                    st.error(f"Update failed: {e}")
    
    with col_train:
        if st.button("🧠 Retrain AI", help="Train model on latest data (Required if price range changes)"):
            with st.spinner("Training AI model... This may take a minute."):
                try:
                    train_model()
                    st.cache_resource.clear()
                    st.success("Model Retrained!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

                
    st.markdown("---")
    
    # Download sample CSV section
    st.markdown("""
    <div style="background: rgba(212, 175, 55, 0.1); border-radius: 12px; padding: 1rem; border: 1px solid rgba(212, 175, 55, 0.3);">
        <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">📥 Sample Data</h3>
        <p style="font-size: 0.8rem; opacity: 0.8;">Download the sample CSV file to test the predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if sample CSV exists and provide download
    sample_csv_path = "Gold Price.csv"
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, 'rb') as f:
            csv_data = f.read()
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_data,
            file_name="Gold_Price_Sample.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style="font-size: 1rem;">🎛️ Forecast Parameters</h3>
    """, unsafe_allow_html=True)
    
    horizon = st.slider(
        "📅 Forecast Horizon (days)",
        min_value=7,
        max_value=30,
        value=30,
        help="Number of days to predict into the future (max 30)"
    )
    
    window_size = st.slider(
        "📊 Window Size (days)",
        min_value=30,
        max_value=180,
        value=60,
        help="Historical data window for prediction"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(20, 20, 35, 0.6); border-radius: 12px; padding: 1rem; border: 1px solid rgba(212, 175, 55, 0.2);">
        <h4 style="font-size: 0.9rem; color: #D4AF37;">📋 CSV Format Required:</h4>
        <p style="font-size: 0.75rem; opacity: 0.8; margin-top: 0.5rem;">
        Your CSV should have these columns:<br>
        • <strong>Date</strong> - Date column<br>
        • <strong>Price</strong> - Gold price values
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Data")

uploaded = st.file_uploader(
    "Upload your gold price CSV file (or leave empty to use live system data)",
    type=["csv"],
    help="Upload a CSV file with 'Date' and 'Price' columns"
)

st.markdown('</div>', unsafe_allow_html=True)

df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
else:
    # Load local system data
    if os.path.exists("Gold Price.csv"):
        try:
            df = pd.read_csv("Gold Price.csv")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            st.info(f"Loaded system data. Latest data point: {df.index[-1].date()}")
        except Exception as e:
            st.error(f"Error reading local database: {e}")
    else:
        st.warning("No local data found. Please upload a CSV.")

if df is not None:
    # Data Overview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
    with col2:
        st.metric("📅 Data Points", f"{len(df):,}")
    with col3:
        price_change = df['Price'].iloc[-1] - df['Price'].iloc[-2]
        st.metric("📊 24h Change", f"${price_change:,.2f}", delta=f"{(price_change/df['Price'].iloc[-2]*100):.2f}%")
    with col4:
        st.metric("📉 Min Price", f"${df['Price'].min():,.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Historical Chart
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Historical Gold Prices")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df.reset_index()["Date"],
        y=df["Price"],
        mode='lines',
        name='Gold Price',
        line=dict(color='#D4AF37', width=2),
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.1)'
    ))
    
    fig_hist.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", color='rgba(255,255,255,0.8)'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(212, 175, 55, 0.1)',
            title='Date'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(212, 175, 55, 0.1)',
            title='Price (USD)'
        ),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast
    if len(df) < window_size:
        st.error(f"⚠️ Need at least {window_size} rows to forecast. Your data has {len(df)} rows.")
    else:
        with st.spinner('🔮 Generating AI predictions...'):
            try:
                future_df = forecast_next_n_days_direct(model, df, scaler, window_size=window_size, horizon=horizon)
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🔮 AI Forecast Results")
                
                # Forecast metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Predicted (Day 1)", f"${future_df['Forecast_Price'].iloc[0]:,.2f}")
                with col2:
                    st.metric(f"📅 Predicted (Day {horizon})", f"${future_df['Forecast_Price'].iloc[-1]:,.2f}")
                with col3:
                    trend = future_df['Forecast_Price'].iloc[-1] - df['Price'].iloc[-1]
                    st.metric("📊 Expected Trend", f"${trend:,.2f}", delta=f"{(trend/df['Price'].iloc[-1]*100):.2f}%")
                
                # Combined chart
                plot_df = pd.concat([
                    df[["Price"]].tail(200).rename(columns={"Price": "Actual"}),
                    future_df.rename(columns={"Forecast_Price": "Forecast"})
                ], axis=0)
                
                fig = go.Figure()
                
                # Historical data
                actual_df = plot_df[plot_df["Actual"].notna()]
                fig.add_trace(go.Scatter(
                    x=actual_df.index,
                    y=actual_df["Actual"],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#D4AF37', width=2)
                ))
                
                # Forecast data
                forecast_df = plot_df[plot_df["Forecast"].notna()]
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df["Forecast"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#00D4AA', width=3, dash='dot')
                ))
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins", color='rgba(255,255,255,0.8)'),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(212, 175, 55, 0.1)',
                        title='Date'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(212, 175, 55, 0.1)',
                        title='Price (USD)'
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    hovermode='x unified',
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Forecast table
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Detailed Forecast Table")
                
                # Format the dataframe for display
                display_df = future_df.reset_index()
                display_df.columns = ['Date', 'Predicted Price (USD)']
                display_df['Predicted Price (USD)'] = display_df['Predicted Price (USD)'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download forecast
                csv = future_df.reset_index().to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="gold_price_forecast.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Forecast generation failed: {e}")

else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">🥇</div>
        <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">Ready to Predict Gold Prices</h2>
        <p style="opacity: 0.7; max-width: 500px; margin: 0 auto;">
            Upload your CSV file with historical gold prices to start forecasting. 
            Download the sample CSV from the sidebar if you need example data.
        </p>
        <br>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(212, 175, 55, 0.1); padding: 1rem 1.5rem; border-radius: 12px; border: 1px solid rgba(212, 175, 55, 0.2);">
                <span style="font-size: 1.5rem;">1️⃣</span><br>
                <span style="font-size: 0.9rem;">Download Sample CSV</span>
            </div>
            <div style="background: rgba(212, 175, 55, 0.1); padding: 1rem 1.5rem; border-radius: 12px; border: 1px solid rgba(212, 175, 55, 0.2);">
                <span style="font-size: 1.5rem;">2️⃣</span><br>
                <span style="font-size: 0.9rem;">Upload Your Data</span>
            </div>
            <div style="background: rgba(212, 175, 55, 0.1); padding: 1rem 1.5rem; border-radius: 12px; border: 1px solid rgba(212, 175, 55, 0.2);">
                <span style="font-size: 1.5rem;">3️⃣</span><br>
                <span style="font-size: 0.9rem;">Get AI Predictions</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem 0; opacity: 0.6;">
    <p style="font-size: 0.85rem;">
        🥇 Gold Price Oracle | Powered by LSTM Neural Networks | Built with Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
