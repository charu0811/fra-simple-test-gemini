import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm

# --- 1. Page Configuration & CSS ---
st.set_page_config(
    page_title="Pro Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the date slider look better
st.markdown("""
<style>
    .stSlider [data-baseweb="slider"] { padding-top: 10px; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4e79a7;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Helper Functions ---
@st.cache_data
def load_and_merge_data(uploaded_files):
    combined_df = None
    logs = []
    
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine='openpyxl')
            
            # Smart Timestamp Detection
            time_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()), None)
            
            if time_col:
                df = df.rename(columns={time_col: 'Timestamp'})
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = df.set_index('Timestamp').sort_index()
                
                # Merge logic
                if combined_df is None:
                    combined_df = df
                else:
                    new_cols = [c for c in df.columns if c not in combined_df.columns]
                    if new_cols:
                        combined_df = pd.merge(combined_df, df[new_cols], left_index=True, right_index=True, how='outer')
                logs.append(f"✅ Loaded: {file.name}")
            else:
                logs.append(f"⚠️ Skipped {file.name}")
                
        except Exception as e:
            logs.append(f"❌ Error {file.name}: {str(e)}")
            
    return combined_df, logs

def calculate_drawdown(series):
    """Calculates the Drawdown series"""
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown

def calculate_regime(series, window):
    """
    Regime Logic:
    - High Vol: > 75th percentile of rolling std
    - Low Vol: < 25th percentile of rolling std
    """
    rolling_vol = series.rolling(window=window).std()
    high_thresh = rolling_vol.quantile(0.75)
    low_thresh = rolling_vol.quantile(0.25)
    
    # Create categorical series
    regime = pd.Series('Normal', index=series.index)
    regime[rolling_vol >= high_thresh] = 'High Volatility'
    regime[rolling_vol <= low_thresh] = 'Low Volatility'
    
    return rolling_vol, high_thresh, low_thresh, regime

def calculate_var(returns, confidence_level=0.95):
    """Calculates Value at Risk (VaR)"""
    if returns.empty: return 0.0
    return np.percentile(returns, (1 - confidence_level) * 100)

# --- 3. Sidebar Control Panel ---
st.sidebar.title("🎛️ Control Panel")
st.sidebar.subheader("Data Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload Files (.xlsx, .csv)", accept_multiple_files=True)

if not uploaded_files:
    st.info("👋 Upload data to start.")
    st.stop()

df_master, logs = load_and_merge_data(uploaded_files)
if df_master is None:
    st.error("No valid data.")
    st.stop()

with st.sidebar.expander("Logs"):
    for l in logs: st.write(l)

num_cols = sorted([c for c in df_master.columns if pd.api.types.is_numeric_dtype(df_master[c])])

# --- 4. Main Layout ---
st.title("📈 Pro Financial Dashboard")

# GLOBAL DATE SLIDER (Moved Top as requested)
min_d, max_d = df_master.index.min().date(), df_master.index.max().date()
col_d1, col_d2 = st.columns([1, 3])
with col_d1:
    st.markdown("### 🗓️ Time Filter")
with col_d2:
    start_date, end_date = st.slider("", min_d, max_d, (min_d, max_d), label_visibility="collapsed")

# Filter Data
mask = (df_master.index.date >= start_date) & (df_master.index.date <= end_date)
df_filt = df_master.loc[mask]

if df_filt.empty:
    st.warning("No data in selected range.")
    st.stop()

# Tabs
tabs = st.tabs(["🔗 Correlation", "📊 Regimes", "⚔️ Price & Technicals", "🧮 Statistics"])

# --- TAB 1: CORRELATION ---
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Rolling Correlation")
        rc_a = st.selectbox("Asset A", num_cols, index=0)
        rc_b = st.selectbox("Asset B", num_cols, index=1 if len(num_cols)>1 else 0)
        win = st.slider("Rolling Window", 10, 200, 30)
        
        roll_corr = df_filt[rc_a].rolling(win).corr(df_filt[rc_b])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode='lines', line=dict(color='#636EFA', width=2), fill='tozeroy', fillcolor='rgba(99, 110, 250, 0.1)'))
        fig.add_hline(y=0, line_dash='dash', line_color='gray')
        fig.update_layout(title=f"{win}-Day Rolling Correlation", height=400, template="plotly_white", yaxis_range=[-1.1, 1.1])
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Correlation Matrix")
        hm_assets = st.multiselect("Select Assets", num_cols, default=num_cols[:8] if len(num_cols)>8 else num_cols)
        
        if len(hm_assets) > 1:
            corr_mat = df_filt[hm_assets].corr()
            fig_hm = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
            fig_hm.update_layout(height=400, title=f"Matrix ({start_date} to {end_date})")
            st.plotly_chart(fig_hm, use_container_width=True)

# --- TAB 2: REGIMES ---
with tabs[1]:
    st.subheader("Volatility Regime Analysis")
    
    col_r1, col_r2 = st.columns([3, 1])
    with col_r2:
        reg_asset = st.selectbox("Target Asset", num_cols)
        reg_win = st.slider("Lookback Window", 10, 100, 20)
        
        st.info("Logic: Background turns **RED** when volatility is in the top 25% of history, and **GREEN** when in the bottom 25%.")

    with col_r1:
        # Calc logic on MASTER to keep thresholds stable
        vol, h_thresh, l_thresh, regime_series = calculate_regime(df_master[reg_asset], reg_win)
        
        # Slice to view
        view_price = df_filt[reg_asset]
        view_regime = regime_series.loc[df_filt.index]
        
        # Create Plot with Colored Backgrounds
        fig = go.Figure()
        
        # We draw the price line
        fig.add_trace(go.Scatter(x=view_price.index, y=view_price, mode='lines', line=dict(color='grey', width=1.5), name='Price'))
        
        # Add colored shapes for regimes
        # This is computationally intensive for many points, so we optimize by grouping
        # Simplified: Scatter plot with filled area logic or simple colored markers for performance in web app
        
        # High Vol Markers (simulating background highlight)
        high_idx = view_regime[view_regime == 'High Volatility'].index
        if not high_idx.empty:
             fig.add_trace(go.Scatter(
                x=high_idx, y=view_price.loc[high_idx],
                mode='markers', marker=dict(color='rgba(231, 76, 60, 0.5)', size=8),
                name='High Volatility'
             ))
             
        # Low Vol Markers
        low_idx = view_regime[view_regime == 'Low Volatility'].index
        if not low_idx.empty:
             fig.add_trace(go.Scatter(
                x=low_idx, y=view_price.loc[low_idx],
                mode='markers', marker=dict(color='rgba(46, 204, 113, 0.5)', size=8),
                name='Low Volatility'
             ))

        fig.update_layout(title=f"Regime Analysis: {reg_asset}", height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: PRICE & TECHNICALS ---
with tabs[2]:
    st.subheader("Price Action & Drawdowns")
    
    pa_asset = st.selectbox("Analyze Asset", num_cols, key='pa')
    
    # 1. Summary Metrics
    ret = (df_filt[pa_asset].iloc[-1] / df_filt[pa_asset].iloc[0]) - 1
    volatility = df_filt[pa_asset].pct_change().std() * np.sqrt(252) # Annualized
    drawdown = calculate_drawdown(df_filt[pa_asset])
    max_dd = drawdown.min()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Period Return", f"{ret:.2%}", delta_color="normal")
    m2.metric("Annualized Volatility", f"{volatility:.2%}", delta_color="off")
    m3.metric("Max Drawdown", f"{max_dd:.2%}", delta_color="inverse")
    
    # 2. Main Chart with Technicals
    show_sma = st.checkbox("Show 50-SMA", value=True)
    show_dd = st.checkbox("Show Drawdown Chart", value=True)
    
    fig_main = make_subplots(rows=2 if show_dd else 1, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if show_dd else [1])
    
    # Price
    fig_main.add_trace(go.Scatter(x=df_filt.index, y=df_filt[pa_asset], name='Price', line=dict(color='grey')), row=1, col=1)
    
    if show_sma:
        sma = df_filt[pa_asset].rolling(50).mean()
        fig_main.add_trace(go.Scatter(x=sma.index, y=sma, name='50 SMA', line=dict(color='orange')), row=1, col=1)
        
    # Drawdown
    if show_dd:
        fig_main.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='Drawdown', fill='tozeroy', line=dict(color='red')), row=2, col=1)
        fig_main.update_yaxes(title="Drawdown %", row=2, col=1)
        
    fig_main.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig_main, use_container_width=True)

# --- TAB 4: STATISTICS ---
with tabs[3]:
    st.subheader("Risk & Distribution Profile")
    
    st_asset = st.selectbox("Select Asset", num_cols, key='st')
    returns = df_filt[st_asset].pct_change().dropna()
    
    # VaR Calc
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    
    col_s1, col_s2 = st.columns([2, 1])
    
    with col_s1:
        # Histogram
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns, histnorm='probability density', name='Returns', marker_color='#3498db', opacity=0.7))
        
        # Normal Curve overlay
        mu, std = norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        p = norm.pdf(x, mu, std)
        fig_dist.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Dist', line=dict(color='red', dash='dash')))
        
        # VaR Line
        fig_dist.add_vline(x=var_95, line_dash="dot", line_color="orange", annotation_text="VaR 95%")
        
        fig_dist.update_layout(title="Return Distribution & Fat Tails", height=450, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_s2:
        st.markdown("### Risk Metrics")
        st.write(pd.DataFrame({
            "Metric": ["Daily VaR (95%)", "Daily VaR (99%)", "Skewness", "Kurtosis"],
            "Value": [f"{var_95:.2%}", f"{var_99:.2%}", f"{returns.skew():.2f}", f"{returns.kurtosis():.2f}"]
        }).set_index("Metric"))
        
        st.caption("""
        * **VaR (95%):** You can expect to lose no more than this amount in a single day, 95% of the time.
        * **Kurtosis > 3:** Implies "Fat Tails" (higher risk of crash).
        * **Skewness < 0:** Implies frequent small gains and few large losses.
        """)
