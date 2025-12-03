import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Helper Functions ---
@st.cache_data
def load_and_merge_data(uploaded_files):
    """
    Robustly loads and merges multiple CSV/Excel files into a single DataFrame.
    """
    combined_df = None
    logs = []
    
    for file in uploaded_files:
        try:
            # Load Data
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine='openpyxl')
            
            # Find Timestamp Column
            time_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()), None)
            
            if time_col:
                df = df.rename(columns={time_col: 'Timestamp'})
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = df.set_index('Timestamp').sort_index()
                
                # Merge
                if combined_df is None:
                    combined_df = df
                else:
                    # Avoid duplicate columns
                    new_cols = [c for c in df.columns if c not in combined_df.columns]
                    if new_cols:
                        combined_df = pd.merge(combined_df, df[new_cols], left_index=True, right_index=True, how='outer')
                logs.append(f"✅ Loaded: {file.name} ({len(df)} rows)")
            else:
                logs.append(f"⚠️ Skipped {file.name}: No date/timestamp column found.")
                
        except Exception as e:
            logs.append(f"❌ Error {file.name}: {str(e)}")
            
    return combined_df, logs

def calculate_regime(series, window):
    """
    Calculates Volatility Regimes based on Rolling Standard Deviation.
    """
    rolling_vol = series.rolling(window=window).std()
    high_thresh = rolling_vol.quantile(0.75)
    low_thresh = rolling_vol.quantile(0.25)
    
    return rolling_vol, high_thresh, low_thresh

# --- 3. Main Application ---

st.sidebar.title("🎛️ Control Panel")

# A. File Upload
st.sidebar.subheader("1. Data Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload Market Data (.xlsx, .csv)", accept_multiple_files=True)

if not uploaded_files:
    st.info("👋 Welcome! Please upload your financial data files in the sidebar to begin.")
    st.stop()

# Load Data
df_master, log_msgs = load_and_merge_data(uploaded_files)

with st.sidebar.expander("Upload Logs", expanded=False):
    for msg in log_msgs:
        st.write(msg)

if df_master is None:
    st.error("No valid data loaded.")
    st.stop()

num_cols = sorted([c for c in df_master.columns if pd.api.types.is_numeric_dtype(df_master[c])])

# B. Global Date Filter
st.sidebar.subheader("2. Time Window")
min_d, max_d = df_master.index.min().date(), df_master.index.max().date()
start_date, end_date = st.sidebar.slider("Select Analysis Period", min_d, max_d, (min_d, max_d))

# Filter Master DF
mask = (df_master.index.date >= start_date) & (df_master.index.date <= end_date)
df_filtered = df_master.loc[mask]

# --- 4. Dashboard Tabs ---
st.title("📈 Market Analysis Dashboard")
st.markdown(f"**Period:** {start_date} to {end_date} • **Assets:** {len(num_cols)} • **Data Points:** {len(df_filtered)}")

tabs = st.tabs([
    "🔗 Correlation Dynamics", 
    "📊 Regime & Volatility", 
    "⚔️ Price Action", 
    "🧮 Statistical Summary"
])

# --- TAB 1: CORRELATION DYNAMICS ---
with tabs[0]:
    st.header("Correlation Analysis")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rolling Correlation")
        rc_col1 = st.selectbox("Asset A", num_cols, index=0, key='rc1')
        rc_col2 = st.selectbox("Asset B", num_cols, index=1 if len(num_cols)>1 else 0, key='rc2')
        roll_win = st.slider("Rolling Window (Days)", 10, 180, 30)
        
        # Calculate
        rolling_corr = df_filtered[rc_col1].rolling(window=roll_win).corr(df_filtered[rc_col2])
        
        # Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', name='Correlation', line=dict(color='purple')))
        
        # Add Horizontal Lines
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3)
        fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.3)
        
        fig.update_layout(
            title=f"{roll_win}-Day Rolling Correlation: {rc_col1} vs {rc_col2}",
            yaxis=dict(range=[-1.1, 1.1], title="Correlation Coefficient"),
            xaxis_title="Date",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Correlation Matrix")
        hm_cols = st.multiselect("Select Assets for Matrix", num_cols, default=num_cols[:6] if len(num_cols) > 6 else num_cols)
        
        if len(hm_cols) > 1:
            corr_mat = df_filtered[hm_cols].corr()
            
            fig = px.imshow(
                corr_mat, 
                text_auto=".2f", 
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="Correlation Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least 2 assets.")

# --- TAB 2: REGIME & VOLATILITY ---
with tabs[1]:
    st.header("Regime & Volatility Analysis")
    
    col_reg, col_param = st.columns([3, 1])
    with col_param:
        st.markdown("### Settings")
        reg_asset = st.selectbox("Select Asset", num_cols, key='reg_asset')
        reg_win = st.slider("Volatility Window", 10, 100, 20)
        show_thresholds = st.checkbox("Show Threshold Lines", value=True)

    with col_reg:
        # Calculate
        vol_series, high_line, low_line = calculate_regime(df_master[reg_asset], reg_win)
        
        # Filter for display
        disp_price = df_filtered[reg_asset]
        disp_vol = vol_series.loc[df_filtered.index]
        
        # Create Subplots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Price Action: {reg_asset}", "Volatility Indicator")
        )
        
        # 1. Price Trace
        fig.add_trace(go.Scatter(x=disp_price.index, y=disp_price, mode='lines', name='Price', line=dict(color='gray', width=1)), row=1, col=1)
        
        # 2. High Vol Dots
        high_mask = disp_vol >= high_line
        if high_mask.any():
            fig.add_trace(go.Scatter(
                x=disp_price.index[high_mask], y=disp_price[high_mask], 
                mode='markers', name='High Volatility', 
                marker=dict(color='#e74c3c', size=6)
            ), row=1, col=1)
            
        # 3. Low Vol Dots
        low_mask = disp_vol <= low_line
        if low_mask.any():
            fig.add_trace(go.Scatter(
                x=disp_price.index[low_mask], y=disp_price[low_mask], 
                mode='markers', name='Low Volatility', 
                marker=dict(color='#2ecc71', size=6)
            ), row=1, col=1)
            
        # 4. Volatility Trace
        fig.add_trace(go.Scatter(x=disp_vol.index, y=disp_vol, mode='lines', name='Volatility', line=dict(color='black', width=1)), row=2, col=1)
        
        if show_thresholds:
            # High Line
            fig.add_hline(y=high_line, line_dash="dash", line_color="#e74c3c", row=2, col=1, annotation_text="High Vol (75%)")
            # Low Line
            fig.add_hline(y=low_line, line_dash="dash", line_color="#2ecc71", row=2, col=1, annotation_text="Low Vol (25%)")
        
        fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: PRICE ACTION ---
with tabs[2]:
    st.header("Price Comparison")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        st.markdown("### Configuration")
        compare_assets = st.multiselect("Select Assets", num_cols, default=num_cols[:2] if len(num_cols)>1 else num_cols[:1])
        normalize = st.checkbox("Rebase to 100", value=True)
        
    with col_p2:
        if compare_assets:
            comp_data = df_filtered[compare_assets].dropna()
            
            if normalize:
                comp_data = (comp_data / comp_data.iloc[0]) * 100
                title = "Relative Performance (Base 100)"
                y_label = "Normalized Return"
            else:
                title = "Absolute Price Comparison"
                y_label = "Price"
            
            fig = px.line(comp_data, x=comp_data.index, y=comp_data.columns, title=title)
            fig.update_layout(yaxis_title=y_label, height=500, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select assets to compare.")

# --- TAB 4: STATISTICAL SUMMARY ---
with tabs[3]:
    st.header("Statistical Profile")
    
    if not df_filtered.empty:
        # Z-Scores
        latest_prices = df_filtered.iloc[-1]
        means = df_filtered.mean()
        stds = df_filtered.std()
        z_scores = (latest_prices - means) / stds
        
        z_df = pd.DataFrame({'Z-Score': z_scores}).sort_values('Z-Score')
        
        # Color logic
        colors = ['red' if abs(z) > 2 else 'blue' for z in z_df['Z-Score']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=z_df.index, y=z_df['Z-Score'],
            marker_color=colors,
            name='Z-Score'
        ))
        
        fig.add_hline(y=2, line_dash="dash", line_color="red")
        fig.add_hline(y=-2, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Current Z-Scores (Mean Reversion Signals)",
            yaxis_title="Standard Deviations from Mean",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("Detailed Metrics")
        stats = df_filtered[num_cols].describe().T
        stats['skew'] = df_filtered[num_cols].skew()
        stats['kurtosis'] = df_filtered[num_cols].kurtosis()
        display_stats = stats[['mean', 'std', 'min', 'max', 'skew', 'kurtosis']]
        st.dataframe(display_stats.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)

    else:
        st.warning("No data in selected range.")
