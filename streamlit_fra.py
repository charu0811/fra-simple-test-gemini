import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Financial Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Apply a custom style to plots to make them look professional
plt.style.use('seaborn-v0_8-whitegrid')

# --- 2. Data Loading Logic ---
@st.cache_data
def load_and_merge_data(uploaded_files):
    """
    Reads uploaded files (CSV/Excel), detects Timestamps, 
    and merges them into a single Master DataFrame.
    """
    combined_df = None
    log_messages = []
    
    for uploaded_file in uploaded_files:
        try:
            # Determine file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Smart Timestamp Detection
            # Look for columns containing 'date', 'time', or 'timestamp' (case insensitive)
            time_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
            
            if time_col:
                df = df.rename(columns={time_col: 'Timestamp'})
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = df.set_index('Timestamp').sort_index()
                
                # Merge logic
                if combined_df is None:
                    combined_df = df
                else:
                    # Only merge new columns to avoid duplicates
                    new_cols = [c for c in df.columns if c not in combined_df.columns]
                    if new_cols:
                        combined_df = pd.merge(combined_df, df[new_cols], left_index=True, right_index=True, how='outer')
                
                log_messages.append(f"✅ Loaded: {uploaded_file.name}")
            else:
                log_messages.append(f"⚠️ Skipped {uploaded_file.name}: No timestamp column found.")
                
        except Exception as e:
            log_messages.append(f"❌ Error loading {uploaded_file.name}: {str(e)}")
            
    return combined_df, log_messages

# --- 3. Sidebar: Controls ---
st.sidebar.header("1. Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload your Excel (.xlsx) or CSV files", 
    type=['xlsx', 'csv'], 
    accept_multiple_files=True
)

df_master = None

if uploaded_files:
    df_master, logs = load_and_merge_data(uploaded_files)
    
    # Show status briefly
    with st.expander("Upload Status"):
        for msg in logs:
            st.write(msg)

if df_master is not None:
    # Get Numeric Columns only
    numeric_cols = sorted([c for c in df_master.columns if pd.api.types.is_numeric_dtype(df_master[c])])
    
    # --- Global Date Slider ---
    st.sidebar.header("2. Time Filter")
    
    # Get min and max dates from data
    min_date = df_master.index.min().date()
    max_date = df_master.index.max().date()
    
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    
    # Filter DataFrame based on selection
    mask = (df_master.index.date >= start_date) & (df_master.index.date <= end_date)
    df_filtered = df_master.loc[mask]
    
    # --- Main Dashboard Area ---
    st.title("📈 Financial Market Analysis")
    st.markdown(f"**Data Range:** {start_date} to {end_date} | **Rows:** {len(df_filtered)}")
    
    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
    else:
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["🔥 Correlation Matrix", "⚔️ Price Comparison", "📊 Regime Analysis"])
        
        # --- TAB 1: CORRELATION ---
        with tab1:
            st.subheader("Correlation Heatmap")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_corr_cols = st.multiselect(
                    "Select Assets (Min 2):", 
                    numeric_cols, 
                    default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
                )
            
            with col2:
                if len(selected_corr_cols) < 2:
                    st.info("Please select at least two assets from the left.")
                else:
                    corr_matrix = df_filtered[selected_corr_cols].corr()
                    
                    # Seaborn Plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        cmap='coolwarm', 
                        center=0, 
                        fmt=".2f", 
                        ax=ax,
                        linewidths=0.5
                    )
                    ax.set_title(f"Correlation Matrix ({start_date} - {end_date})")
                    st.pyplot(fig)

        # --- TAB 2: COMPARISON ---
        with tab2:
            st.subheader("Side-by-Side Price Movement")
            
            c1, c2 = st.columns(2)
            with c1:
                asset_left = st.selectbox("Left Axis Asset:", numeric_cols, index=0)
            with c2:
                asset_right = st.selectbox("Right Axis Asset:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
            # Dual Axis Plot
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            color_left = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel(asset_left, color=color_left, fontsize=12)
            ax1.plot(df_filtered.index, df_filtered[asset_left], color=color_left, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color_left)
            ax1.grid(True, alpha=0.3)
            
            # Second Axis
            ax2 = ax1.twinx()
            color_right = 'tab:orange'
            ax2.set_ylabel(asset_right, color=color_right, fontsize=12)
            ax2.plot(df_filtered.index, df_filtered[asset_right], color=color_right, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color_right)
            
            # Format Date Axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            st.pyplot(fig)

        # --- TAB 3: REGIME ANALYSIS ---
        with tab3:
            st.subheader("Volatility Regime Classification")
            
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                regime_asset = st.selectbox("Select Asset to Analyze:", numeric_cols, index=0)
            with rc2:
                vol_window = st.slider("Volatility Lookback Window (Days):", 5, 100, 20)
                
            # Logic
            # 1. Calculate Volatility on the FULL dataset (to get accurate quantiles)
            full_series = df_master[regime_asset]
            rolling_vol = full_series.rolling(window=vol_window).std()
            
            high_thresh = rolling_vol.quantile(0.75)
            low_thresh = rolling_vol.quantile(0.25)
            
            # 2. Filter for display
            display_data = df_filtered[[regime_asset]].copy()
            # Map the volatility data to the filtered index
            display_vol = rolling_vol.loc[display_data.index]
            
            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Main Price Line
            ax.plot(display_data.index, display_data[regime_asset], color='black', alpha=0.5, label='Price', linewidth=1)
            
            # High Volatility Dots (Red)
            high_vol_mask = display_vol >= high_thresh
            if high_vol_mask.any():
                ax.scatter(
                    display_data.index[high_vol_mask], 
                    display_data.loc[high_vol_mask, regime_asset], 
                    color='red', s=20, label='High Volatility', zorder=5
                )
                
            # Low Volatility Dots (Green)
            low_vol_mask = display_vol <= low_thresh
            if low_vol_mask.any():
                ax.scatter(
                    display_data.index[low_vol_mask], 
                    display_data.loc[low_vol_mask, regime_asset], 
                    color='green', s=20, label='Low Volatility', zorder=5
                )
            
            ax.set_title(f"Regime Analysis: {regime_asset} (Window: {vol_window})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format Date Axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            st.pyplot(fig)
            
            st.info(f"ℹ️ **High Volatility:** > {high_thresh:.4f} | **Low Volatility:** < {low_thresh:.4f} (Calculated based on full history)")

else:
    # Empty State
    st.info("👈 Please upload your Excel (.xlsx) or CSV files in the sidebar to begin.")
    st.markdown("""
    ### How to use:
    1. **Upload Files:** Use the sidebar to upload your financial data.
    2. **Filter Time:** Use the slider in the sidebar to narrow down the analysis period.
    3. **Explore:** Switch between the tabs to see Correlations, Price Comparisons, and Regime Analysis.
    """)