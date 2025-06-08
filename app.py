import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Custom CSS for Futuristic Theme ---
def apply_custom_css():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap" rel="stylesheet">
        """, unsafe_allow_html=True) # Import Google Fonts

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Configuration ---
st.set_page_config(
    page_title="GHANA MARKET INSIGHTS // ANALYTICS NEXUS",
    page_icon="📡", # Futuristic icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# --- Load and Prepare Data Function (to be cached) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ghana_market_prices_mock.csv')
    except FileNotFoundError:
        st.error("SYSTEM ALERT: Data core 'ghana_market_prices_mock.csv' not found. Please upload to the designated repository.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Price (GHS)'] = pd.to_numeric(df['Price (GHS)'], errors='coerce')
    df.dropna(subset=['Price (GHS)'], inplace=True)
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

df = load_data()

# --- Title and Description ---
st.title("GHANA MARKET INSIGHTS // ANALYTICS NEXUS")
st.markdown("---") # Separator
st.markdown(
    """
    <div style="text-align: center; font-size: 1.2em; color: #a0faff;">
    Engaging advanced algorithms for real-time market flux analysis across Ghanaian sectors.
    Initiating data stream protocol.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---") # Separator


# --- Sidebar Filters ---
st.sidebar.header(":: DATA STREAM CONTROLS ::")

# Product Selection
all_products = df['Product'].unique()
selected_products = st.sidebar.multiselect(
    "Select Product(s) for Analysis",
    options=all_products,
    default=all_products[0] if len(all_products) > 0 else []
)

# Region Selection
all_regions = df['Region'].unique()
selected_regions = st.sidebar.multiselect(
    "Select Regional Data Feeds",
    options=all_regions,
    default=all_regions[0] if len(all_regions) > 0 else []
)

# Date Range Slider
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.slider(
    "Temporal Scan Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM"
)

# Filter the DataFrame based on selections
filtered_df = df[
    df['Product'].isin(selected_products) &
    df['Region'].isin(selected_regions) &
    (df['Date'] >= date_range[0]) &
    (df['Date'] <= date_range[1])
]

# Checkbox to show raw filtered data
if st.sidebar.checkbox("Activate Raw Data Display"):
    st.subheader(":: DECODED DATA LOGS ::")
    st.dataframe(filtered_df)
    st.write(f"Total entries decoded: :blue-badge[{len(filtered_df)}]")

# --- Main Dashboard (Body of App) ---

# Check if filtered_df is empty
if filtered_df.empty:
    st.warning("SYSTEM ALERT: No data signature detected for selected parameters. Adjust filters and re-initiate scan.")
else:
    st.markdown("---") # Separator

    # --- Section 1: Key Metrics Snapshot ---
    st.header(":: KEY METRIC SYNTHESIS ::")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Latest Data Timestamp", df['Date'].max().strftime('%Y-%m'))
    with col2:
        st.metric("Regional Nodes Active", df['Region'].nunique())
    with col3:
        st.metric("Product Categories Monitored", df['Product'].nunique())

    st.subheader("National Average Price per Product (Global Data Stream)")
    avg_national_price_product = df.groupby('Product')['Price (GHS)'].mean().reset_index()
    avg_national_price_product.columns = ['Product', 'Average Price (GHS)']
    st.dataframe(avg_national_price_product.set_index('Product').style.format({'Average Price (GHS)': "GHS {:.2f}"}))


    st.markdown("---") # Separator

    # --- Section 2: Line Charts ---
    st.header(":: TEMPORAL PRICE FLUCTUATION ::")

    # Monthly price trends per region for selected products
    st.subheader("Monthly Price Trajectories per Region (Filtered)")
    if not filtered_df.empty:
        fig_regional_trend, ax_regional_trend = plt.subplots(figsize=(12, 6))
        # Set plot style for futuristic look
        plt.style.use('dark_background') # Start with dark background
        ax_regional_trend.set_facecolor('#0d1117') # Match app background

        # Use a futuristic color palette for lines
        colors = plt.cm.get_cmap('viridis', len(selected_products) * len(selected_regions))
        color_idx = 0

        for product in selected_products:
            for region in selected_regions:
                plot_data = filtered_df[(filtered_df['Product'] == product) & (filtered_df['Region'] == region)]
                if not plot_data.empty:
                    ax_regional_trend.plot(plot_data['Date'], plot_data['Price (GHS)'],
                                           label=f'{product} - {region}',
                                           marker='o', markersize=4,
                                           color=colors(color_idx))
                    color_idx += 1

        ax_regional_trend.set_title("Monthly Price Trajectories by Product and Region", color='#a0faff')
        ax_regional_trend.set_xlabel("Temporal Axis", color='#a0faff')
        ax_regional_trend.set_ylabel("Value (GHS)", color='#a0faff')
        ax_regional_trend.tick_params(axis='x', colors='#00ffc4') # X-axis ticks
        ax_regional_trend.tick_params(axis='y', colors='#00ffc4') # Y-axis ticks
        ax_regional_trend.legend(title="Product - Region", bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#a0faff', facecolor='#1a222e', edgecolor='#005642')
        ax_regional_trend.grid(True, linestyle=':', alpha=0.6, color='#005642') # Dotted grid lines
        fig_regional_trend.autofmt_xdate()
        st.pyplot(fig_regional_trend)
        plt.close(fig_regional_trend)

    # Line plot for national average per product (using filtered data for relevance)
    st.subheader("National Average Price Trajectory per Product (Filtered)")
    if not filtered_df.empty:
        national_avg_filtered = filtered_df.groupby(['Date', 'Product'])['Price (GHS)'].mean().unstack()
        fig_national_avg, ax_national_avg = plt.subplots(figsize=(12, 6))
        plt.style.use('dark_background')
        ax_national_avg.set_facecolor('#0d1117')

        national_avg_filtered.plot(ax=ax_national_avg, marker='o', markersize=4, cmap='plasma') # Use 'plasma' for another futuristic feel
        ax_national_avg.set_title("National Average Price Trajectory per Product (Filtered)", color='#a0faff')
        ax_national_avg.set_xlabel("Temporal Axis", color='#a0faff')
        ax_national_avg.set_ylabel("Average Value (GHS)", color='#a0faff')
        ax_national_avg.tick_params(axis='x', colors='#00ffc4')
        ax_national_avg.tick_params(axis='y', colors='#00ffc4')
        ax_national_avg.legend(title="Product", bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#a0faff', facecolor='#1a222e', edgecolor='#005642')
        ax_national_avg.grid(True, linestyle=':', alpha=0.6, color='#005642')
        fig_national_avg.autofmt_xdate()
        st.pyplot(fig_national_avg)
        plt.close(fig_national_avg)

    st.markdown("---") # Separator

    # --- Section 3: Summary Tables ---
    st.header(":: DATA MATRIX SUMMARIES ::")

    if not filtered_df.empty:
        # Grouped by Product and Region: Mean, Min, Max, Std Dev
        st.subheader("Statistical Breakdown: Product & Region")
        summary_stats = filtered_df.groupby(['Product', 'Region'])['Price (GHS)'].agg(['mean', 'min', 'max', 'std']).reset_index()
        summary_stats.columns = ['Product', 'Region', 'Mean Price (GHS)', 'Min Price (GHS)', 'Max Price (GHS)', 'Std Dev (GHS)']
        st.dataframe(summary_stats.style.format({
            'Mean Price (GHS)': "GHS {:.2f}",
            'Min Price (GHS)': "GHS {:.2f}",
            'Max Price (GHS)': "GHS {:.2f}",
            'Std Dev (GHS)': "GHS {:.2f}"
        }))

        # % price change (last month vs previous) - Requires at least 2 months of data
        st.subheader("Temporal Value Oscillation: Last Cycle vs. Current Cycle")
        price_change_df = filtered_df.copy()
        price_change_df['YearMonth'] = price_change_df['Date'].dt.to_period('M')
        price_change_df = price_change_df.sort_values(by=['Product', 'Region', 'YearMonth'])

        price_change_df['Previous_Month_Price'] = price_change_df.groupby(['Product', 'Region'])['Price (GHS)'].shift(1)
        price_change_df['Price_Change_Pct'] = ((price_change_df['Price (GHS)'] - price_change_df['Previous_Month_Price']) / price_change_df['Previous_Month_Price']) * 100

        latest_month_prices = price_change_df.groupby(['Product', 'Region']).apply(lambda x: x.iloc[-1] if not x.empty else pd.Series()).reset_index(drop=True)
        latest_month_prices = latest_month_prices[['Product', 'Region', 'YearMonth', 'Price (GHS)', 'Price_Change_Pct']]
        latest_month_prices.rename(columns={'Price (GHS)': 'Latest Value (GHS)'}, inplace=True)
        latest_month_prices = latest_month_prices[latest_month_prices['Price_Change_Pct'].notna()]

        if not latest_month_prices.empty:
            st.dataframe(latest_month_prices.style.format({
                'Latest Value (GHS)': "GHS {:.2f}",
                'Price_Change_Pct': "{:.2f}%"
            }).set_caption("Note: Oscillation detected for latest cycle relative to previous cycle."))
        else:
            st.info("No sufficient temporal data for oscillation analysis (requires minimum two cycles).")

        # Top 3 most expensive regions per product
        st.subheader("High-Value Regional Hotspots (Top 3 per Product)")
        if not filtered_df.empty:
            avg_price_per_product_region = filtered_df.groupby(['Product', 'Region'])['Price (GHS)'].mean().reset_index()
            top_regions = avg_price_per_product_region.groupby('Product').apply(lambda x: x.nlargest(3, 'Price (GHS)')).reset_index(drop=True)
            top_regions.columns = ['Product', 'Region', 'Average Value (GHS)']
            st.dataframe(top_regions.style.format({'Average Value (GHS)': "GHS {:.2f}"}))
        else:
            st.info("Insufficient data to determine high-value hotspots.")


    st.markdown("---") # Separator

    # --- Section 4: Visual EDA ---
    st.header(":: VISUAL DATA INTERPRETATION ::")

    # Heatmap of average prices across products & regions
    st.subheader("Value Density Across Product & Region Matrix")
    if not filtered_df.empty:
        pivot_table = filtered_df.pivot_table(index='Product', columns='Region', values='Price (GHS)', aggfunc='mean')
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
        plt.style.use('dark_background')
        ax_heatmap.set_facecolor('#0d1117')
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, linecolor='#005642', cbar_kws={'label': 'Average Value (GHS)'}, ax=ax_heatmap)
        ax_heatmap.set_title("Average Values (GHS) Across Product and Region Spectrum", color='#a0faff')
        ax_heatmap.set_xlabel("Regional Node", color='#a0faff')
        ax_heatmap.set_ylabel("Product Category", color='#a0faff')
        ax_heatmap.tick_params(axis='x', colors='#00ffc4')
        ax_heatmap.tick_params(axis='y', colors='#00ffc4')
        st.pyplot(fig_heatmap)
        plt.close(fig_heatmap)

    # Bar chart comparing product prices across selected regions
    st.subheader("Product Value Discrepancy Across Regional Nodes")
    if not filtered_df.empty and len(selected_regions) > 1:
        avg_prices_product_region = filtered_df.groupby(['Product', 'Region'])['Price (GHS)'].mean().unstack()
        fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
        plt.style.use('dark_background')
        ax_bar.set_facecolor('#0d1117')
        avg_prices_product_region.plot(kind='bar', ax=ax_bar, width=0.8, cmap='viridis') # Use 'viridis' cmap
        ax_bar.set_title("Average Product Values Across Selected Regions", color='#a0faff')
        ax_bar.set_xlabel("Product Category", color='#a0faff')
        ax_bar.set_ylabel("Average Value (GHS)", color='#a0faff')
        ax_bar.tick_params(axis='x', rotation=45, colors='#00ffc4')
        ax_bar.tick_params(axis='y', colors='#00ffc4')
        ax_bar.legend(title="Regional Node", bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#a0faff', facecolor='#1a222e', edgecolor='#005642')
        ax_bar.grid(axis='y', linestyle=':', alpha=0.6, color='#005642')
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    elif len(selected_regions) <= 1:
        st.info("Initiate comparison protocol: Select at least two regional nodes for discrepancy analysis.")

    # Optional: Box plots for price distribution per product
    st.subheader("Value Distribution Profile per Product")
    if not filtered_df.empty:
        fig_box, ax_box = plt.subplots(figsize=(12, 6))
        plt.style.use('dark_background')
        ax_box.set_facecolor('#0d1117')
        sns.boxplot(x='Product', y='Price (GHS)', data=filtered_df, ax=ax_box, palette='viridis') # Use 'viridis' palette
        ax_box.set_title("Value Distribution per Product Category", color='#a0faff')
        ax_box.set_xlabel("Product Category", color='#a0faff')
        ax_box.set_ylabel("Value (GHS)", color='#a0faff')
        ax_box.tick_params(axis='x', rotation=45, colors='#00ffc4')
        ax_box.tick_params(axis='y', colors='#00ffc4')
        ax_box.grid(axis='y', linestyle=':', alpha=0.6, color='#005642')
        st.pyplot(fig_box)
        plt.close(fig_box)


    st.markdown("---") # Separator

    # --- Section 5: Download Options ---
    st.header(":: DATA EXTRACTION PROTOCOLS ::")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)

    st.download_button(
        label="Download Filtered Data Log (.CSV)",
        data=csv,
        file_name="ghana_market_prices_filtered_data_log.csv",
        mime="text/csv",
    )

    st.info("NOTE: For graphic data extraction, right-click visual displays and select 'Save Image As...'.")