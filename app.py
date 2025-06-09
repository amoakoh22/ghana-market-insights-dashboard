import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Custom CSS for Futuristic Theme ---
def apply_custom_css():
    # Import Google Fonts for futuristic look
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap" rel="stylesheet">
        """, unsafe_allow_html=True)

    # Load custom CSS from style.css file
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("SYSTEM ALERT: 'style.css' not found. Default styling initiated. Ensure 'style.css' is in the same directory.")

# --- Configuration ---
st.set_page_config(
    page_title="GHANA MARKET INSIGHTS // ANALYTICS NEXUS",
    page_icon="📡", # Futuristic icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for the futuristic theme
apply_custom_css()

# --- Load and Prepare Data Function (to be cached for performance) ---
@st.cache_data
def load_data():
    """
    Loads and preprocesses the Ghana market prices data.
    Caches the data to improve app performance on subsequent runs.
    """
    try:
        df = pd.read_csv('ghana_market_prices_mock.csv')
    except FileNotFoundError:
        st.error("SYSTEM ALERT: Data core 'ghana_market_prices_mock.csv' not found. Please upload to the designated repository.")
        st.stop() # Halts the app execution if the data file is missing

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure 'Price (GHS)' is numeric, coercing errors to NaN
    df['Price (GHS)'] = pd.to_numeric(df['Price (GHS)'], errors='coerce')

    # Drop rows where 'Price (GHS)' is missing (NaN) after conversion
    df.dropna(subset=['Price (GHS)'], inplace=True)

    # Sort data by 'Date' for proper time-series analysis
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

# Load the preprocessed data
df = load_data()

# --- Title and Description ---
st.title("GHANA MARKET INSIGHTS // ANALYTICS NEXUS")
st.markdown("---") # Visual separator for futuristic aesthetic
st.markdown(
    """
    <div style="text-align: center; font-size: 1.2em; color: #a0faff;">
    Engaging advanced algorithms for real-time market flux analysis across Ghanaian sectors.
    Initiating data stream protocol.
    </div>
    """,
    unsafe_allow_html=True # Allows rendering of HTML for custom styling
)
st.markdown("---") # Another visual separator


# --- Sidebar Filters (Data Stream Controls) ---
st.sidebar.header(":: DATA STREAM CONTROLS ::")

# Product Selection Multiselect
all_products = df['Product'].unique()
selected_products = st.sidebar.multiselect(
    "Select Product(s) for Analysis",
    options=all_products,
    default=all_products[0] if len(all_products) > 0 else [] # Defaults to the first product if available
)

# Region Selection Multiselect
all_regions = df['Region'].unique()
selected_regions = st.sidebar.multiselect(
    "Select Regional Data Feeds",
    options=all_regions,
    default=all_regions[0] if len(all_regions) > 0 else [] # Defaults to the first region if available
)

# Date Range Slider for temporal filtering
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.slider(
    "Temporal Scan Range",
    min_value=min_date.to_pydatetime(), # Convert to Python datetime for slider
    max_value=max_date.to_pydatetime(), # Convert to Python datetime for slider
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY-MM" # Display format for the slider
)

# Filter the DataFrame based on user selections
filtered_df = df[
    df['Product'].isin(selected_products) &
    df['Region'].isin(selected_regions) &
    (df['Date'] >= date_range[0]) &
    (df['Date'] <= date_range[1])
]

# Checkbox to display raw filtered data
if st.sidebar.checkbox("Activate Raw Data Display"):
    st.subheader(":: DECODED DATA LOGS ::")
    st.dataframe(filtered_df)
    # Using Streamlit's built-in colored markdown for a badge effect
    st.write(f"Total entries decoded: :blue-badge[{len(filtered_df)}]")

# --- Main Dashboard Sections ---

# Check if the filtered DataFrame is empty due to filter selections
if filtered_df.empty:
    st.warning("SYSTEM ALERT: No data signature detected for selected parameters. Adjust filters and re-initiate scan.")
else:
    st.markdown("---") # Separator

    # --- Section 1: Key Metrics Snapshot ---
    st.header(":: KEY METRIC SYNTHESIS ::")
    # Using columns for a structured layout of metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Latest Data Timestamp", df['Date'].max().strftime('%Y-%m'))
    with col2:
        st.metric("Regional Nodes Active", df['Region'].nunique())
    with col3:
        st.metric("Product Categories Monitored", df['Product'].nunique())

    # National average price per product across the *entire* dataset
    st.subheader("National Average Price per Product (Global Data Stream)")
    avg_national_price_product = df.groupby('Product')['Price (GHS)'].mean().reset_index()
    avg_national_price_product.columns = ['Product', 'Average Price (GHS)']
    st.dataframe(avg_national_price_product.set_index('Product').style.format({'Average Price (GHS)': "GHS {:.2f}"}))


    st.markdown("---") # Separator

    # --- Section 2: Line Charts for Price Trends ---
    st.header(":: TEMPORAL PRICE FLUCTUATION ::")

    # Monthly price trends per region for selected products
    st.subheader("Monthly Price Trajectories per Region (Filtered)")
    if not filtered_df.empty:
        # Create a matplotlib figure and axes for the plot
        fig_regional_trend, ax_regional_trend = plt.subplots(figsize=(12, 6))
        plt.style.use('dark_background') # Apply dark background theme for plots
        ax_regional_trend.set_facecolor('#0d1117') # Match app background color

        # Generate a color palette for multiple lines
        # Using a colormap that fits the futuristic theme
        colors = plt.colormaps.get_cmap('viridis', len(selected_products) * len(selected_regions))
        color_idx = 0 # Index for cycling through colors

        for product in selected_products:
            for region in selected_regions:
                # Filter data for the current product and region combination
                plot_data = filtered_df[(filtered_df['Product'] == product) & (filtered_df['Region'] == region)]
                if not plot_data.empty:
                    ax_regional_trend.plot(plot_data['Date'], plot_data['Price (GHS)'],
                                           label=f'{product} - {region}',
                                           marker='o', markersize=4,
                                           color=colors(color_idx)) # Assign color from the colormap
                    color_idx += 1 # Move to the next color

        # Set plot titles and labels with futuristic styling
        ax_regional_trend.set_title("Monthly Price Trajectories by Product and Region", color='#a0faff')
        ax_regional_trend.set_xlabel("Temporal Axis", color='#a0faff')
        ax_regional_trend.set_ylabel("Value (GHS)", color='#a0faff')
        ax_regional_trend.tick_params(axis='x', colors='#00ffc4') # Set X-axis tick color
        ax_regional_trend.tick_params(axis='y', colors='#00ffc4') # Set Y-axis tick color
        ax_regional_trend.legend(title="Product - Region", bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#a0faff', facecolor='#1a222e', edgecolor='#005642')
        ax_regional_trend.grid(True, linestyle=':', alpha=0.6, color='#005642') # Dotted grid lines for a digital feel
        fig_regional_trend.autofmt_xdate() # Automatically format X-axis dates
        st.pyplot(fig_regional_trend) # Display the plot in Streamlit
        plt.close(fig_regional_trend) # Close the figure to free up memory

    # Line plot for national average price per product (using filtered data)
    st.subheader("National Average Price Trajectory per Product (Filtered)")
    if not filtered_df.empty:
        # Calculate national average for filtered data, unstack for separate lines
        national_avg_filtered = filtered_df.groupby(['Date', 'Product'])['Price (GHS)'].mean().unstack()
        fig_national_avg, ax_national_avg = plt.subplots(figsize=(12, 6))
        plt.style.use('dark_background')
        ax_national_avg.set_facecolor('#0d1117')

        # Plot with 'plasma' colormap for another futuristic touch
        national_avg_filtered.plot(ax=ax_national_avg, marker='o', markersize=4, cmap='plasma')
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
        # Summary statistics (Mean, Min, Max, Std Dev) grouped by Product and Region
        st.subheader("Statistical Breakdown: Product & Region")
        summary_stats = filtered_df.groupby(['Product', 'Region'])['Price (GHS)'].agg(['mean', 'min', 'max', 'std']).reset_index()
        summary_stats.columns = ['Product', 'Region', 'Mean Price (GHS)', 'Min Price (GHS)', 'Max Price (GHS)', 'Std Dev (GHS)']
        st.dataframe(summary_stats.style.format({
            'Mean Price (GHS)': "GHS {:.2f}",
            'Min Price (GHS)': "GHS {:.2f}",
            'Max Price (GHS)': "GHS {:.2f}",
            'Std Dev (GHS)': "GHS {:.2f}"
        }))

        # Percentage price change (latest month vs. previous month)
        st.subheader("Temporal Value Oscillation: Last Cycle vs. Current Cycle")
        price_change_df = filtered_df.copy()
        # FIX: Convert Period object to string to avoid JSON serialization error with st.dataframe
        price_change_df['YearMonth'] = price_change_df['Date'].dt.to_period('M').astype(str)
        price_change_df = price_change_df.sort_values(by=['Product', 'Region', 'YearMonth'])

        price_change_df['Previous_Month_Price'] = price_change_df.groupby(['Product', 'Region'])['Price (GHS)'].shift(1)
        price_change_df['Price_Change_Pct'] = ((price_change_df['Price (GHS)'] - price_change_df['Previous_Month_Price']) / price_change_df['Previous_Month_Price']) * 100

        # Get the latest month's data for each product-region combination
        # The .apply method ensures the 'YearMonth' column (now string) is correctly handled
        latest_month_prices = price_change_df.groupby(['Product', 'Region']).apply(lambda x: x.iloc[-1] if not x.empty else pd.Series()).reset_index(drop=True)
        latest_month_prices = latest_month_prices[['Product', 'Region', 'YearMonth', 'Price (GHS)', 'Price_Change_Pct']]
        latest_month_prices.rename(columns={'Price (GHS)': 'Latest Value (GHS)'}, inplace=True)
        # Filter out rows where price change could not be calculated (e.g., only one month of data)
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
            # Use nlargest to get the top 3 regions by average price for each product
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
        # Create a pivot table for the heatmap
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
    if not filtered_df.empty and len(selected_regions) > 1: # Only show if at least two regions are selected for comparison
        avg_prices_product_region = filtered_df.groupby(['Product', 'Region'])['Price (GHS)'].mean().unstack()
        fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
        plt.style.use('dark_background')
        ax_bar.set_facecolor('#0d1117')
        avg_prices_product_region.plot(kind='bar', ax=ax_bar, width=0.8, cmap='viridis') # Use 'viridis' colormap for bars
        ax_bar.set_title("Average Product Values Across Selected Regions", color='#a0faff')
        ax_bar.set_xlabel("Product Category", color='#a0faff')
        ax_bar.set_ylabel("Average Value (GHS)", color='#a0faff')
        ax_bar.tick_params(axis='x', rotation=45, colors='#00ffc4')
        ax_bar.tick_params(axis='y', colors='#00ffc4')
        ax_bar.legend(title="Regional Node", bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='#a0faff', facecolor='#1a222e', edgecolor='#005642')
        ax_bar.grid(axis='y', linestyle=':', alpha=0.6, color='#005642')
        plt.tight_layout() # Adjust layout to prevent labels overlapping
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
        sns.boxplot(x='Product', y='Price (GHS)', data=filtered_df, ax=ax_box, palette='viridis') # Use 'viridis' palette for box plots
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

    # Function to convert DataFrame to CSV, cached for efficiency
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)

    # Download button for filtered data
    st.download_button(
        label="Download Filtered Data Log (.CSV)",
        data=csv,
        file_name="ghana_market_prices_filtered_data_log.csv",
        mime="text/csv",
    )

    st.info("NOTE: For graphic data extraction, right-click visual displays and select 'Save Image As...'.")

