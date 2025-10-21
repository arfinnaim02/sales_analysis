import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from PIL import Image
import sys
import traceback

# ---------------------------
# Helper: safe fig -> PNG
# ---------------------------
def fig_to_png_bytes(fig, width=1200, height=600, scale=1):
    """
    Try to export Plotly figure to PNG bytes using built-in renderer (kaleido).
    If that fails, return None and the calling code will show an error message.
    """
    try:
        # plotly >= 4.9 with kaleido support
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        return img_bytes, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Failed to export figure to PNG (kaleido may be missing). Error: {e}\n{tb}"

# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def preprocess_base(df_raw):
    df = df_raw.copy()
    # Convert margin to numeric
    df['Product Base Margin'] = df['Product Base Margin'].astype(str).str.rstrip('%').replace('', np.nan)
    df['Margin_Numeric'] = pd.to_numeric(df['Product Base Margin'], errors='coerce') / 100

    # Convert dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

    # Drop rows with invalid dates (this is core to your logic)
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    # Additional columns
    df['Ship_Time_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Profit'] = df['Sales'] * df['Margin_Numeric']
    df['Month'] = df['Order Date'].dt.month
    df['Month_Name'] = df['Order Date'].dt.strftime('%B')
    df['Margin_Category'] = pd.cut(df['Margin_Numeric'],
                                    bins=[0, 0.35, 0.45, 0.55, 1.0],
                                    labels=['Low (0-35%)', 'Medium (35-45%)', 'High (45-55%)', 'Very High (55%+)'])
    return df

# Page configuration
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide", page_icon="📊")

# Title
st.title("📊 Comprehensive Sales Data Analysis")

# File upload
uploaded_file = st.file_uploader("Upload your sales CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data (cached)
    try:
        df_raw = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        raise

    st.success(f"✅ File uploaded successfully! Total rows: {len(df_raw)}")

    # ========================================================================
    # DATA PREPROCESSING SECTION
    # ========================================================================
    st.header("🔧 Data Preprocessing")

    with st.expander("📋 View Raw Data & Preprocessing Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raw Data Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)

        with col2:
            st.subheader("Data Info")
            st.write(f"**Total Rows:** {len(df_raw)}")
            st.write(f"**Total Columns:** {len(df_raw.columns)}")
            st.write(f"**Columns:** {', '.join(df_raw.columns.tolist())}")

            # Check for missing values
            missing_values = df_raw.isnull().sum()
            if missing_values.sum() > 0:
                st.write("**Missing Values:**")
                st.dataframe(missing_values[missing_values > 0], use_container_width=True)
            else:
                st.write("**Missing Values:** None")

        st.subheader("Preprocessing Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Margins",
                ["Keep as is", "Remove rows", "Fill with median", "Fill with mean"]
            )

        with col2:
            remove_zero_sales = st.checkbox("Remove zero/negative sales", value=True)

        with col3:
            remove_outliers = st.checkbox("Remove outliers (Sales > 99th percentile)", value=False)

        # Apply preprocessing based on cached base preprocessing
        df = preprocess_base(df_raw)

        # Handle missing margins options (modify df in memory)
        if handle_missing == "Remove rows":
            rows_before = len(df)
            df = df.dropna(subset=['Margin_Numeric'])
            st.info(f"Removed {rows_before - len(df)} rows with missing margins")
        elif handle_missing == "Fill with median":
            median_val = df['Margin_Numeric'].median()
            df['Margin_Numeric'].fillna(median_val, inplace=True)
            st.info(f"Filled missing margins with median: {median_val*100:.2f}%")
        elif handle_missing == "Fill with mean":
            mean_val = df['Margin_Numeric'].mean()
            df['Margin_Numeric'].fillna(mean_val, inplace=True)
            st.info(f"Filled missing margins with mean: {mean_val*100:.2f}%")

        # Remove zero/negative sales
        if remove_zero_sales:
            rows_before = len(df)
            df = df[df['Sales'] > 0]
            if rows_before - len(df) > 0:
                st.info(f"Removed {rows_before - len(df)} rows with zero/negative sales")

        # Remove outliers
        if remove_outliers:
            rows_before = len(df)
            threshold = df['Sales'].quantile(0.99)
            df = df[df['Sales'] <= threshold]
            st.info(f"Removed {rows_before - len(df)} outlier rows (Sales > ${threshold:,.2f})")

        # Recompute the dependent columns if needed
        df['Ship_Time_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
        df['Profit'] = df['Sales'] * df['Margin_Numeric']
        df['Month'] = df['Order Date'].dt.month
        df['Month_Name'] = df['Order Date'].dt.strftime('%B')
        df['Margin_Category'] = pd.cut(df['Margin_Numeric'],
                                        bins=[0, 0.35, 0.45, 0.55, 1.0],
                                        labels=['Low (0-35%)', 'Medium (35-45%)', 'High (45-55%)', 'Very High (55%+)'])

        st.success(f"✅ Preprocessing complete! Final dataset: {len(df)} rows")

    # Store preprocessed data (backup if the expander was not opened)
    if 'df' not in locals():
        df = preprocess_base(df_raw)

    st.divider()

    # ========================================================================
    # GLOBAL FILTERS
    # ========================================================================
    st.sidebar.header("🔍 Global Filters")

    # Guard: if columns missing, show message and stop
    required_cols = ['Region', 'State or Province', 'Order Date', 'Order ID', 'City', 'Sales', 'Quantity ordered new', 'Product Base Margin', 'Ship Date']
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        st.error(f"Your CSV is missing required columns: {missing_req}. Please fix and re-upload.")
        st.stop()

    selected_regions = st.sidebar.multiselect(
        "Select Region(s)",
        options=sorted(df['Region'].unique()),
        default=df['Region'].unique()
    )

    selected_states = st.sidebar.multiselect(
        "Select State(s)",
        options=sorted(df['State or Province'].unique()),
        default=df['State or Province'].unique()
    )

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Order Date'].min(), df['Order Date'].max()),
        min_value=df['Order Date'].min(),
        max_value=df['Order Date'].max()
    )

    # Filter data
    filtered_df = df[
        (df['Region'].isin(selected_regions)) &
        (df['State or Province'].isin(selected_states)) &
        (df['Order Date'] >= pd.to_datetime(date_range[0])) &
        (df['Order Date'] <= pd.to_datetime(date_range[1]))
    ]

    st.sidebar.info(f"Showing {len(filtered_df)} out of {len(df)} rows")

    # Initialize list to store all figures for dashboard
    all_figures = []

    # ========================================================================
    # KEY METRICS
    # ========================================================================
    st.header("📈 Key Metrics Overview")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
    with col2:
        st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
    with col3:
        st.metric("Average Margin", f"{filtered_df['Margin_Numeric'].mean()*100:.2f}%")
    with col4:
        st.metric("Total Orders", f"{filtered_df['Order ID'].nunique():,}")
    with col5:
        st.metric("Total Quantity", f"{filtered_df['Quantity ordered new'].sum():,}")

    st.divider()

    # ========================================================================
    # SALES PERFORMANCE ANALYSIS
    # ========================================================================
    st.header("💰 Sales Performance Analysis")

    # Chart 1: Sales by Region
    with st.expander("📊 Sales by Region", expanded=True):
        col_filter, col_chart = st.columns([1, 3])
        with col_filter:
            chart1_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart1_regions"
            )

        chart1_df = filtered_df[filtered_df['Region'].isin(chart1_regions)]
        region_sales = chart1_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        fig1 = px.bar(x=region_sales.values, y=region_sales.index,
                      orientation='h',
                      title="Sales by Region",
                      labels={'x': 'Sales ($)', 'y': 'Region'},
                      color=region_sales.values,
                      color_continuous_scale='Blues')
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
        all_figures.append(fig1)

        img_bytes, err = fig_to_png_bytes(fig1, width=1200, height=600)
        if img_bytes is not None:
            st.download_button("💾 Download Chart", img_bytes, "sales_by_region.png", "image/png", key="dl_fig1")
        else:
            st.warning("Chart export failed for Sales by Region. Install `kaleido` (pip install kaleido) to enable PNG exports. Error details shown in console.")
            # Optionally write error to console
            # st.write(err)

    # Chart 2: Profit by Region
    with st.expander("💰 Profit by Region", expanded=True):
        col_filter, col_chart = st.columns([1, 3])
        with col_filter:
            chart2_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart2_regions"
            )

        chart2_df = filtered_df[filtered_df['Region'].isin(chart2_regions)]
        profit_by_region = chart2_df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
        fig2 = px.bar(x=profit_by_region.values, y=profit_by_region.index,
                      orientation='h',
                      title="Profit by Region",
                      labels={'x': 'Profit ($)', 'y': 'Region'},
                      color=profit_by_region.values,
                      color_continuous_scale='Greens')
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)
        all_figures.append(fig2)

        img_bytes, err = fig_to_png_bytes(fig2, width=1200, height=600)
        if img_bytes is not None:
            st.download_button("💾 Download Chart", img_bytes, "profit_by_region.png", "image/png", key="dl_fig2")
        else:
            st.warning("Chart export failed for Profit by Region. Install `kaleido` (pip install kaleido).")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 3: Top States by Sales
        with st.expander("🏛️ Top States by Sales", expanded=True):
            top_n_states = st.slider("Number of States", 5, 20, 10, key="top_states_slider")
            chart3_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart3_regions"
            )

            chart3_df = filtered_df[filtered_df['Region'].isin(chart3_regions)]
            top_states = chart3_df.groupby('State or Province')['Sales'].sum().nlargest(top_n_states).sort_values()
            fig3 = px.bar(x=top_states.values, y=top_states.index,
                          orientation='h',
                          title=f"Top {top_n_states} States by Sales",
                          labels={'x': 'Sales ($)', 'y': 'State'},
                          color=top_states.values,
                          color_continuous_scale='Purples')
            fig3.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig3, use_container_width=True)
            all_figures.append(fig3)

            img_bytes, err = fig_to_png_bytes(fig3, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "top_states_sales.png", "image/png", key="dl_fig3")
            else:
                st.warning("Chart export failed for Top States. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 4: Top Cities by Sales
        with st.expander("🏙️ Top Cities by Sales", expanded=True):
            top_n_cities = st.slider("Number of Cities", 5, 20, 10, key="top_cities_slider")
            chart4_states = st.multiselect(
                "Filter States",
                options=sorted(filtered_df['State or Province'].unique()),
                default=filtered_df['State or Province'].unique(),
                key="chart4_states"
            )

            chart4_df = filtered_df[filtered_df['State or Province'].isin(chart4_states)]
            top_cities = chart4_df.groupby('City')['Sales'].sum().nlargest(top_n_cities).sort_values()
            fig4 = px.bar(x=top_cities.values, y=top_cities.index,
                          orientation='h',
                          title=f"Top {top_n_cities} Cities by Sales",
                          labels={'x': 'Sales ($)', 'y': 'City'},
                          color=top_cities.values,
                          color_continuous_scale='Oranges')
            fig4.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig4, use_container_width=True)
            all_figures.append(fig4)

            img_bytes, err = fig_to_png_bytes(fig4, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "top_cities_sales.png", "image/png", key="dl_fig4")
            else:
                st.warning("Chart export failed for Top Cities. Install `kaleido` (pip install kaleido).")

    st.divider()

    # ========================================================================
    # TIME-BASED ANALYSIS
    # ========================================================================
    st.header("📅 Time-Based Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 5: Monthly Sales Trend
        with st.expander("📈 Monthly Sales Trend", expanded=True):
            chart5_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart5_regions"
            )

            chart5_df = filtered_df[filtered_df['Region'].isin(chart5_regions)]
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            monthly_sales = chart5_df.groupby('Month_Name')['Sales'].sum()
            monthly_sales = monthly_sales.reindex([m for m in month_order if m in monthly_sales.index])

            fig5 = px.line(x=monthly_sales.index, y=monthly_sales.values,
                           title="Monthly Sales Trend",
                           labels={'x': 'Month', 'y': 'Sales ($)'},
                           markers=True)
            fig5.update_traces(line_color='coral', line_width=3, marker_size=10)
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
            all_figures.append(fig5)

            img_bytes, err = fig_to_png_bytes(fig5, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "monthly_sales_trend.png", "image/png", key="dl_fig5")
            else:
                st.warning("Chart export failed for Monthly Sales Trend. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 6: Monthly Order Count
        with st.expander("📊 Monthly Order Count", expanded=True):
            chart6_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart6_regions"
            )

            chart6_df = filtered_df[filtered_df['Region'].isin(chart6_regions)]
            monthly_orders = chart6_df.groupby('Month_Name')['Order ID'].count()
            monthly_orders = monthly_orders.reindex([m for m in month_order if m in monthly_orders.index])

            fig6 = px.bar(x=monthly_orders.index, y=monthly_orders.values,
                          title="Monthly Order Count",
                          labels={'x': 'Month', 'y': 'Number of Orders'},
                          color=monthly_orders.values,
                          color_continuous_scale='Viridis')
            fig6.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig6, use_container_width=True)
            all_figures.append(fig6)

            img_bytes, err = fig_to_png_bytes(fig6, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "monthly_order_count.png", "image/png", key="dl_fig6")
            else:
                st.warning("Chart export failed for Monthly Order Count. Install `kaleido` (pip install kaleido).")

    # Shipping Time Analysis
    st.subheader("🚚 Shipping Time Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Shipping Time", f"{filtered_df['Ship_Time_Days'].mean():.2f} days")
    with col2:
        st.metric("Median Shipping Time", f"{filtered_df['Ship_Time_Days'].median():.0f} days")
    with col3:
        st.metric("Min Shipping Time", f"{filtered_df['Ship_Time_Days'].min()} days")
    with col4:
        st.metric("Max Shipping Time", f"{filtered_df['Ship_Time_Days'].max()} days")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 7: Shipping Time Distribution
        with st.expander("📦 Shipping Time Distribution", expanded=True):
            chart7_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart7_regions"
            )

            chart7_df = filtered_df[filtered_df['Region'].isin(chart7_regions)]
            fig7 = px.histogram(chart7_df, x='Ship_Time_Days',
                               title="Shipping Time Distribution",
                               labels={'Ship_Time_Days': 'Days to Ship', 'count': 'Frequency'},
                               nbins=20,
                               color_discrete_sequence=['lightcoral'])
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
            all_figures.append(fig7)

            img_bytes, err = fig_to_png_bytes(fig7, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "shipping_time_dist.png", "image/png", key="dl_fig7")
            else:
                st.warning("Chart export failed for Shipping Time Distribution. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 8: Shipping Time by Region
        with st.expander("🌍 Shipping Time by Region", expanded=True):
            chart8_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart8_regions"
            )

            chart8_df = filtered_df[filtered_df['Region'].isin(chart8_regions)]
            ship_by_region = chart8_df.groupby('Region')['Ship_Time_Days'].mean().sort_values()
            fig8 = px.bar(x=ship_by_region.values, y=ship_by_region.index,
                          orientation='h',
                          title="Average Shipping Time by Region",
                          labels={'x': 'Average Days', 'y': 'Region'},
                          color=ship_by_region.values,
                          color_continuous_scale='RdYlGn_r')
            fig8.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig8, use_container_width=True)
            all_figures.append(fig8)

            img_bytes, err = fig_to_png_bytes(fig8, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "shipping_time_region.png", "image/png", key="dl_fig8")
            else:
                st.warning("Chart export failed for Shipping Time by Region. Install `kaleido` (pip install kaleido).")

    st.divider()

    # ========================================================================
    # GEOGRAPHIC ANALYSIS
    # ========================================================================
    st.header("🗺️ Geographic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 9: Average Order Value by Region
        with st.expander("💵 Average Order Value by Region", expanded=True):
            chart9_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart9_regions"
            )

            chart9_df = filtered_df[filtered_df['Region'].isin(chart9_regions)]
            avg_order_region = chart9_df.groupby('Region')['Sales'].mean().sort_values(ascending=False)
            fig9 = px.bar(x=avg_order_region.values, y=avg_order_region.index,
                          orientation='h',
                          title="Average Order Value by Region",
                          labels={'x': 'Average Sales ($)', 'y': 'Region'},
                          color=avg_order_region.values,
                          color_continuous_scale='Teal')
            fig9.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig9, use_container_width=True)
            all_figures.append(fig9)

            img_bytes, err = fig_to_png_bytes(fig9, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "avg_order_value_region.png", "image/png", key="dl_fig9")
            else:
                st.warning("Chart export failed for Avg Order Value. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 10: Order Distribution by Region
        with st.expander("🥧 Order Distribution by Region", expanded=True):
            chart10_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart10_regions"
            )

            chart10_df = filtered_df[filtered_df['Region'].isin(chart10_regions)]
            order_count_region = chart10_df.groupby('Region')['Order ID'].count().sort_values(ascending=False)
            fig10 = px.pie(values=order_count_region.values, names=order_count_region.index,
                           title="Order Distribution by Region",
                           color_discrete_sequence=px.colors.sequential.RdBu)
            fig10.update_layout(height=400)
            st.plotly_chart(fig10, use_container_width=True)
            all_figures.append(fig10)

            img_bytes, err = fig_to_png_bytes(fig10, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "order_distribution_region.png", "image/png", key="dl_fig10")
            else:
                st.warning("Chart export failed for Order Distribution. Install `kaleido` (pip install kaleido).")

    st.divider()

    # ========================================================================
    # ORDER ANALYSIS
    # ========================================================================
    st.header("🛒 Order Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Order Value", f"${filtered_df['Sales'].mean():.2f}")
    with col2:
        st.metric("Median Order Value", f"${filtered_df['Sales'].median():.2f}")
    with col3:
        correlation = filtered_df['Quantity ordered new'].corr(filtered_df['Sales'])
        st.metric("Qty-Sales Correlation", f"{correlation:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 11: Quantity vs Sales
        with st.expander("📊 Quantity vs Sales", expanded=True):
            chart11_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart11_regions"
            )

            chart11_df = filtered_df[filtered_df['Region'].isin(chart11_regions)]
            # sample for performance
            sample_df = chart11_df.sample(min(500, len(chart11_df)))
            fig11 = px.scatter(sample_df, x='Quantity ordered new', y='Sales',
                              title="Quantity vs Sales",
                              labels={'Quantity ordered new': 'Quantity Ordered', 'Sales': 'Sales ($)'},
                              color='Region',
                              hover_data=['City', 'State or Province'])
            fig11.update_layout(height=400)
            st.plotly_chart(fig11, use_container_width=True)
            all_figures.append(fig11)

            img_bytes, err = fig_to_png_bytes(fig11, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "quantity_vs_sales.png", "image/png", key="dl_fig11")
            else:
                st.warning("Chart export failed for Quantity vs Sales. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 12: Order Size Distribution
        with st.expander("📈 Order Size Distribution", expanded=True):
            chart12_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart12_regions"
            )

            chart12_df = filtered_df[filtered_df['Region'].isin(chart12_regions)]
            fig12 = px.histogram(chart12_df, x='Sales',
                                title="Order Size Distribution",
                                labels={'Sales': 'Order Value ($)', 'count': 'Frequency'},
                                nbins=30,
                                color_discrete_sequence=['steelblue'])
            fig12.update_layout(height=400)
            st.plotly_chart(fig12, use_container_width=True)
            all_figures.append(fig12)

            img_bytes, err = fig_to_png_bytes(fig12, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "order_size_dist.png", "image/png", key="dl_fig12")
            else:
                st.warning("Chart export failed for Order Size Distribution. Install `kaleido` (pip install kaleido).")

    st.divider()

    # ========================================================================
    # PROFITABILITY ANALYSIS
    # ========================================================================
    st.header("💎 Profitability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Chart 13: Margin Distribution
        with st.expander("🥧 Margin Distribution", expanded=True):
            chart13_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart13_regions"
            )

            chart13_df = filtered_df[filtered_df['Region'].isin(chart13_regions)]
            margin_dist = chart13_df['Margin_Category'].value_counts()
            fig13 = px.pie(values=margin_dist.values, names=margin_dist.index,
                           title="Margin Distribution",
                           color_discrete_sequence=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            fig13.update_layout(height=400)
            st.plotly_chart(fig13, use_container_width=True)
            all_figures.append(fig13)

            img_bytes, err = fig_to_png_bytes(fig13, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "margin_distribution.png", "image/png", key="dl_fig13")
            else:
                st.warning("Chart export failed for Margin Distribution. Install `kaleido` (pip install kaleido).")

    with col2:
        # Chart 14: Profit by Margin Category (COMPLETED)
        with st.expander("💰 Profit by Margin Category", expanded=True):
            chart14_regions = st.multiselect(
                "Filter Regions",
                options=sorted(filtered_df['Region'].unique()),
                default=filtered_df['Region'].unique(),
                key="chart14_regions"
            )

            chart14_df = filtered_df[filtered_df['Region'].isin(chart14_regions)]
            # NOTE: As decided: Total Profit grouped by Margin Category
            profit_by_margin = chart14_df.groupby('Margin_Category')['Profit'].sum().sort_values(ascending=False).fillna(0)
            fig14 = px.bar(x=profit_by_margin.index.astype(str), y=profit_by_margin.values,
                           title="Profit by Margin Category (Total Profit)",
                           labels={'x': 'Margin Category', 'y': 'Total Profit ($)'},
                           color=profit_by_margin.values,
                           color_continuous_scale='YlGn')
            fig14.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig14, use_container_width=True)
            all_figures.append(fig14)

            img_bytes, err = fig_to_png_bytes(fig14, width=1200, height=600)
            if img_bytes is not None:
                st.download_button("💾 Download", img_bytes, "profit_by_margin.png", "image/png", key="dl_fig14")
            else:
                st.warning("Chart export failed for Profit by Margin Category. Install `kaleido` (pip install kaleido).")

    st.divider()

    # ========================================================================
    # COMPLETE DASHBOARD EXPORT
    # ========================================================================
    st.header("📸 Export Complete Dashboard")

    st.info("Click the button below to generate and download a complete dashboard image with all charts")

    # NOTE: st.button does not accept use_container_width in some versions; keep original but be safe
    if st.button("🎨 Generate Complete Dashboard Image"):
        with st.spinner("Generating dashboard... This may take a moment..."):
            try:
                # Create a subplot figure with all charts
                fig_dashboard = make_subplots(
                    rows=5, cols=3,
                    subplot_titles=(
                        'Sales by Region', 'Profit by Region', 'Top States by Sales',
                        'Top Cities by Sales', 'Monthly Sales Trend', 'Monthly Order Count',
                        'Shipping Time Distribution', 'Avg Shipping Time by Region', 'Avg Order Value by Region',
                        'Order Distribution by Region', 'Quantity vs Sales', 'Order Size Distribution',
                        'Margin Distribution', 'Profit by Margin Category', ''
                    ),
                    specs=[
                        [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                        [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                        [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}],
                        [{"type": "pie"}, {"type": "scatter"}, {"type": "histogram"}],
                        [{"type": "pie"}, {"type": "bar"}, {"type": "table"}]
                    ],
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08
                )

                # Add traces from all figures (recompute here to be safe)
                # Row 1
                region_sales = filtered_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
                fig_dashboard.add_trace(
                    go.Bar(x=region_sales.values, y=region_sales.index, orientation='h',
                           marker_color='lightblue', showlegend=False),
                    row=1, col=1
                )

                profit_by_region = filtered_df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
                fig_dashboard.add_trace(
                    go.Bar(x=profit_by_region.values, y=profit_by_region.index, orientation='h',
                           marker_color='lightgreen', showlegend=False),
                    row=1, col=2
                )

                top_states = filtered_df.groupby('State or Province')['Sales'].sum().nlargest(10).sort_values()
                fig_dashboard.add_trace(
                    go.Bar(x=top_states.values, y=top_states.index, orientation='h',
                           marker_color='plum', showlegend=False),
                    row=1, col=3
                )

                # Row 2
                top_cities = filtered_df.groupby('City')['Sales'].sum().nlargest(10).sort_values()
                fig_dashboard.add_trace(
                    go.Bar(x=top_cities.values, y=top_cities.index, orientation='h',
                           marker_color='orange', showlegend=False),
                    row=2, col=1
                )

                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                monthly_sales = filtered_df.groupby('Month_Name')['Sales'].sum()
                monthly_sales = monthly_sales.reindex([m for m in month_order if m in monthly_sales.index])
                fig_dashboard.add_trace(
                    go.Scatter(x=monthly_sales.index, y=monthly_sales.values, mode='lines+markers',
                               line=dict(color='coral', width=3), marker=dict(size=8), showlegend=False),
                    row=2, col=2
                )

                monthly_orders = filtered_df.groupby('Month_Name')['Order ID'].count()
                monthly_orders = monthly_orders.reindex([m for m in month_order if m in monthly_orders.index])
                fig_dashboard.add_trace(
                    go.Bar(x=monthly_orders.index, y=monthly_orders.values,
                           marker_color='teal', showlegend=False),
                    row=2, col=3
                )

                # Row 3
                fig_dashboard.add_trace(
                    go.Histogram(x=filtered_df['Ship_Time_Days'], marker_color='lightcoral',
                                 showlegend=False, nbinsx=20),
                    row=3, col=1
                )

                ship_by_region = filtered_df.groupby('Region')['Ship_Time_Days'].mean().sort_values()
                fig_dashboard.add_trace(
                    go.Bar(x=ship_by_region.values, y=ship_by_region.index, orientation='h',
                           marker_color='salmon', showlegend=False),
                    row=3, col=2
                )

                avg_order_region = filtered_df.groupby('Region')['Sales'].mean().sort_values(ascending=False)
                fig_dashboard.add_trace(
                    go.Bar(x=avg_order_region.values, y=avg_order_region.index, orientation='h',
                           marker_color='cadetblue', showlegend=False),
                    row=3, col=3
                )

                # Row 4
                order_count_region = filtered_df.groupby('Region')['Order ID'].count()
                fig_dashboard.add_trace(
                    go.Pie(labels=order_count_region.index, values=order_count_region.values,
                           showlegend=True),
                    row=4, col=1
                )

                # Sample scatter plot (limit points for performance)
                sample_df = filtered_df.sample(min(500, len(filtered_df)))
                for region in sample_df['Region'].unique():
                    region_data = sample_df[sample_df['Region'] == region]
                    fig_dashboard.add_trace(
                        go.Scatter(x=region_data['Quantity ordered new'], y=region_data['Sales'],
                                   mode='markers', name=region, showlegend=False,
                                   marker=dict(size=5, opacity=0.6)),
                        row=4, col=2
                    )

                fig_dashboard.add_trace(
                    go.Histogram(x=filtered_df['Sales'], marker_color='steelblue',
                                 showlegend=False, nbinsx=30),
                    row=4, col=3
                )

                # Row 5
                margin_dist = filtered_df['Margin_Category'].value_counts()
                fig_dashboard.add_trace(
                    go.Pie(labels=margin_dist.index, values=margin_dist.values,
                           showlegend=True),
                    row=5, col=1
                )

                profit_by_margin = filtered_df.groupby('Margin_Category')['Profit'].sum().sort_values(ascending=False).fillna(0)
                fig_dashboard.add_trace(
                    go.Bar(x=profit_by_margin.index.astype(str), y=profit_by_margin.values,
                           marker_color='yellowgreen', showlegend=False),
                    row=5, col=2
                )

                # Add summary table
                summary_data = {
                    'Metric': ['Total Sales', 'Total Profit', 'Avg Margin', 'Total Orders', 'Total Quantity'],
                    'Value': [
                        f"${filtered_df['Sales'].sum():,.2f}",
                        f"${filtered_df['Profit'].sum():,.2f}",
                        f"{filtered_df['Margin_Numeric'].mean()*100:.2f}%",
                        f"{filtered_df['Order ID'].nunique():,}",
                        f"{filtered_df['Quantity ordered new'].sum():,}"
                    ]
                }
                fig_dashboard.add_trace(
                    go.Table(
                        header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                                    fill_color='paleturquoise',
                                    align='left',
                                    font=dict(size=12, color='black')),
                        cells=dict(values=[summary_data['Metric'], summary_data['Value']],
                                   fill_color='lavender',
                                   align='left',
                                   font=dict(size=11))
                    ),
                    row=5, col=3
                )

                # Update layout
                fig_dashboard.update_layout(
                    title_text="<b>Complete Sales Analysis Dashboard</b>",
                    title_font_size=24,
                    title_x=0.5,
                    showlegend=False,
                    height=2400,
                    width=2400
                )

                # Export as image (use helper to handle missing kaleido)
                dashboard_img_bytes, err = fig_to_png_bytes(fig_dashboard, width=2400, height=2400, scale=2)
                if dashboard_img_bytes is None:
                    st.error("Failed to generate PNG for the dashboard. Please install `kaleido` (pip install kaleido) and try again.")
                else:
                    st.success("✅ Dashboard generated successfully!")
                    # Display preview
                    st.image(dashboard_img_bytes, caption="Complete Dashboard Preview", use_container_width=True)

                    # Download button
                    st.download_button(
                        label="📥 Download Complete Dashboard",
                        data=dashboard_img_bytes,
                        file_name="complete_sales_dashboard.png",
                        mime="image/png",
                    )

            except Exception as e:
                st.error(f"An unexpected error occurred while generating the dashboard: {e}")
                st.exception(traceback.format_exc())

    st.divider()

    # ========================================================================
    # RAW DATA
    # ========================================================================
    st.header("📄 Processed Data")
    st.dataframe(filtered_df, use_container_width=True, height=400)

    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload your CSV file to begin the analysis")
    st.markdown("""
    ### Expected CSV Format:
    - Product Base Margin
    - Country
    - Region
    - State or Province
    - City
    - Postal Code
    - Order Date
    - Ship Date
    - Quantity ordered new
    - Sales
    - Order ID

    ### Features:
    - ✅ Data preprocessing with multiple options
    - 📊 Interactive visualizations with individual filters
    - 💾 Download each chart as PNG image
    - 🎨 Generate complete dashboard image
    - 🔍 Filter by region, state, and date
    - 📥 Export filtered data
    """)
