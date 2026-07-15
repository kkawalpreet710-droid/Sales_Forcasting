import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------------------
# CUSTOM STYLING — purple/lavender theme inspired by modern SaaS dashboards
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

    /* Global font application */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .kpi-value, .brand-title {
        font-family: 'Manrope', sans-serif;
    }

    /* Overall app background */
    .stApp {
        background-color: #FAF9FC;
    }

    /* Sidebar brand header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.3px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E;
    }
    section[data-testid="stSidebar"] * {
        color: #F4F2FA !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 15px;
        padding: 4px 0;
    }

    /* Headings */
    h1 {
        color: #1A1A2E;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #1A1A2E;
        font-weight: 600;
    }

    /* Section header accent bar */
    .section-header {
        border-left: 4px solid #7C3AED;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    .section-header h3 {
        margin: 0;
    }

    /* KPI Card */
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(124, 58, 237, 0.08);
        border: 1px solid #EEEAF7;
    }
    .kpi-label {
        color: #6B6B85;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #1A1A2E;
        font-size: 28px;
        font-weight: 700;
    }
    .kpi-delta-positive {
        color: #16A34A;
        font-size: 13px;
        font-weight: 600;
    }
    .kpi-delta-negative {
        color: #DC2626;
        font-size: 13px;
        font-weight: 600;
    }

    /* Chart container card */
    .chart-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 12px rgba(124, 58, 237, 0.06);
        border: 1px solid #EEEAF7;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #7C3AED;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton>button:hover {
        background-color: #6D28D9;
        color: white;
    }

    /* Multiselect / selectbox tags */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #7C3AED;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER: styled KPI card renderer
# ---------------------------------------------------------
def kpi_card(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta:
        delta_class = "kpi-delta-positive" if delta_positive else "kpi-delta-negative"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{delta_class}">{arrow} {delta}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_header(title):
    st.markdown(f'<div class="section-header"><h3>{title}</h3></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# PLOTLY THEME — matching the purple/lavender palette
# ---------------------------------------------------------
PURPLE_SEQUENCE = ["#7C3AED", "#A78BFA", "#C4B5FD", "#DDD6FE", "#5B21B6", "#4C1D95"]

def style_fig(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1A1A2E", family="Arial, sans-serif"),
        title_font=dict(size=16, color="#1A1A2E"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, l=10, r=10, b=10)
    )
    fig.update_xaxes(gridcolor="#F0EEF7", zerolinecolor="#F0EEF7")
    fig.update_yaxes(gridcolor="#F0EEF7", zerolinecolor="#F0EEF7")
    return fig

# ---------------------------------------------------------
# DATA LOADING (cached so it only runs once, not on every click)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('dashboard_data.csv', parse_dates=['Order Date', 'Ship Date'])
    monthly_sales = pd.read_csv('monthly_sales.csv', parse_dates=['Order Date'])
    anomaly_data = pd.read_csv('anomaly_data.csv', parse_dates=['Order Date'])
    cluster_data = pd.read_csv('cluster_data.csv')
    prophet_forecast = pd.read_csv('prophet_forecast.csv', parse_dates=['ds'])
    segment_forecasts = pd.read_csv('segment_forecasts.csv', parse_dates=['ds'])
    segment_metrics = pd.read_csv('segment_metrics.csv')
    return df, monthly_sales, anomaly_data, cluster_data, prophet_forecast, segment_forecasts, segment_metrics

df, monthly_sales, anomaly_data, cluster_data, prophet_forecast, segment_forecasts, segment_metrics = load_data()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <span style="font-size:22px;">📊</span>
        <span class="brand-title">DemandIQ</span>
    </div>
    """,
    unsafe_allow_html=True
)
page = st.sidebar.radio(
    "NAVIGATION",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)

# ===========================================================
# PAGE 1 — SALES OVERVIEW DASHBOARD
# ===========================================================
if page == "Sales Overview":
    st.title("Sales Overview")
    st.markdown("<p style='color:#6B6B85; font-size:16px; margin-top:-10px;'>A high-level view of historical performance across years, regions, and categories.</p>", unsafe_allow_html=True)

    # --- Top KPI row ---
    yearly_sales = df.groupby(df['Order Date'].dt.year)['Sales'].sum().reset_index()
    yearly_sales.columns = ['Year', 'Total_Sales']

    total_sales_all = df['Sales'].sum()
    latest_year_sales = yearly_sales.iloc[-1]['Total_Sales']
    prev_year_sales = yearly_sales.iloc[-2]['Total_Sales']
    yoy_growth = ((latest_year_sales - prev_year_sales) / prev_year_sales) * 100
    total_orders = df['Order ID'].nunique()
    avg_order_value = df['Sales'].sum() / total_orders

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Sales (All Years)", f"${total_sales_all:,.0f}")
    with k2:
        kpi_card(f"Sales in {int(yearly_sales.iloc[-1]['Year'])}", f"${latest_year_sales:,.0f}",
                  delta=f"{yoy_growth:.1f}% YoY", delta_positive=(yoy_growth >= 0))
    with k3:
        kpi_card("Total Orders", f"{total_orders:,}")
    with k4:
        kpi_card("Avg Order Value", f"${avg_order_value:,.2f}")

    st.write("")  # spacing

    # --- Total sales by year (bar chart) ---
    section_header("Total Sales by Year")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_year = px.bar(
        yearly_sales, x='Year', y='Total_Sales',
        text_auto='.2s',
        labels={'Total_Sales': 'Total Sales ($)'},
        color_discrete_sequence=["#7C3AED"]
    )
    fig_year.update_traces(marker_line_width=0, marker_cornerradius=6)
    st.plotly_chart(style_fig(fig_year), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Monthly sales trend line chart ---
    section_header("Monthly Sales Trend")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_monthly = px.line(
        monthly_sales, x='Order Date', y='Total_Sales',
        labels={'Total_Sales': 'Total Sales ($)', 'Order Date': 'Date'},
        color_discrete_sequence=["#7C3AED"]
    )
    fig_monthly.update_traces(line=dict(width=3), fill='tozeroy', fillcolor='rgba(124, 58, 237, 0.08)')
    st.plotly_chart(style_fig(fig_monthly), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Sales by region and category with interactive filters ---
    section_header("Sales by Region and Category")

    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect(
            "Filter by Region:",
            options=df['Region'].unique(),
            default=list(df['Region'].unique())
        )
    with col2:
        selected_categories = st.multiselect(
            "Filter by Category:",
            options=df['Category'].unique(),
            default=list(df['Category'].unique())
        )

    filtered_df = df[df['Region'].isin(selected_regions) & df['Category'].isin(selected_categories)]

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
        fig_region = px.bar(region_sales, x='Region', y='Sales', title="Sales by Region",
                             color_discrete_sequence=["#7C3AED"])
        fig_region.update_traces(marker_cornerradius=6)
        st.plotly_chart(style_fig(fig_region), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_category = px.pie(category_sales, names='Category', values='Sales', title="Sales Share by Category",
                               color_discrete_sequence=PURPLE_SEQUENCE, hole=0.45)
        st.plotly_chart(style_fig(fig_category), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    kpi_card("Total Filtered Sales", f"${filtered_df['Sales'].sum():,.2f}")

# ===========================================================
# PAGE 2 — FORECAST EXPLORER
# ===========================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.markdown("<p style='color:#6B6B85; font-size:16px; margin-top:-10px;'>Explore Prophet-based 3-month forecasts by category or region.</p>", unsafe_allow_html=True)

    # --- Dropdown: Category or Region selector ---
    col1, col2 = st.columns([1, 1])
    with col1:
        segment_type = st.selectbox("Select Segment Type:", ["Overall", "Category", "Region"])

    with col2:
        if segment_type == "Category":
            available_categories = sorted(segment_forecasts[segment_forecasts['segment_type'] == 'Category']['segment_value'].unique())
            segment_value = st.selectbox("Select Category:", available_categories)
        elif segment_type == "Region":
            available_regions = sorted(segment_forecasts[segment_forecasts['segment_type'] == 'Region']['segment_value'].unique())
            segment_value = st.selectbox("Select Region:", available_regions)
        else:
            segment_value = "Overall"

    # --- Date range slider: 1, 2, or 3 months ahead ---
    horizon = st.select_slider(
        "Forecast Horizon (months ahead):",
        options=[1, 2, 3],
        value=3
    )

    # --- Build the historical series for the selected segment ---
    if segment_type == "Overall":
        hist_series = monthly_sales.rename(columns={'Total_Sales': 'y', 'Order Date': 'ds'})
    elif segment_type == "Category":
        filtered = df[df['Category'] == segment_value]
        hist_series = filtered.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
        hist_series.columns = ['ds', 'y']
    else:
        filtered = df[df['Region'] == segment_value]
        hist_series = filtered.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
        hist_series.columns = ['ds', 'y']

    # --- Forecast display ---
    section_header(f"Forecast: {segment_value} — Next {horizon} Month(s)")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    # Look up the correct forecast: overall model, or the matching segment-specific model
    if segment_type == "Overall":
        forecast_slice = prophet_forecast.head(horizon)
    else:
        forecast_slice = segment_forecasts[
            (segment_forecasts['segment_type'] == segment_type) &
            (segment_forecasts['segment_value'] == segment_value)
        ].sort_values('ds').head(horizon)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=hist_series['ds'], y=hist_series['y'],
        mode='lines', name='Historical', line=dict(color='#A78BFA', width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_slice['ds'], y=forecast_slice['yhat'],
        mode='lines+markers', name='Forecast', line=dict(color='#7C3AED', width=3, dash='dash')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([forecast_slice['ds'], forecast_slice['ds'][::-1]]),
        y=pd.concat([forecast_slice['yhat_upper'], forecast_slice['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(124,58,237,0.12)',
        line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval', showlegend=True
    ))
    st.plotly_chart(style_fig(fig_forecast), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- MAE and RMSE display below the chart ---
    st.write("")
    if segment_type == "Overall":
        section_header("Model Accuracy (Overall Prophet Model)")
        mae_val, rmse_val, mape_val = "$11,275.24", "$14,629.52", "16.08%"
    else:
        matched = segment_metrics[
            (segment_metrics['segment_type'] == segment_type) &
            (segment_metrics['segment_value'] == segment_value)
        ]
        if matched.empty:
            st.warning(f"No forecast model was trained for {segment_value}. Please select a different {segment_type.lower()}.")
            st.stop()
        row = matched.iloc[0]
        section_header(f"Model Accuracy ({segment_value})")
        mae_val, rmse_val, mape_val = f"${row['MAE']:,.2f}", f"${row['RMSE']:,.2f}", f"{row['MAPE']:.2f}%"

    m1, m2, m3 = st.columns(3)
    with m1:
        kpi_card("Model MAE", mae_val)
    with m2:
        kpi_card("Model RMSE", rmse_val)
    with m3:
        kpi_card("Model MAPE", mape_val)

    st.caption(f"Metrics from a Prophet model trained and evaluated specifically on {segment_value}'s historical data (80/20 time-ordered train/test split), consistent with the methodology used for the overall model in Task 3.")

# ===========================================================
# PAGE 3 — ANOMALY REPORT
# ===========================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.markdown("<p style='color:#6B6B85; font-size:16px; margin-top:-10px;'>Weekly sales anomalies detected via Isolation Forest and Z-Score methods.</p>", unsafe_allow_html=True)

    # --- Method selector ---
    method = st.radio("Detection Method:", ["Isolation Forest", "Z-Score", "Both (Agreement Only)"], horizontal=True)

    if method == "Isolation Forest":
        flagged = anomaly_data[anomaly_data['anomaly_iso'] == 'Anomaly']
        marker_color = "#7C3AED"
    elif method == "Z-Score":
        flagged = anomaly_data[anomaly_data['anomaly_zscore'] == 'Anomaly']
        marker_color = "#FF8B6B"
    else:
        flagged = anomaly_data[(anomaly_data['anomaly_iso'] == 'Anomaly') & (anomaly_data['anomaly_zscore'] == 'Anomaly')]
        marker_color = "#1DB894"

    # --- KPI row ---
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Anomalies Detected", f"{len(flagged)}")
    with k2:
        kpi_card("% of All Weeks", f"{len(flagged) / len(anomaly_data) * 100:.1f}%")
    with k3:
        kpi_card("Total Weeks Analyzed", f"{len(anomaly_data)}")

    st.write("")

    # --- Anomaly chart ---
    section_header("Weekly Sales with Anomalies Highlighted")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(go.Scatter(
        x=anomaly_data['Order Date'], y=anomaly_data['Total_Sales'],
        mode='lines', name='Weekly Sales', line=dict(color='#C4B5FD', width=2)
    ))
    fig_anomaly.add_trace(go.Scatter(
        x=flagged['Order Date'], y=flagged['Total_Sales'],
        mode='markers', name='Anomaly', marker=dict(color=marker_color, size=11, symbol='circle',
                                                       line=dict(color='white', width=1))
    ))
    st.plotly_chart(style_fig(fig_anomaly), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Table of anomaly dates and sales values ---
    section_header("Detected Anomalies")
    display_table = flagged[['Order Date', 'Total_Sales']].sort_values('Order Date').reset_index(drop=True)
    display_table['Order Date'] = display_table['Order Date'].dt.strftime('%b %d, %Y')
    display_table['Total_Sales'] = display_table['Total_Sales'].apply(lambda x: f"${x:,.2f}")
    display_table.columns = ['Week Of', 'Sales']
    st.dataframe(display_table, use_container_width=True, hide_index=True)

# ===========================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ===========================================================
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.markdown("<p style='color:#6B6B85; font-size:16px; margin-top:-10px;'>K-Means clusters (k=4) grouping sub-categories by demand behavior.</p>", unsafe_allow_html=True)

    CLUSTER_LABELS = {
        0: "High Volume, Core Demand",
        1: "Explosive Growth, High-Value Outlier",
        2: "Low Volume, Steady Demand",
        3: "Declining, High Volatility"
    }
    CLUSTER_COLORS = {0: "#7C3AED", 1: "#FF8B6B", 2: "#1DB894", 3: "#DC2626"}
    CLUSTER_STRATEGY = {
        0: "Maintain steady baseline inventory. High-volume revenue drivers — prioritize consistent availability.",
        1: "Increase safety stock, monitor monthly. Explosive growth + high volatility needs active management.",
        2: "Lean, low-buffer inventory with infrequent reordering. Low volatility means low forecasting risk.",
        3: "Reduce inventory commitment, reorder cautiously. Declining + volatile — the riskiest category to overstock."
    }

    cluster_data['Cluster_Label'] = cluster_data['Cluster'].map(CLUSTER_LABELS)

    # --- Cluster filter ---
    selected_cluster_labels = st.multiselect(
        "Filter by Segment:",
        options=list(CLUSTER_LABELS.values()),
        default=list(CLUSTER_LABELS.values())
    )
    filtered_clusters = cluster_data[cluster_data['Cluster_Label'].isin(selected_cluster_labels)]

    # --- PCA scatter plot ---
    section_header("Cluster Visualization (PCA-Reduced)")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_cluster = go.Figure()
    for c in sorted(filtered_clusters['Cluster'].unique()):
        subset = filtered_clusters[filtered_clusters['Cluster'] == c]
        fig_cluster.add_trace(go.Scatter(
            x=subset['PCA1'], y=subset['PCA2'],
            mode='markers+text', text=subset['Sub-Category'], textposition='top center',
            name=CLUSTER_LABELS[c],
            marker=dict(color=CLUSTER_COLORS[c], size=16, line=dict(color='white', width=1.5))
        ))
    fig_cluster.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    st.plotly_chart(style_fig(fig_cluster), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Table: sub-categories by cluster, with stocking strategy ---
    section_header("Segment Details & Stocking Strategy")
    for c in sorted(filtered_clusters['Cluster'].unique()):
        subset = filtered_clusters[filtered_clusters['Cluster'] == c]
        st.markdown(
            f"""
            <div class="chart-card" style="border-left: 5px solid {CLUSTER_COLORS[c]};">
                <div style="font-weight:700; font-size:16px; color:#1A1A2E; margin-bottom:4px;">
                    {CLUSTER_LABELS[c]}
                </div>
                <div style="color:#6B6B85; font-size:14px; margin-bottom:10px;">
                    {CLUSTER_STRATEGY[c]}
                </div>
                <div style="color:#1A1A2E; font-size:14px;">
                    <b>Sub-Categories:</b> {', '.join(subset['Sub-Category'].tolist())}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
