import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(layout="wide", page_title="Web Traffic Analytics — Dark", initial_sidebar_state="expanded")

# ================================
# CUSTOM DARK CSS
# ================================
st.markdown(
    """
    <style>
    :root {
        --primary: #60a5fa;
        --bg: #0b1220;
        --card: #0f1724;
        --muted: #94a3b8;
        --accent: #7c3aed;
    }
    body {
        background-color: var(--bg);
        color: #e6eef8;
    }
    .stApp {
        background-color: var(--bg);
    }
    .header {
        padding: 18px 24px;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(124,58,237,0.08), rgba(96,165,250,0.04));
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
    }
    .title {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .subtitle {
        color: var(--muted);
        margin: 0;
        font-size: 13px;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 10px rgba(2,6,23,0.6);
    }
    .metric {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
    }
    .metric-sub {
        color: var(--muted);
        font-size: 12px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path("web_traffic.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_data(DATA_PATH)

# HEADER
st.markdown('<div class="header"><h1 class="title">Web Traffic Analytics</h1><p class="subtitle">Dark Theme Dashboard • Streamlit</p></div>', unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("Filters & Forecast")
pages = sorted(df['page'].unique().tolist())

page_selected = st.sidebar.multiselect("Select pages", pages, default=pages[:3])

date_min = df['date'].min().date()
date_max = df['date'].max().date()

start_date, end_date = st.sidebar.date_input("Date range", value=(date_min, date_max))

forecast_days = st.sidebar.number_input("Forecast days", min_value=7, max_value=90, value=14, step=1)
show_heatmap = st.sidebar.checkbox("Show heatmap", value=True)
anomaly_method = st.sidebar.selectbox("Anomaly detection", ["IQR", "Z-score"])

# FILTERING
data = df.copy()
if page_selected:
    data = data[data['page'].isin(page_selected)]

data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]

# KPIs
total_views = int(data['views'].sum())
total_visitors = int(data['visitors'].sum())
avg_time = round(data['avg_time_sec'].mean(), 1)

k1, k2, k3 = st.columns([1.6,1,1])
with k1:
    st.markdown(f'<div class="card"><div class="metric">📈 {total_views:,}</div><div class="metric-sub">Total views</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="card"><div class="metric">👥 {total_visitors:,}</div><div class="metric-sub">Total visitors</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="card"><div class="metric">⏱️ {avg_time}</div><div class="metric-sub">Avg time (s)</div></div>', unsafe_allow_html=True)

st.markdown("---")

# TIME SERIES GRAPH
st.markdown("### Views over time")
time_series = data.groupby('date')['views'].sum().reset_index()

fig = px.line(time_series, x='date', y='views', template="plotly_dark", markers=True)
fig.update_traces(line=dict(width=2, color="#60a5fa"))
st.plotly_chart(fig, use_container_width=True)

# ANOMALY DETECTION
st.markdown("### Anomalies")

ts = time_series.copy()

if anomaly_method == "IQR":
    q1 = ts['views'].quantile(0.25)
    q3 = ts['views'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    ts['anomaly'] = (ts['views'] < lower) | (ts['views'] > upper)
else:
    mean = ts['views'].mean()
    std = ts['views'].std()
    ts['anomaly'] = abs(ts['views'] - mean) > 2.5 * std

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=ts['date'], y=ts['views'], mode='lines', line=dict(color="#60a5fa")))
fig2.add_trace(go.Scatter(x=ts[ts['anomaly']]['date'], y=ts[ts['anomaly']]['views'],
                          mode='markers', marker=dict(color="#ef4444", size=8)))
fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# HEATMAP
if show_heatmap:
    st.markdown("### Heatmap: average views by weekday and month")

    heat = data.copy()
    heat['weekday'] = heat['date'].dt.day_name()
    heat['month'] = heat['date'].dt.month_name().str.slice(stop=3)

    pivot = heat.groupby(['weekday','month'])['views'].mean().reset_index()

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot['weekday'] = pd.Categorical(pivot['weekday'], categories=order, ordered=True)

    table = pivot.pivot(index="weekday", columns="month", values="views").fillna(0)

    fig_h = px.imshow(table, color_continuous_scale='viridis', labels={"color":"Avg views"})
    fig_h.update_layout(template="plotly_dark")
    st.plotly_chart(fig_h, use_container_width=True)

# TOP PAGES
st.markdown("### Top pages")
top_pages = data.groupby('page')[['views','visitors']].sum().sort_values('views', ascending=False)
st.table(top_pages.head(10))

# FORECAST
st.markdown("### Forecast (Holt-Winters)")

if len(time_series) >= 30:
    ts_hw = time_series.set_index('date').asfreq('D')['views'].fillna(method='ffill')

    try:
        model = ExponentialSmoothing(ts_hw, trend="add", seasonal="add", seasonal_periods=7)
        fitted = model.fit()
        pred = fitted.forecast(forecast_days)

        figf = px.line(template="plotly_dark")
        figf.add_scatter(x=ts_hw.index, y=ts_hw.values, name="history")
        figf.add_scatter(x=pred.index, y=pred.values, name="forecast")
        st.plotly_chart(figf, use_container_width=True)

        st.dataframe(pred.rename("forecast_views"))

    except Exception as e:
        st.error(f"Error in forecast: {e}")

# DOWNLOAD DATA
st.markdown("---")
st.download_button("Download filtered CSV", data.to_csv(index=False), "filtered.csv")
