from __future__ import annotations

import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    calculate_kpis,
    generate_business_insights,
    get_category_summary,
    get_customer_frequency,
    get_customer_summary,
    get_region_category_matrix,
    get_region_summary,
    get_sales_timeseries,
    segment_customers,
)
from src.forecasting import compare_forecast_models

REQUIRED_COLUMNS = (
    "order_id",
    "order_date",
    "region",
    "category",
    "product",
    "sales",
    "profit",
    "quantity",
    "customer_name",
)

DATA_FILE = "salesdashboard/src/sales.csv"

st.set_page_config(
    page_title="Sales Data Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 1.25rem 1rem;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="metric-container"] label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.07em !important;
            text-transform: uppercase !important;
            color: #64748b !important;
        }
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            line-height: 1.2 !important;
        }
        .stTabs [data-baseweb="tab-panel"] {padding-top: 1.25rem;}
        h2, h3 {color: #0f172a !important; font-weight: 700 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv_dataset(path: str, modified_time: float) -> pd.DataFrame:
    _ = modified_time
    frame = pd.read_csv(path).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns in {path}: {missing_text}")

    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    for column in ("sales", "profit", "quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["order_date", "sales"]).copy()
    frame["sales"] = frame["sales"].clip(lower=0)
    frame["raw_profit"] = frame["profit"].fillna(0)
    frame["profit"] = frame["raw_profit"].clip(lower=0)
    frame["quantity"] = frame["quantity"].fillna(1).clip(lower=1)
    frame["order_id"] = frame["order_id"].astype(str).str.strip()
    frame["customer_name"] = frame["customer_name"].astype(str).str.strip().replace({"": "Unknown Customer"})

    return frame.sort_values("order_date").reset_index(drop=True)



def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def _df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()


def _chart_layout(height: int = 400, margin_top: int = 50, **extra) -> dict:
    layout: dict = dict(
        height=height,
        margin=dict(l=10, r=10, t=margin_top, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#374151"),
        title_font=dict(size=15, color="#0f172a"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", showline=True),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", showline=False),
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0", font_size=12),
    )
    layout.update(extra)
    return layout


st.title("Sales Data Analytics Dashboard")
st.caption(
    "Interactive sales analytics, business intelligence, and machine learning forecasting built with Streamlit, Pandas, Plotly, and Scikit-learn."
)

try:
    sales_df = load_csv_dataset(DATA_FILE, os.path.getmtime(DATA_FILE))
except Exception as exc:
    st.error(f"Failed to load {DATA_FILE}: {exc}")
    st.stop()

with st.sidebar:
    st.header("Dashboard Controls")
    forecast_periods = st.slider("Forecast horizon (months)", min_value=3, max_value=12, value=6)

min_date = sales_df["order_date"].min().date()
max_date = sales_df["order_date"].max().date()

with st.sidebar:
    st.subheader("Filters")
    selected_dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    region_options = sorted(sales_df["region"].dropna().unique().tolist())
    category_options = sorted(sales_df["category"].dropna().unique().tolist())
    product_options = sorted(sales_df["product"].dropna().unique().tolist())

    selected_regions = st.multiselect("Region", region_options, default=region_options)
    selected_categories = st.multiselect("Category", category_options, default=category_options)
    selected_products = st.multiselect("Product", product_options, default=product_options)

with st.sidebar:
    st.divider()
    st.subheader("Export Data")
    st.caption("Downloads apply current filters.")

filtered_df = sales_df.loc[
    (sales_df["order_date"].dt.date >= start_date)
    & (sales_df["order_date"].dt.date <= end_date)
    & (sales_df["region"].isin(selected_regions))
    & (sales_df["category"].isin(selected_categories))
    & (sales_df["product"].isin(selected_products))
].copy()

if filtered_df.empty:
    st.warning("No data matches the selected filters. Adjust the filters to continue.")
    st.stop()

export_df = filtered_df.drop(columns=["raw_profit"], errors="ignore")
with st.sidebar:
    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button(
            "⬇ CSV",
            data=_df_to_csv(export_df),
            file_name="sales_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "⬇ Excel",
            data=_df_to_excel(export_df),
            file_name="sales_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

kpis = calculate_kpis(filtered_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", format_currency(kpis.total_sales))
col2.metric("Total Profit", format_currency(kpis.total_profit))
col3.metric("Orders", format_number(kpis.total_orders))
col4.metric("Average Order Value", format_currency(kpis.average_order_value))

overview_tab, product_tab, region_tab, customer_tab, forecast_tab, insights_tab = st.tabs(
    ["Overview", "Products", "Regions", "Customers", "Forecast", "Insights"]
)

with overview_tab:
    st.subheader("Sales Trend Analysis")
    granularity = st.radio("Trend granularity", ["D", "W", "M"], horizontal=True, format_func=lambda x: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[x])
    time_series = get_sales_timeseries(filtered_df, frequency=granularity)
    point_count = len(time_series)
    series_mode = "lines+markers" if point_count <= 2 else "lines"
    series_shape = "linear" if point_count <= 2 else "spline"

    trend_figure = go.Figure()
    trend_figure.add_trace(
        go.Scatter(
            x=time_series["order_date"],
            y=time_series["sales"],
            mode=series_mode,
            name="Sales",
            line=dict(color="#2563EB", width=2.5, shape=series_shape, smoothing=0.8),
            marker=dict(size=8, color="#2563EB"),
        )
    )
    trend_figure.add_trace(
        go.Scatter(
            x=time_series["order_date"],
            y=time_series["profit"],
            mode=series_mode,
            name="Profit",
            line=dict(color="#16A34A", width=2, shape=series_shape, smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(22, 163, 74, 0.08)",
            marker=dict(size=7, color="#16A34A"),
        )
    )
    trend_figure.update_layout(**_chart_layout(420, 20, legend_title_text=""))
    st.plotly_chart(trend_figure, use_container_width=True)

    total_profit_over_period = float(time_series["profit"].sum()) if not time_series.empty else 0.0
    total_sales_loss = float(filtered_df.loc[filtered_df["profit"] < 0, "profit"].sum())
    total_sales_loss = abs(total_sales_loss)
    average_sales = float(time_series["sales"].mean()) if not time_series.empty else 0.0
    total_sales_over_period = float(time_series["sales"].sum()) if not time_series.empty else 0.0
    profit_rate = (total_profit_over_period / total_sales_over_period * 100) if total_sales_over_period > 0 else 0.0
    loss_rate = (total_sales_loss / total_sales_over_period * 100) if total_sales_over_period > 0 else 0.0

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Total Sales Loss", format_currency(total_sales_loss))
    metric_2.metric("Average Sales", format_currency(average_sales))
    metric_3.metric("Profit Rate", f"{profit_rate:.2f}%")
    metric_4.metric("Loss Rate", f"{loss_rate:.2f}%")

    monthly = get_sales_timeseries(filtered_df, frequency="M")
    monthly["growth_pct"] = monthly["sales"].pct_change().fillna(0) * 100

    left, right = st.columns(2)
    with left:
        monthly_bar = px.bar(
            monthly,
            x="order_date",
            y="sales",
            color="sales",
            color_continuous_scale="Blues",
            title="Monthly Revenue",
        )
        monthly_bar.update_layout(**_chart_layout(360, coloraxis_showscale=False))
        st.plotly_chart(monthly_bar, use_container_width=True)
    with right:
        growth_line = px.line(
            monthly,
            x="order_date",
            y="growth_pct",
            markers=True,
            title="Month-over-Month Growth (%)",
        )
        growth_line.update_traces(line_color="#7C3AED")
        growth_line.update_layout(**_chart_layout(360))
        st.plotly_chart(growth_line, use_container_width=True)

with product_tab:
    st.subheader("Product Performance Analysis")
    _ps_exp = (
        filtered_df.groupby(["product", "category"], as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        .sort_values("sales", ascending=False)
    )
    _p_ex1, _p_ex2, _p_ex3 = st.columns([1, 1, 6])
    with _p_ex1:
        st.download_button("⬇ CSV", _df_to_csv(_ps_exp), "products.csv", "text/csv", use_container_width=True, key="prod_csv")
    with _p_ex2:
        st.download_button("⬇ Excel", _df_to_excel(_ps_exp), "products.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="prod_xlsx")
    product_summary = (
        filtered_df.groupby(["product", "category"], as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
    )
    product_summary["sales_profit_score"] = product_summary["quantity"] * product_summary["profit"]

    loss_summary = (
        filtered_df[filtered_df["raw_profit"] < 0]
        .groupby(["product", "category"], as_index=False)
        .agg(raw_profit=("raw_profit", "sum"), quantity=("quantity", "sum"), sales=("sales", "sum"))
    )
    loss_summary["loss_score"] = loss_summary["quantity"] * loss_summary["raw_profit"].abs()
    loss_summary["loss_label"] = loss_summary["raw_profit"].apply(lambda v: f"₹{abs(v):,.0f} loss")

    top_products = product_summary.sort_values("sales_profit_score", ascending=False).head(10)
    low_products = loss_summary.sort_values("loss_score", ascending=False).head(10)
    category_summary = get_category_summary(filtered_df)

    p1, p2 = st.columns(2)
    with p1:
        top_chart = px.bar(
            top_products.sort_values("sales_profit_score"),
            x="sales_profit_score",
            y="product",
            color="category",
            orientation="h",
            title="Top Products by Units Sold x Profit",
            hover_data={"quantity": True, "profit": ":,.2f", "sales_profit_score": ":,.2f"},
        )
        top_chart.update_xaxes(title="Units x Profit Score")
        top_chart.update_yaxes(categoryorder="total ascending")
        top_chart.update_layout(**_chart_layout(420))
        st.plotly_chart(top_chart, use_container_width=True)
    with p2:
        if loss_summary.empty:
            st.info("No loss-making products in the selected filters.")
        else:
            low_chart = px.bar(
                low_products.sort_values("loss_score"),
                x="loss_score",
                y="product",
                color="category",
                orientation="h",
                title="Highest Loss Products (Units × Loss)",
                hover_data={"quantity": True, "loss_label": True, "loss_score": ":,.0f"},
            )
            low_chart.update_xaxes(title="Loss Score (Units × Total Loss)")
            low_chart.update_yaxes(categoryorder="total ascending")
            low_chart.update_layout(**_chart_layout(420))
            st.plotly_chart(low_chart, use_container_width=True)

    p3, p4 = st.columns(2)
    with p3:
        category_pie = px.pie(
            category_summary,
            names="category",
            values="sales",
            hole=0.45,
            title="Category-wise Sales Distribution",
        )
        category_pie.update_layout(**_chart_layout(360))
        st.plotly_chart(category_pie, use_container_width=True)
    with p4:
        demand_scatter = px.scatter(
            filtered_df.groupby(["product", "category"], as_index=False).agg(sales=("sales", "sum"), quantity=("quantity", "sum")),
            x="quantity",
            y="sales",
            size="sales",
            color="category",
            hover_name="product",
            title="Product Demand Trend",
        )
        demand_scatter.update_layout(**_chart_layout(360))
        st.plotly_chart(demand_scatter, use_container_width=True)

    st.divider()
    st.subheader("Custom Chart Builder")
    st.caption("Build your own product chart — pick a chart type, metric, and how many products to show.")

    cb1, cb2, cb3, cb4 = st.columns([2, 2, 2, 1])
    with cb1:
        custom_chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Horizontal Bar", "Pie Chart", "Donut Chart", "Treemap", "Funnel Chart",
             "Scatter Plot", "Bubble Chart", "Line Chart", "Area Chart", "Heatmap", "Box Plot"],
            key="custom_chart_type",
        )
    with cb2:
        custom_metric = st.selectbox("Metric", ["sales", "profit", "quantity"], key="custom_metric",
                                     format_func=lambda x: {"sales": "Total Sales", "profit": "Total Profit", "quantity": "Units Sold"}[x])
    with cb3:
        custom_group = st.selectbox("Group By", ["product", "category"], key="custom_group",
                                    format_func=lambda x: "Product" if x == "product" else "Category")
    with cb4:
        custom_top_n = st.number_input("Top N", min_value=3, max_value=30, value=10, step=1, key="custom_top_n")

    custom_data = (
        filtered_df.groupby([custom_group, "category"] if custom_group == "product" else [custom_group], as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
    )
    custom_data = custom_data.sort_values(custom_metric, ascending=False).head(int(custom_top_n))
    metric_label = {"sales": "Total Sales ($)", "profit": "Total Profit ($)", "quantity": "Units Sold"}[custom_metric]
    color_col = "category" if custom_group == "product" else custom_group

    if custom_chart_type == "Bar Chart":
        custom_fig = px.bar(
            custom_data.sort_values(custom_metric, ascending=False),
            x=custom_group,
            y=custom_metric,
            color=color_col,
            title=f"Top {int(custom_top_n)} {custom_group.title()}s by {metric_label}",
            text_auto=".2s",
        )
        custom_fig.update_xaxes(title="", categoryorder="total descending")
        custom_fig.update_yaxes(title=metric_label)
        custom_fig.update_traces(textposition="outside")
        custom_fig.update_layout(**_chart_layout(420))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Horizontal Bar":
        custom_fig = px.bar(
            custom_data.sort_values(custom_metric),
            x=custom_metric,
            y=custom_group,
            color=color_col,
            orientation="h",
            title=f"Top {int(custom_top_n)} {custom_group.title()}s by {metric_label}",
            text_auto=".2s",
        )
        custom_fig.update_xaxes(title=metric_label)
        custom_fig.update_yaxes(categoryorder="total ascending", title="")
        custom_fig.update_traces(textposition="outside")
        custom_fig.update_layout(**_chart_layout(max(380, int(custom_top_n) * 38)))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Pie Chart":
        custom_fig = px.pie(
            custom_data,
            names=custom_group,
            values=custom_metric,
            title=f"{custom_group.title()} Share by {metric_label}",
        )
        custom_fig.update_traces(textposition="inside", textinfo="percent+label")
        custom_fig.update_layout(**_chart_layout(460))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Donut Chart":
        custom_fig = px.pie(
            custom_data,
            names=custom_group,
            values=custom_metric,
            hole=0.5,
            title=f"{custom_group.title()} Share by {metric_label}",
        )
        custom_fig.update_traces(textposition="inside", textinfo="percent+label")
        custom_fig.update_layout(**_chart_layout(460))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Treemap":
        treemap_path = ["category", "product"] if custom_group == "product" else ["category"]
        treemap_data = (
            filtered_df.groupby(["category", "product"], as_index=False)
            .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        ) if custom_group == "product" else custom_data
        custom_fig = px.treemap(
            treemap_data,
            path=treemap_path,
            values=custom_metric,
            color=custom_metric,
            color_continuous_scale="Blues",
            title=f"Treemap: {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_traces(textinfo="label+value+percent parent")
        custom_fig.update_layout(**_chart_layout(500))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Funnel Chart":
        funnel_data = custom_data.sort_values(custom_metric, ascending=False)
        custom_fig = px.funnel(
            funnel_data,
            x=custom_metric,
            y=custom_group,
            color=color_col,
            title=f"Funnel: {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_layout(**_chart_layout(max(380, int(custom_top_n) * 38)))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Scatter Plot":
        scatter_data = (
            filtered_df.groupby([custom_group, "category"] if custom_group == "product" else [custom_group], as_index=False)
            .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        ).head(int(custom_top_n))
        x_col = "quantity" if custom_metric == "sales" else "sales"
        custom_fig = px.scatter(
            scatter_data,
            x=x_col,
            y=custom_metric,
            color=color_col,
            hover_name=custom_group,
            title=f"Scatter: {metric_label} vs {'Units Sold' if x_col == 'quantity' else 'Total Sales ($)'}",
        )
        custom_fig.update_layout(**_chart_layout(420))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Bubble Chart":
        bubble_data = (
            filtered_df.groupby([custom_group, "category"] if custom_group == "product" else [custom_group], as_index=False)
            .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        ).head(int(custom_top_n))
        custom_fig = px.scatter(
            bubble_data,
            x="quantity",
            y="sales",
            size=custom_metric,
            color=color_col,
            hover_name=custom_group,
            size_max=60,
            title=f"Bubble Chart: Size = {metric_label}",
        )
        custom_fig.update_layout(**_chart_layout(460))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Line Chart":
        line_data = (
            filtered_df.groupby(["order_date", custom_group], as_index=False)
            .agg(**{custom_metric: (custom_metric, "sum")})
        )
        top_items = custom_data[custom_group].tolist()
        line_data = line_data[line_data[custom_group].isin(top_items)]
        custom_fig = px.line(
            line_data,
            x="order_date",
            y=custom_metric,
            color=custom_group,
            markers=True,
            title=f"Trend Over Time: {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_xaxes(title="Date")
        custom_fig.update_yaxes(title=metric_label)
        custom_fig.update_layout(**_chart_layout(420))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Area Chart":
        area_data = (
            filtered_df.groupby(["order_date", custom_group], as_index=False)
            .agg(**{custom_metric: (custom_metric, "sum")})
        )
        top_items = custom_data[custom_group].tolist()
        area_data = area_data[area_data[custom_group].isin(top_items)]
        custom_fig = px.area(
            area_data,
            x="order_date",
            y=custom_metric,
            color=custom_group,
            title=f"Area Trend: {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_xaxes(title="Date")
        custom_fig.update_yaxes(title=metric_label)
        custom_fig.update_layout(**_chart_layout(420))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Heatmap":
        if custom_group == "product":
            heatmap_pivot = (
                filtered_df.groupby(["category", "product"], as_index=False)
                .agg(**{custom_metric: (custom_metric, "sum")})
            )
            top_products_list = (
                heatmap_pivot.groupby("product")[custom_metric].sum()
                .nlargest(int(custom_top_n)).index.tolist()
            )
            heatmap_pivot = heatmap_pivot[heatmap_pivot["product"].isin(top_products_list)]
            heatmap_matrix = heatmap_pivot.pivot(index="category", columns="product", values=custom_metric).fillna(0)
        else:
            region_cat = (
                filtered_df.groupby(["region", "category"], as_index=False)
                .agg(**{custom_metric: (custom_metric, "sum")})
            )
            heatmap_matrix = region_cat.pivot(index="region", columns="category", values=custom_metric).fillna(0)

        custom_fig = px.imshow(
            heatmap_matrix,
            text_auto=".2s",
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Heatmap: {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_layout(**_chart_layout(460))
        st.plotly_chart(custom_fig, use_container_width=True)

    elif custom_chart_type == "Box Plot":
        box_data = filtered_df[filtered_df[custom_group].isin(custom_data[custom_group].tolist())]
        custom_fig = px.box(
            box_data,
            x=custom_group,
            y=custom_metric,
            color=color_col,
            points="all",
            title=f"Distribution of {metric_label} by {custom_group.title()}",
        )
        custom_fig.update_xaxes(title="", categoryorder="total descending")
        custom_fig.update_yaxes(title=metric_label)
        custom_fig.update_layout(**_chart_layout(460))
        st.plotly_chart(custom_fig, use_container_width=True)

with region_tab:
    st.subheader("Regional Sales Analysis")
    region_summary = get_region_summary(filtered_df)
    region_matrix = get_region_category_matrix(filtered_df)
    _r_ex1, _r_ex2, _r_ex3 = st.columns([1, 1, 6])
    with _r_ex1:
        st.download_button("⬇ CSV", _df_to_csv(region_summary), "regions.csv", "text/csv", use_container_width=True, key="reg_csv")
    with _r_ex2:
        st.download_button("⬇ Excel", _df_to_excel(region_summary), "regions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="reg_xlsx")

    r1, r2 = st.columns(2)
    with r1:
        region_sales_chart = px.bar(
            region_summary,
            x="region",
            y="sales",
            color="profit_margin",
            color_continuous_scale="Viridis",
            title="Regional Sales Distribution",
        )
        region_sales_chart.update_layout(**_chart_layout(380))
        st.plotly_chart(region_sales_chart, use_container_width=True)
    with r2:
        region_profit_chart = px.bar(
            region_summary,
            x="region",
            y="profit",
            color="orders",
            color_continuous_scale="Tealgrn",
            title="Region-wise Profit",
        )
        region_profit_chart.update_layout(**_chart_layout(380))
        st.plotly_chart(region_profit_chart, use_container_width=True)

    heatmap_source = region_matrix.set_index("region")
    heatmap = px.imshow(
        heatmap_source,
        text_auto=".0f",
        color_continuous_scale="Blues",
        aspect="auto",
        title="Sales Heatmap: Region vs Category",
    )
    heatmap.update_layout(**_chart_layout(400))
    st.plotly_chart(heatmap, use_container_width=True)

    best_region = region_summary.iloc[0]
    weakest_region = region_summary.sort_values("profit").iloc[0]
    st.info(
        f"Best performing region: **{best_region['region']}** with {format_currency(best_region['sales'])} sales. "
        f"Lowest profit region: **{weakest_region['region']}** with {format_currency(weakest_region['profit'])} profit."
    )

with customer_tab:
    st.subheader("Customer Insights")
    top_customers = get_customer_summary(filtered_df)
    frequency_df = get_customer_frequency(filtered_df)
    _c_ex1, _c_ex2, _c_ex3 = st.columns([1, 1, 6])
    with _c_ex1:
        st.download_button("⬇ CSV", _df_to_csv(top_customers), "customers.csv", "text/csv", use_container_width=True, key="cust_csv")
    with _c_ex2:
        st.download_button("⬇ Excel", _df_to_excel(top_customers), "customers.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="cust_xlsx")

    c1, c2 = st.columns(2)
    with c1:
        customer_sales_chart = px.bar(
            top_customers.sort_values("sales"),
            x="sales",
            y="customer_name",
            orientation="h",
            color="orders",
            title="Top Customers by Revenue",
        )
        customer_sales_chart.update_layout(**_chart_layout(420))
        st.plotly_chart(customer_sales_chart, use_container_width=True)
    with c2:
        order_frequency_chart = px.histogram(
            frequency_df,
            x="orders",
            nbins=20,
            title="Customer Order Frequency",
            color_discrete_sequence=["#2563EB"],
        )
        order_frequency_chart.update_layout(**_chart_layout(420))
        st.plotly_chart(order_frequency_chart, use_container_width=True)

    repeat_rate = (frequency_df["orders"] > 1).mean() * 100 if not frequency_df.empty else 0
    avg_customer_value = frequency_df["sales"].mean() if not frequency_df.empty else 0
    st.markdown(
        f"**Repeat customer rate:** {repeat_rate:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Average customer value:** {format_currency(avg_customer_value)}"
    )
    st.dataframe(top_customers, use_container_width=True, hide_index=True)

    st.subheader("Customer Segmentation (K-Means Clustering)")
    st.caption("Customers grouped into High, Medium, and Low value tiers using K-Means clustering on revenue, profit, orders, and quantity.")

    customer_segments = segment_customers(filtered_df)

    segment_colors = {"High Value": "#16A34A", "Medium Value": "#2563EB", "Low Value": "#DC2626"}
    segment_counts = customer_segments["segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Customers"]

    s1, s2 = st.columns(2)
    with s1:
        segment_pie = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            color="Segment",
            color_discrete_map=segment_colors,
            hole=0.45,
            title="Customer Distribution by Segment",
        )
        segment_pie.update_layout(**_chart_layout(360))
        st.plotly_chart(segment_pie, use_container_width=True)
    with s2:
        segment_scatter = px.scatter(
            customer_segments,
            x="total_orders",
            y="total_sales",
            color="segment",
            size="total_profit",
            hover_name="customer_name",
            color_discrete_map=segment_colors,
            title="Segment Map: Orders vs Revenue (size = Profit)",
        )
        segment_scatter.update_layout(**_chart_layout(360))
        st.plotly_chart(segment_scatter, use_container_width=True)

    segment_summary = (
        customer_segments.groupby("segment", as_index=False)
        .agg(
            Customers=("customer_name", "count"),
            Avg_Sales=("total_sales", "mean"),
            Avg_Profit=("total_profit", "mean"),
            Avg_Orders=("total_orders", "mean"),
        )
        .rename(columns={"segment": "Segment", "Avg_Sales": "Avg Revenue", "Avg_Profit": "Avg Profit", "Avg_Orders": "Avg Orders"})
    )
    segment_summary["Avg Revenue"] = segment_summary["Avg Revenue"].map(format_currency)
    segment_summary["Avg Profit"] = segment_summary["Avg Profit"].map(format_currency)
    segment_summary["Avg Orders"] = segment_summary["Avg Orders"].map(lambda value: f"{value:.1f}")
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)

with forecast_tab:
    st.subheader("Sales Forecast")
    st.caption("Three forecasting methods are tested against your historical data. The one that predicts most accurately drives the forecast below.")
    _f_ex1, _f_ex2, _f_ex3 = st.columns([1, 1, 6])

    comparison_result = compare_forecast_models(filtered_df, forecast_periods=forecast_periods)
    if comparison_result.error:
        st.warning(comparison_result.error)
    else:
        history_frame = comparison_result.history.copy()
        future_frame = comparison_result.future_frame.copy()

        forecast_export_df = future_frame[["order_date", "predicted_sales"]].copy()
        forecast_export_df.columns = ["Forecast Month", "Predicted Sales"]
        with _f_ex1:
            st.download_button("⬇ CSV", _df_to_csv(forecast_export_df), "forecast.csv", "text/csv", use_container_width=True, key="fc_csv")
        with _f_ex2:
            st.download_button("⬇ Excel", _df_to_excel(forecast_export_df), "forecast.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="fc_xlsx")

        recent_avg = history_frame["sales"].tail(3).mean() if not history_frame.empty else 0
        projected_avg = future_frame["predicted_sales"].mean() if not future_frame.empty else 0
        growth_pct = (projected_avg - recent_avg) / recent_avg * 100 if recent_avg > 0 else 0
        projected_total = float(future_frame["predicted_sales"].sum()) if not future_frame.empty else 0

        forecast_metric_1, forecast_metric_2, forecast_metric_3, forecast_metric_4 = st.columns(4)
        forecast_metric_1.metric("Total Revenue", format_currency(kpis.total_sales))
        forecast_metric_2.metric("Projected Revenue", format_currency(projected_total), delta=f"{growth_pct:+.1f}% vs recent trend")
        forecast_metric_3.metric("Best Forecasting Method", comparison_result.best_model)
        forecast_metric_4.metric("Prediction Accuracy", f"{comparison_result.best_accuracy:.1f}%")

        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>", unsafe_allow_html=True)

        st.markdown("#### Which Forecasting Method Works Best?")
        comparison_table = comparison_result.comparison_df.copy()

        accuracy_chart = px.bar(
            comparison_table,
            x="Model",
            y="Accuracy (%)",
            color="Model",
            text="Accuracy (%)",
            color_discrete_sequence=["#2563EB", "#16A34A", "#F59E0B"],
            title="Prediction Accuracy by Method (higher = better)",
        )
        accuracy_chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        accuracy_chart.update_layout(**_chart_layout(340, showlegend=False))

        error_chart = px.bar(
            comparison_table,
            x="Model",
            y="Avg. Prediction Error ($)",
            color="Model",
            text="Avg. Prediction Error ($)",
            color_discrete_sequence=["#2563EB", "#16A34A", "#F59E0B"],
            title="Average Prediction Error by Method (lower = better)",
        )
        error_chart.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        error_chart.update_layout(**_chart_layout(340, showlegend=False))

        comparison_left, comparison_right = st.columns(2)
        with comparison_left:
            st.plotly_chart(accuracy_chart, use_container_width=True)
        with comparison_right:
            st.plotly_chart(error_chart, use_container_width=True)

        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        st.success(f"Most accurate method: **{comparison_result.best_model}** — used for the forecast below.")

        st.markdown("<hr style='border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0'>", unsafe_allow_html=True)

        st.markdown("#### Sales Forecast")
        forecast_chart = go.Figure()
        forecast_chart.add_trace(
            go.Scatter(
                x=history_frame["order_date"],
                y=history_frame["sales"],
                mode="lines",
                name="Actual Sales",
                line=dict(color="#1D4ED8", width=2.5, shape="spline", smoothing=0.8),
            )
        )
        forecast_chart.add_trace(
            go.Scatter(
                x=history_frame["order_date"],
                y=history_frame["predicted_sales"],
                mode="lines",
                name="Predicted (Training Data)",
                line=dict(color="#10B981", width=2, dash="dot", shape="spline", smoothing=0.8),
            )
        )
        forecast_chart.add_trace(
            go.Scatter(
                x=future_frame["order_date"],
                y=future_frame["predicted_sales"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#7C3AED", width=3, shape="spline", smoothing=0.8),
                marker=dict(size=8, color="#7C3AED"),
            )
        )
        forecast_chart.update_layout(**_chart_layout(460, 20, legend_title_text=""))
        st.plotly_chart(forecast_chart, use_container_width=True)

        actual_monthly = (
            filtered_df.groupby(pd.Grouper(key="order_date", freq="MS"))
            .agg(actual_sales=("sales", "sum"))
            .reset_index()
        )
        forecast_table = future_frame[["order_date", "predicted_sales"]].copy()
        forecast_table = forecast_table.merge(actual_monthly, on="order_date", how="left")
        forecast_table["order_date"] = forecast_table["order_date"].dt.strftime("%b %Y")
        forecast_table["predicted_sales"] = forecast_table["predicted_sales"].map(lambda v: f"${v:,.2f}")
        forecast_table["actual_sales"] = forecast_table["actual_sales"].apply(
            lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
        )
        forecast_table.rename(
            columns={"order_date": "Forecast Month", "predicted_sales": "Predicted Sales", "actual_sales": "Actual Sales"},
            inplace=True,
        )
        forecast_table = forecast_table[["Forecast Month", "Actual Sales", "Predicted Sales"]]
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

with insights_tab:
    st.subheader("Automated Business Insights")
    st.caption("AI-derived plain-language insights from your filtered sales data.")

    business_insights = generate_business_insights(filtered_df)
    for i, insight in enumerate(business_insights, start=1):
        st.markdown(
            f"""
            <div style="background:#F0F9FF;border-left:4px solid #2563EB;padding:0.75rem 1rem;
                        border-radius:0.5rem;margin-bottom:0.6rem;">
                <span style="font-size:0.85rem;color:#64748B;font-weight:600;">INSIGHT {i:02d}</span><br/>
                {insight}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "**Business Questions Answered:** revenue trends, top products, profitable regions, customer segmentation, model-compared forecasting, and automated intelligence."
)
