from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class KPIBundle:
    total_sales: float
    total_profit: float
    total_orders: int
    average_order_value: float


def calculate_kpis(df: pd.DataFrame) -> KPIBundle:
    total_sales = float(df["sales"].sum())
    total_profit = float(df["profit"].sum())
    total_orders = int(df["order_id"].nunique())
    average_order_value = total_sales / total_orders if total_orders else 0.0
    return KPIBundle(
        total_sales=total_sales,
        total_profit=total_profit,
        total_orders=total_orders,
        average_order_value=average_order_value,
    )


def summarize_data_quality(df: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "Rows": int(len(df)),
        "Duplicate rows": int(df.duplicated().sum()),
        "Missing sales": int(df["sales"].isna().sum()),
        "Missing dates": int(df["order_date"].isna().sum()),
        "Date range": f"{df['order_date'].min():%Y-%m-%d} to {df['order_date'].max():%Y-%m-%d}",
    }


def get_sales_timeseries(df: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    freq = {"D": "D", "W": "W", "M": "MS"}.get(frequency, "MS")
    timeline = (
        df.groupby(pd.Grouper(key="order_date", freq=freq))
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )
    return timeline.dropna(subset=["order_date"])


def get_product_summary(df: pd.DataFrame, top_n: int = 10, ascending: bool = False) -> pd.DataFrame:
    product_summary = (
        df.groupby(["product", "category"], as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        .sort_values("sales", ascending=ascending)
    )
    return product_summary.head(top_n)


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category", as_index=False)
        .agg(sales=("sales", "sum"), profit=("profit", "sum"), quantity=("quantity", "sum"))
        .sort_values("sales", ascending=False)
    )


def get_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("region", as_index=False)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique"),
        )
        .sort_values("sales", ascending=False)
    )
    summary["profit_margin"] = summary["profit"] / summary["sales"].where(summary["sales"] != 0, 1)
    return summary


def get_customer_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    summary = (
        df.groupby("customer_name", as_index=False)
        .agg(
            sales=("sales", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique"),
            quantity=("quantity", "sum"),
        )
        .sort_values(["sales", "orders"], ascending=False)
    )
    return summary.head(top_n)


def get_customer_frequency(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("customer_name", as_index=False)
        .agg(orders=("order_id", "nunique"), sales=("sales", "sum"))
        .sort_values("orders", ascending=False)
    )


def get_region_category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="region",
            columns="category",
            values="sales",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    customer_df = (
        df.groupby("customer_name", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_orders=("order_id", "nunique"),
            total_quantity=("quantity", "sum"),
        )
    )

    if len(customer_df) < 3:
        customer_df["segment"] = "Medium Value"
        return customer_df

    features = customer_df[["total_sales", "total_profit", "total_orders", "total_quantity"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_df = customer_df.copy()
    customer_df["cluster"] = kmeans.fit_predict(scaled)

    cluster_mean_sales = customer_df.groupby("cluster")["total_sales"].mean().sort_values()
    segment_map = {
        cluster_mean_sales.index[0]: "Low Value",
        cluster_mean_sales.index[1]: "Medium Value",
        cluster_mean_sales.index[2]: "High Value",
    }
    customer_df["segment"] = customer_df["cluster"].map(segment_map)
    return customer_df.drop(columns=["cluster"])


def generate_business_insights(df: pd.DataFrame) -> list[str]:
    def strong(text: str) -> str:
        return f"<strong>{text}</strong>"

    insights: list[str] = []
    total_sales = float(df["sales"].sum())
    if total_sales == 0:
        return ["Insufficient data to generate insights."]

    cat_summary = get_category_summary(df)
    if not cat_summary.empty:
        top_cat = cat_summary.iloc[0]
        pct = top_cat["sales"] / total_sales * 100
        insights.append(
            f"{strong(top_cat['category'])} is the top revenue category, contributing {strong(f'{pct:.1f}%')} of total sales."
        )

    region_summary = get_region_summary(df)
    if not region_summary.empty:
        best_margin = region_summary.sort_values("profit_margin", ascending=False).iloc[0]
        insights.append(
            f"{strong(best_margin['region'])} region leads in profit margin at {strong(f'{best_margin['profit_margin'] * 100:.1f}%')}."
        )

    freq_df = get_customer_frequency(df)
    if not freq_df.empty:
        total_customers = len(freq_df)
        repeat_customers = int((freq_df["orders"] > 1).sum())
        repeat_pct = repeat_customers / total_customers * 100 if total_customers else 0
        avg_orders_repeat = freq_df.loc[freq_df["orders"] > 1, "orders"].mean() if repeat_customers else 0
        insights.append(
            f"{strong(f'{repeat_pct:.0f}%')} of customers are repeat buyers "
            f"({strong(str(repeat_customers))} of {strong(str(total_customers))}), "
            f"averaging {strong(f'{avg_orders_repeat:.1f}')} orders each."
        )

    prod_summary = get_product_summary(df, top_n=1)
    if not prod_summary.empty:
        top_prod = prod_summary.iloc[0]
        prod_pct = top_prod["sales"] / total_sales * 100
        insights.append(
            f"Best-selling product {strong(top_prod['product'])} accounts for {strong(f'{prod_pct:.1f}%')} of total revenue "
            f"with {strong(f"${top_prod['profit']:,.0f}")} in profit."
        )

    monthly = get_sales_timeseries(df, frequency="M")
    if not monthly.empty:
        peak = monthly.loc[monthly["sales"].idxmax()]
        trough = monthly.loc[monthly["sales"].idxmin()]
        insights.append(
            f"Peak revenue month was {strong(peak['order_date'].strftime('%B %Y'))} "
            f"with {strong(f"${peak['sales']:,.0f}")} in sales; "
            f"that was {strong(f"{peak['sales'] / trough['sales']:.1f}x")} the slowest month ({trough['order_date'].strftime('%B %Y')})."
        )

    if not monthly.empty:
        yearly_sales = monthly.assign(year=monthly["order_date"].dt.year).groupby("year")["sales"].sum().sort_index()
        if len(yearly_sales) >= 2:
            latest_year = int(yearly_sales.index[-1])
            prior_year = int(yearly_sales.index[-2])
            latest_sales = float(yearly_sales.iloc[-1])
            prior_sales = float(yearly_sales.iloc[-2])
            yoy_pct = (latest_sales - prior_sales) / prior_sales * 100 if prior_sales else 0
            direction = "grew" if yoy_pct >= 0 else "declined"
            insights.append(
                f"Revenue {strong(direction)} {strong(f'{abs(yoy_pct):.1f}%')} year-over-year from {strong(str(prior_year))} "
                f"({strong(f'${prior_sales:,.0f}')}) to {strong(str(latest_year))} ({strong(f'${latest_sales:,.0f}')})."
            )

    overall_margin = df["profit"].sum() / total_sales * 100
    if not cat_summary.empty:
        category_margins = cat_summary.copy()
        category_margins["margin"] = category_margins["profit"] / category_margins["sales"].where(category_margins["sales"] != 0, 1) * 100
        best_cat = category_margins.sort_values("margin", ascending=False).iloc[0]
        worst_cat = category_margins.sort_values("margin").iloc[0]
        insights.append(
            f"Overall profit margin is {strong(f'{overall_margin:.1f}%')}. "
            f"{strong(best_cat['category'])} has the highest category margin ({strong(f'{best_cat['margin']:.1f}%')}); "
            f"{strong(worst_cat['category'])} the lowest ({strong(f'{worst_cat['margin']:.1f}%')})."
        )
    else:
        insights.append(
            f"Overall blended profit margin across all categories and regions is {strong(f'{overall_margin:.1f}%')}."
        )

    return insights
