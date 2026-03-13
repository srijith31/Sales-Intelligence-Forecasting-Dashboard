# Sales Data Analytics Dashboard <div align="center">
  
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://salesdashboard-mrre4qkxgkhqhtnx2iwr3m.streamlit.app/#sales-forecast
)

</div>

A product-grade Streamlit dashboard for sales analytics, business intelligence, ML-powered forecasting, and automated insights.

Designed for **Indian e-commerce datasets**, but works with **any structured sales CSV**.

---
 

## Key Features
...
Your content is already strong, but it is **too long and slightly repetitive** for a professional **GitHub README**. For portfolios, internships, and recruiters, the README should be:

* Clean
* Structured
* Non-repetitive
* Easy to scan in 30 seconds

I formatted it into a **professional, recruiter-ready README** below.

---

#  Sales Data Analytics Dashboard

A **product-grade Streamlit dashboard** for sales analytics, business intelligence, ML-powered forecasting, and automated insights.

Designed for **Indian e-commerce datasets**, but works with **any structured sales CSV**.

---

#  Key Features

## Core Analytics

* KPI cards: **Total Sales, Profit, Orders, Average Order Value**
* Sales trend chart with **Daily / Weekly / Monthly views**
* Secondary metrics:

  * Total Sales Loss
  * Average Sales
  * Profit Rate
  * Loss Rate

---

#  Products Analytics

### Product Performance

* **Top Products by Units × Profit**
* **Highest Loss Products**
* **Category-wise Sales Distribution**
* **Product Demand Scatter Trend**

### Custom Chart Builder

Interactive chart generator with **12 chart types**

| Chart          | Description                   |
| -------------- | ----------------------------- |
| Bar Chart      | Ranked vertical bars          |
| Horizontal Bar | Better for long product names |
| Pie Chart      | Category share                |
| Donut Chart    | Pie with center hole          |
| Treemap        | Category → product hierarchy  |
| Funnel         | Step-wise metric comparison   |
| Scatter        | Two-metric comparison         |
| Bubble         | Scatter with size metric      |
| Line           | Product/category trends       |
| Area           | Filled trend visualization    |
| Heatmap        | Category × Region grid        |
| Box Plot       | Distribution across orders    |

**Controls**

* Chart Type
* Metric (Sales / Profit / Units)
* Group By (Product / Category)
* Top N (3–30)

---

# Regional Performance

* Regional sales comparison
* Profit margin analysis
* Region × Category **heatmap**
* Automatic **best / weakest region detection**

---

# Customer Intelligence

Customer behavior analytics including:

* **Top Customers by Revenue**
* Order frequency distribution
* **K-Means Customer Segmentation**

Customer segments:

| Segment      | Description                |
| ------------ | -------------------------- |
| High Value   | Large revenue contribution |
| Medium Value | Moderate engagement        |
| Low Value    | Low purchase frequency     |

Segmentation uses:

* Total Revenue
* Total Profit
* Number of Orders
* Quantity Purchased

---

#  Sales Forecasting

Three ML models are compared automatically.

| Model             | Description                   |
| ----------------- | ----------------------------- |
| Linear Regression | Baseline trend forecast       |
| Random Forest     | Pattern-based ensemble model  |
| XGBoost           | Boosted structured data model |

### Features Used

* Time index
* Year / Quarter / Month
* Seasonality (sin / cos encoding)
* Quantity sold
* Lag-1 and Lag-2 features
* 3-month rolling average

### Forecast Outputs

* Projected Revenue
* Prediction Accuracy
* Best Model Selection
* Historical vs Predicted Chart
* Forecast Table

Accuracy formula:

```
Accuracy = max(0, min(100, (1 − MAE / mean) × 100))
```

---

# 🧠 Automated Business Insights

The dashboard generates **7 plain-language insights** automatically.

Examples include:

* Top revenue category and its contribution
* Highest profit-margin region
* Repeat buyer analysis
* Best-selling product performance
* Peak vs slowest sales month
* Year-over-year revenue change
* Strongest and weakest category margins

Insights update dynamically based on **active filters**.

---

#  Export Features

Every section supports **CSV and Excel export**.

| Location      | Export                |
| ------------- | --------------------- |
| Sidebar       | Full filtered dataset |
| Products Tab  | Product summary       |
| Regions Tab   | Regional summary      |
| Customers Tab | Customer performance  |
| Forecast Tab  | Forecast predictions  |

Exports always respect the **active filters**.

---

#  Dashboard Filters

Located in the sidebar:

* Date Range
* Region Selection
* Category Selection
* Product Selection
* Forecast Horizon (3–12 months)

All charts update **in real-time**.

---

#Tech Stack

| Technology   | Purpose                  |
| ------------ | ------------------------ |
| Streamlit    | Dashboard UI             |
| Pandas       | Data processing          |
| Plotly       | Interactive charts       |
| Scikit-learn | ML models & segmentation |
| XGBoost      | Forecasting model        |
| NumPy        | Numerical computation    |
| OpenPyXL     | Excel export             |

---

#Project Structure

```
sales-dashboard
│
├── app.py
│
├── src
│   ├── analytics.py
│   ├── forecasting.py
│   ├── sales.csv
│
├── requirements.txt
```

| File           | Purpose                       |
| -------------- | ----------------------------- |
| app.py         | Main Streamlit dashboard      |
| analytics.py   | KPIs, summaries, segmentation |
| forecasting.py | Forecast models               |
| sales.csv      | Example dataset               |

---

# Dataset

Example dataset contains **182 e-commerce orders** across Indian cities.

### Cities

Mumbai, Bangalore, Delhi, Chennai, Hyderabad, Pune, Ahmedabad, Kolkata, Jaipur

### Categories

Smartphones, Laptops, Televisions, Home Appliances, Audio, Gaming, Cameras, Refrigerators

### Time Period

**Jan 2024 – Dec 2024**

---

# Required CSV Columns

| Column        | Type              |
| ------------- | ----------------- |
| order_id      | string            |
| order_date    | date              |
| region        | string            |
| category      | string            |
| product       | string            |
| sales         | float             |
| profit        | float             |
| quantity      | int               |
| customer_name | string (optional) |

---

#  Installation

###  Clone Repository

```bash
git clone https://github.com/yourusername/sales-dashboard.git
cd sales-dashboard
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Mac/Linux

```
source .venv/bin/activate
```

Windows

```
.venv\Scripts\activate
```

###  Install Dependencies

```
pip install -r requirements.txt
```

###  Run Dashboard

```
streamlit run app.py
```

---

# Real-World Applications

* Retail sales analytics
* E-commerce performance monitoring
* Customer segmentation
* Inventory demand forecasting
* Regional sales strategy
* Executive business reporting

---
