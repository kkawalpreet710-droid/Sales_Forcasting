# Sales Forecasting & Demand Intelligence System

End-to-end sales forecasting and demand intelligence system built on the Superstore Sales dataset — covering time series analysis, forecasting (SARIMA, Prophet, XGBoost), anomaly detection, product demand segmentation, and a deployed interactive dashboard.

Built as a 4-week data science internship capstone project.

---

## 🔗 Live Links

- **Live Dashboard (Streamlit):**   https://salesforcasting-v6zha4ezbtinksxmhggnrr.streamlit.app/
- **Notebook (Google Colab):** https://colab.research.google.com/drive/1OdiyXNlccWzYhF5iRhwgwxj-Ylb-7h2P?usp=sharing

---

## 📁 Repository Structure

```
SalesForecasting_[YourName]/
│
├── analysis.ipynb          # Full analysis notebook (Tasks 1–6), with markdown explanations
├── train.csv                # Superstore Sales dataset (source data)
├── app.py                   # Streamlit dashboard (Task 7)
├── requirements.txt          # Python dependencies for the dashboard
├── summary.docx / summary.pdf   # Executive business report (Task 8)
├── charts/                   # Exported chart images (.png) from the notebook
│
├── dashboard_data.csv        # Cleaned transaction data (feeds the dashboard)
├── monthly_sales.csv         # Aggregated monthly sales totals
├── anomaly_data.csv          # Weekly sales + anomaly flags (Isolation Forest & Z-Score)
├── cluster_data.csv          # Product sub-category clusters + PCA coordinates
├── prophet_forecast.csv      # Overall 3-month Prophet forecast
├── segment_forecasts.csv     # Category/region-level 3-month forecasts
├── segment_metrics.csv       # Category/region-level model accuracy (MAE/RMSE/MAPE)
│
└── README.md                 # This file
```

---

## 🎯 Project Overview

The goal: predict future product demand, detect unusual sales patterns, segment products by demand behavior, and present it all through a dashboard a business manager could actually use on a Monday morning.

**Dataset:** [Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) — 9,800 transactions across 4 years (2015–2018), multiple regions, categories, and sub-categories. A secondary dataset ([Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)) was used for multi-source merging practice in Task 5.

---

## ✅ What Was Built

| Task | What it covers |
|---|---|
| 1. EDA | Data cleaning, time feature engineering, weekly/monthly aggregation, key business questions answered with data |
| 2. Decomposition | Trend/seasonal/residual breakdown, ADF stationarity test |
| 3. Forecasting | SARIMA, Prophet, and XGBoost compared on MAE/RMSE/MAPE; Prophet selected for production |
| 4. Segment Forecasting | Category and region-level 3-month forecasts, each independently evaluated |
| 5. Anomaly Detection | Isolation Forest + Z-Score methods compared, with real-world explanations for top anomalies |
| 6. Clustering | K-Means (k=4, via Elbow Method) segmenting products by demand behavior, visualized with PCA |
| 7. Dashboard | 4-page Streamlit app: Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments |
| 8. Business Report | 2-page executive summary for non-technical stakeholders |

**Best model:** Prophet (MAPE 16.08% on the overall series) — outperformed SARIMA and XGBoost, primarily due to greater robustness on a relatively short (4-year) historical dataset.

---

## 🛠 Tech Stack

- **Python 3.x** — Pandas, NumPy for data processing
- **Statsmodels** — SARIMA, seasonal decomposition, ADF test
- **Prophet** — production forecasting model
- **XGBoost** — ML-based forecasting via lag features
- **Scikit-learn** — Isolation Forest, K-Means, PCA
- **Plotly / Matplotlib** — visualization
- **Streamlit** — interactive dashboard deployment

---

## ▶️ Running the Dashboard Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects the following CSV files in the same directory as `app.py`: `dashboard_data.csv`, `monthly_sales.csv`, `anomaly_data.csv`, `cluster_data.csv`, `prophet_forecast.csv`, `segment_forecasts.csv`, `segment_metrics.csv`.

---

## ⚠️ Known Limitations

- **Short training history (4 years / ~4 seasonal cycles)** limits confidence in seasonal parameter estimation, particularly for SARIMA — documented in detail in `analysis.ipynb` (Task 3).
- **East region forecasts** are notably less reliable than other segments (MAPE ~50%) — flagged in both the dashboard and the executive report as needing wider risk buffers.
- **Video game dataset (Task 5)** has incomplete data past 2016 and was used only for merging-mechanics practice, not for drawing real business conclusions.

---

## 📓 Author's Notes

This project was built iteratively with deliberate documentation of blockers, wrong turns, and fixes along the way — see the markdown cells throughout `analysis.ipynb` for the full reasoning behind key decisions (parameter choices, model tuning, and a few honest dead ends).
