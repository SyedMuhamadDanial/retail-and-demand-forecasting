# Demand & Time Series Forecasting - 6 Week Sales Prediction

A comprehensive demand forecasting system that predicts sales up to **6 weeks in advance** using multiple machine learning models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the Jupyter Notebook
```bash
jupyter notebook demand_forecasting_analysis.ipynb
```

Or use JupyterLab:
```bash
jupyter lab demand_forecasting_analysis.ipynb
```

### 3. Run All Cells
- Click **Cells** → **Run All**
- Or press `Shift + Enter` to run cells one by one

## 📊 What's Included

### Main Notebook
**`demand_forecasting_analysis.ipynb`** - Complete analysis with:
- ✅ Data Exploration & Visualization
- ✅ Feature Engineering (lag features, rolling stats, time features)
- ✅ 3 Forecasting Models: Prophet, XGBoost, LightGBM
- ✅ Model Performance Comparison
- ✅ 6-Week Sales Forecasts
- ✅ Business Insights & Recommendations
- ✅ Clean Tables & Interactive Charts

### Datasets
- `train.csv` - Historical sales data (38MB)
- `test.csv` - Test period for 6-week predictions (1.4MB)
- `store.csv` - Store metadata (45KB)

### Configuration
- `config.py` - Model hyperparameters and settings
- `requirements.txt` - All dependencies

## 📈 Models & Features

### Models
1. **Prophet** - Facebook's time series tool (handles seasonality)
2. **XGBoost** - Gradient boosting with lag features
3. **LightGBM** - Fast gradient boosting

### Features Created
- Time features (day, month, year, day of week, quarter)
- Lag features (1-28 days)
- Rolling averages (7, 14, 28 days)
- Cyclical encodings (sin/cos for seasonality)
- Weekend/holiday flags

## 📋 Output Files

Results are saved to `outputs/`:
- `6_week_forecast.csv` - Daily predictions
- `weekly_forecast_summary.csv` - Weekly aggregates
- `model_comparison.csv` - Model performance metrics

## 🎯 Performance Metrics

All models evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## 📖 For More Details

See the complete walkthrough in the artifacts folder or run the notebook to see your specific results!

## 🔧 Troubleshooting

**Jupyter not found?**
```bash
pip install notebook jupyterlab
```

**Memory issues?**
- Reduce `n_estimators` in `config.py`
- Use LightGBM (more memory efficient)

**Prophet installation issues?**
```bash
pip install prophet==1.1.0
```

---

**Ready to forecast?** Open the notebook and run all cells! 🚀
