# Retail Demand Forecasting System

## 1. Simple Overview
This project provides a comprehensive demand forecasting solution designed to predict retail sales for over 1,000 stores up to 6 weeks in advance. It leverages historical sales data, store metadata, and temporal features to build a robust ensemble of machine learning models. The system is designed to handle missing data challenges and differing store counts between training and production environments, ensuring stable and reliable predictions for business planning.

## 2. Tools Used and Rationale
- **XGBoost and LightGBM**: These gradient boosting frameworks were selected for their state-of-the-art performance on tabular data and their ability to capture complex, non-linear relationships without requiring extensive feature scaling.
- **Scikit-learn**: Used for traditional machine learning baselines (Linear Regression), advanced neural networks (MLPRegressor), and mandatory preprocessing steps like StandardScaler to ensure distance-sensitive models function correctly.
- **Prophet**: Utilized to capture strong seasonal patterns and holiday effects specifically tailored for time series data.
- **Pandas and Numpy**: The backbone for high-performance data manipulation, feature engineering, and handling large-scale datasets.
- **Plotly and Seaborn**: Chosen to create both interactive and static visualizations that help translate raw model numbers into clear business trends.

## 3. Achievements and Model Performance
The system successfully resolved a critical forecast discrepancy issue where initial models under-predicted sales by 76%. By implementing "Horizon-Safe" historical averages and store-normalized diagnostics, the system now delivers highly accurate forecasts across five different architectures.

### Model Accuracy (Mean Absolute Percentage Error)
- **XGBoost**: 11.00% (89% accuracy)
- **LightGBM**: 11.24%
- **Neural Network (MLP)**: 11.40%
- **Linear Regression**: 14.98%
- **Prophet**: 23.90% (Calculated on daily aggregate)

The top-performing models (XGBoost and LightGBM) provide extremely reliable signals for both store-level and company-wide planning.

## 4. Business Value (Rossmann Case Study)
Businesses like Rossmann gain several strategic advantages from this system:
- **Inventory Optimization**: Accurate 6-week forecasts allow for precision stock management, reducing both stockouts during peak periods and wastage during low-demand months.
- **Resource Allocation**: Identifying high-demand days (e.g., Mondays and Fridays) allows management to optimize staffing schedules and operational hours.
- **Promotional Planning**: The system quantifies the impact of promotions and growth trends, allowing the marketing department to target declining areas with specific campaigns.
- **Risk Mitigation**: Store-normalized diagnostics allow for "apples-to-apples" performance comparisons even when new stores are added or existing ones are temporarily closed.

## 5. Additional Key Implementation Details
- **Cyclical Time Encoding**: Used Sine and Cosine transformations for months and days to ensure the model understands that December is temporally close to January.
- **Standardized Scaling**: Implemented automated feature scaling to allow seamless integration of traditional statistical models and localized neural networks.
- **Diagnostic Normalization**: Built a custom verification engine that accounts for store count variances, ensuring that "Performance Ratios" are scientifically valid.
- **Windows Optimized Environment**: The system is fully tested and optimized for Windows environments, featuring clean logging systems that avoid common unicode and encoding errors found in other data science pipelines.
