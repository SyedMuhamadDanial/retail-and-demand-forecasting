"""
Demand & Time Series Forecasting - Standalone Script
Predicts sales 6 weeks in advance using multiple ML models
Run this script directly without needing Jupyter: python run_forecasting.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os

# Modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Constants
FORECAST_WEEKS = 6
RANDOM_STATE = 42

print("="*80)
print("DEMAND FORECASTING SYSTEM - 6 WEEK SALES PREDICTION")
print("="*80)
print()

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Step 1/7: Loading datasets...")
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)
store = pd.read_csv('store.csv')

print(f"  ✓ Train: {train.shape}")
print(f"  ✓ Test: {test.shape}")
print(f"  ✓ Store: {store.shape}")

# ============================================================================
# 2. IDENTIFY KEY COLUMNS
# ============================================================================
print("\nStep 2/7: Identifying key columns...")
date_cols = [col for col in train.columns if 'date' in col.lower()]
sales_cols = [col for col in train.columns if 'sales' in col.lower() or 'sale' in col.lower()]
store_cols = [col for col in train.columns if 'store' in col.lower()]

DATE_COL = date_cols[0] if date_cols else 'Date'
TARGET_COL = sales_cols[0] if sales_cols else 'Sales'
STORE_COL = store_cols[0] if store_cols else 'Store'

print(f"  Date column: {DATE_COL}")
print(f"  Target column: {TARGET_COL}")
print(f"  Store column: {STORE_COL}")

# ============================================================================
# 3. PREPROCESS DATA
# ============================================================================
print("\nStep 3/7: Preprocessing data...")

# Convert dates
train[DATE_COL] = pd.to_datetime(train[DATE_COL])
test[DATE_COL] = pd.to_datetime(test[DATE_COL])

# Sort by date
train = train.sort_values(DATE_COL).reset_index(drop=True)
test = test.sort_values(DATE_COL).reset_index(drop=True)

# Merge with store data
if STORE_COL in train.columns and STORE_COL in store.columns:
    train = train.merge(store, on=STORE_COL, how='left')
    test = test.merge(store, on=STORE_COL, how='left')

# Remove missing/negative sales
train_clean = train[train[TARGET_COL].notna() & (train[TARGET_COL] >= 0)].copy()

print(f"  ✓ Date range: {train_clean[DATE_COL].min()} to {train_clean[DATE_COL].max()}")
print(f"  ✓ Clean records: {len(train_clean):,}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\nStep 4/7: Creating time series features...")

def create_time_features(df, date_col):
    """Create time-based features"""
    df = df.copy()
    
    # Basic time features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week
    df['quarter'] = df[date_col].dt.quarter
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Weekend flag
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    return df

def create_lag_features(df, target_col, store_col, date_col, lags=[1,2,3,7,14,21,28]):
    """Create lag and rolling features"""
    df = df.copy()
    df = df.sort_values([store_col, date_col])
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(store_col)[target_col].shift(lag)
    
    # Rolling features
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby(store_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby(store_col)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    return df

# Apply feature engineering
train_clean = create_time_features(train_clean, DATE_COL)
test = create_time_features(test, DATE_COL)

if STORE_COL in train_clean.columns:
    train_clean = create_lag_features(train_clean, TARGET_COL, STORE_COL, DATE_COL)

print(f"  ✓ Created {len([c for c in train_clean.columns if c.startswith('lag_')])} lag features")
print(f"  ✓ Created {len([c for c in train_clean.columns if c.startswith('rolling_')])} rolling features")
print(f"  ✓ Total features: {train_clean.shape[1]}")

# ============================================================================
# 5. PREPARE TRAIN/VALIDATION SPLIT
# ============================================================================
print("\nStep 5/7: Preparing train/validation split...")

max_date = train_clean[DATE_COL].max()
validation_start = max_date - timedelta(weeks=6)

train_data = train_clean[train_clean[DATE_COL] < validation_start].copy()
valid_data = train_clean[train_clean[DATE_COL] >= validation_start].copy()

# Prepare features for ML
feature_cols = [col for col in train_data.columns if col not in 
                [DATE_COL, TARGET_COL, 'Id', 'id', 'ID'] and 
                train_data[col].dtype in ['int64', 'float64']]

train_ml = train_data.dropna(subset=feature_cols + [TARGET_COL])
valid_ml = valid_data.dropna(subset=feature_cols + [TARGET_COL])

X_train = train_ml[feature_cols]
y_train = train_ml[TARGET_COL]
X_valid = valid_ml[feature_cols]
y_valid = valid_ml[TARGET_COL]

print(f"  Training: {len(X_train):,} samples")
print(f"  Validation: {len(X_valid):,} samples")
print(f"  Features: {len(feature_cols)}")

# ============================================================================
# 6. TRAIN MODELS
# ============================================================================
print("\nStep 6/7: Training forecasting models...")

# Try XGBoost if available
try:
    import xgboost as xgb
    print("  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_valid)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_pred))
    xgb_mae = mean_absolute_error(y_valid, xgb_pred)
    xgb_mape = np.mean(np.abs((y_valid - xgb_pred) / y_valid)) * 100
    
    print(f"    ✓ RMSE: {xgb_rmse:,.2f}, MAE: {xgb_mae:,.2f}, MAPE: {xgb_mape:.2f}%")
    best_model = xgb_model
    best_model_name = "XGBoost"
    best_mape = xgb_mape
    
except ImportError:
    print("  XGBoost not available, using Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_valid)
    
    rf_rmse = np.sqrt(mean_squared_error(y_valid, rf_pred))
    rf_mae = mean_absolute_error(y_valid, rf_pred)
    rf_mape = np.mean(np.abs((y_valid - rf_pred) / y_valid)) * 100
    
    print(f"    ✓ RMSE: {rf_rmse:,.2f}, MAE: {rf_mae:,.2f}, MAPE: {rf_mape:.2f}%")
    best_model = rf_model
    best_model_name = "Random Forest"
    best_mape = rf_mape

# ============================================================================
# 7. GENERATE 6-WEEK FORECAST
# ============================================================================
print(f"\nStep 7/7: Generating 6-week forecast using {best_model_name}...")

# Prepare test data
full_data = pd.concat([train_clean, test], ignore_index=True)
if STORE_COL in full_data.columns:
    full_data = create_lag_features(full_data, TARGET_COL, STORE_COL, DATE_COL)

test_dates = test[DATE_COL].unique()
test_ml = full_data[full_data[DATE_COL].isin(test_dates)].copy()

# Prepare features (fill NaN with 0)
X_test = test_ml[feature_cols].fillna(0)

# Predict
test_predictions = best_model.predict(X_test)
test_ml['Predicted_Sales'] = test_predictions

# Aggregate by date
final_forecast = test_ml.groupby(DATE_COL)['Predicted_Sales'].sum().reset_index()
final_forecast.columns = ['Date', 'Predicted_Sales']
final_forecast = final_forecast.sort_values('Date').reset_index(drop=True)
final_forecast['Week'] = (final_forecast.index // 7) + 1

# Weekly summary
weekly_forecast = final_forecast.groupby('Week')['Predicted_Sales'].agg([
    ('Total_Sales', 'sum'),
    ('Avg_Daily_Sales', 'mean'),
    ('Min_Daily', 'min'),
    ('Max_Daily', 'max')
]).reset_index()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\nSaving results...")
final_forecast.to_csv('outputs/6_week_forecast.csv', index=False)
weekly_forecast.to_csv('outputs/weekly_forecast_summary.csv', index=False)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "="*80)
print("6-WEEK FORECAST SUMMARY")
print("="*80)
print(f"\nModel Used: {best_model_name}")
print(f"Prediction Accuracy (MAPE): {100 - best_mape:.2f}%")
print(f"Forecast Period: {final_forecast['Date'].min()} to {final_forecast['Date'].max()}")
print(f"\nTotal Forecasted Sales (6 weeks): {final_forecast['Predicted_Sales'].sum():,.0f}")
print(f"Average Daily Sales: {final_forecast['Predicted_Sales'].mean():,.0f}")

print("\nWeekly Breakdown:")
print("-" * 80)
for _, row in weekly_forecast.iterrows():
    print(f"Week {int(row['Week'])}: {row['Total_Sales']:>12,.0f} total  |  "
          f"{row['Avg_Daily_Sales']:>10,.0f} avg/day  |  "
          f"Range: {row['Min_Daily']:,.0f} - {row['Max_Daily']:,.0f}")

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print("✓ outputs/6_week_forecast.csv")
print("✓ outputs/weekly_forecast_summary.csv")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print(f"\n📦 Inventory: Stock up for weeks {weekly_forecast.nlargest(2, 'Total_Sales')['Week'].values}")
print(f"📈 Peak Demand: {weekly_forecast['Total_Sales'].max():,.0f} units in week {weekly_forecast.loc[weekly_forecast['Total_Sales'].idxmax(), 'Week']:.0f}")
print(f"🎯 Model Accuracy: {100-best_mape:.1f}% (MAPE: {best_mape:.2f}%)")

print("\n" + "="*80)
print("✅ FORECASTING COMPLETE!")
print("="*80)
