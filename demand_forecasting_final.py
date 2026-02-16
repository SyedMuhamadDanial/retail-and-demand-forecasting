# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from datetime import datetime, timedelta



# Modeling

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import LabelEncoder

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.seasonal import seasonal_decompose

from prophet import Prophet

import xgboost as xgb

import lightgbm as lgb



# Visualization

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Settings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

plt.style.use('seaborn-v0_8-darkgrid')

sns.set_palette('husl')



# Constants

FORECAST_WEEKS = 6

RANDOM_STATE = 42



print('✓ All libraries imported successfully')

# Load datasets

print('Loading datasets...')

train = pd.read_csv('train.csv', low_memory=False)

test = pd.read_csv('test.csv', low_memory=False)

store = pd.read_csv('store.csv')



print(f'\nTrain shape: {train.shape}')

print(f'Test shape: {test.shape}')

print(f'Store shape: {store.shape}')



print('\n✓ Data loaded successfully')

# Display dataset samples

print('=' * 80)

print('TRAIN DATASET')

print('=' * 80)

display(train.head())

print(f'\nColumns: {train.columns.tolist()}')

print(f'\nData Types:\n{train.dtypes}')

print(f'\nMissing Values:\n{train.isnull().sum()}')

print('=' * 80)

print('TEST DATASET')

print('=' * 80)

display(test.head())

print(f'\nColumns: {test.columns.tolist()}')

print('=' * 80)

print('STORE DATASET')

print('=' * 80)

display(store.head(10))

print(f'\nColumns: {store.columns.tolist()}')

print(f'\nStore Types: {store.columns.tolist()}')

# Statistical Summary

print('STATISTICAL SUMMARY')

print('=' * 80)

summary_df = train.describe().T

summary_df['missing'] = train.isnull().sum()

summary_df['missing_pct'] = (train.isnull().sum() / len(train) * 100).round(2)

display(summary_df.style.background_gradient(cmap='YlOrRd', subset=['missing_pct']))

# Identify date and target columns

date_cols = [col for col in train.columns if 'date' in col.lower()]

sales_cols = [col for col in train.columns if 'sales' in col.lower() or 'sale' in col.lower()]

store_cols = [col for col in train.columns if 'store' in col.lower()]



print(f'Date columns: {date_cols}')

print(f'Sales columns: {sales_cols}')

print(f'Store columns: {store_cols}')



# Assume first date column is the date, first sales column is target

DATE_COL = date_cols[0] if date_cols else 'Date'

TARGET_COL = sales_cols[0] if sales_cols else 'Sales'

STORE_COL = store_cols[0] if store_cols else 'Store'



print(f'\nUsing:')

print(f'  Date column: {DATE_COL}')

print(f'  Target column: {TARGET_COL}')

print(f'  Store column: {STORE_COL}')

# Convert date column to datetime

train[DATE_COL] = pd.to_datetime(train[DATE_COL])

test[DATE_COL] = pd.to_datetime(test[DATE_COL])



# Sort by date

train = train.sort_values(DATE_COL).reset_index(drop=True)

test = test.sort_values(DATE_COL).reset_index(drop=True)



print(f'Date range (train): {train[DATE_COL].min()} to {train[DATE_COL].max()}')

print(f'Date range (test): {test[DATE_COL].min()} to {test[DATE_COL].max()}')

print(f'\nTotal days in train: {(train[DATE_COL].max() - train[DATE_COL].min()).days}')

print(f'Total days in test: {(test[DATE_COL].max() - test[DATE_COL].min()).days}')

# Merge with store information

if STORE_COL in train.columns and STORE_COL in store.columns:

    train = train.merge(store, on=STORE_COL, how='left')

    test = test.merge(store, on=STORE_COL, how='left')

    print('✓ Merged store information')

else:

    print('! No store column found for merging')



print(f'\nTrain shape after merge: {train.shape}')

print(f'Test shape after merge: {test.shape}')

# Feature Engineering Function

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

    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)

    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)

    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    

    # Weekend flag

    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    

    # Month start/end

    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)

    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    

    return df



# Apply feature engineering

train = create_time_features(train, DATE_COL)

test = create_time_features(test, DATE_COL)



print('✓ Time features created')

print(f'\nNew features: {[col for col in train.columns if col in ["year", "month", "day", "dayofweek", "is_weekend"]]}')

# Handle missing values in target

print(f'Missing values in {TARGET_COL}: {train[TARGET_COL].isnull().sum()}')



# Remove rows where target is missing or negative

train_clean = train[train[TARGET_COL].notna() & (train[TARGET_COL] >= 0)].copy()



print(f'\nRows removed: {len(train) - len(train_clean)}')

print(f'Clean dataset shape: {train_clean.shape}')

# Aggregate sales by date

daily_sales = train_clean.groupby(DATE_COL)[TARGET_COL].sum().reset_index()

daily_sales.columns = ['Date', 'Sales']



# Plot overall trend

fig = go.Figure()

fig.add_trace(go.Scatter(

    x=daily_sales['Date'],

    y=daily_sales['Sales'],

    mode='lines',

    name='Daily Sales',

    line=dict(color='#1f77b4', width=1)

))



# Add 7-day moving average

daily_sales['MA7'] = daily_sales['Sales'].rolling(window=7).mean()

fig.add_trace(go.Scatter(

    x=daily_sales['Date'],

    y=daily_sales['MA7'],

    mode='lines',

    name='7-Day Moving Avg',

    line=dict(color='#ff7f0e', width=2)

))



fig.update_layout(

    title='Overall Sales Trend Over Time',

    xaxis_title='Date',

    yaxis_title='Total Sales',

    height=500,

    hovermode='x unified'

)

fig.show()

# Monthly sales pattern

monthly_sales = train_clean.groupby('month')[TARGET_COL].agg(['mean', 'sum', 'count']).reset_index()

monthly_sales.columns = ['Month', 'Average Sales', 'Total Sales', 'Count']



fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Sales by Month', 'Total Sales by Month'))



fig.add_trace(go.Bar(x=monthly_sales['Month'], y=monthly_sales['Average Sales'], 

                      name='Avg Sales', marker_color='lightblue'), row=1, col=1)

fig.add_trace(go.Bar(x=monthly_sales['Month'], y=monthly_sales['Total Sales'], 

                      name='Total Sales', marker_color='coral'), row=1, col=2)



fig.update_xaxes(title_text='Month', row=1, col=1)

fig.update_xaxes(title_text='Month', row=1, col=2)

fig.update_layout(height=400, showlegend=False)

fig.show()



print('\nMonthly Sales Summary:')

display(monthly_sales.style.background_gradient(cmap='Blues', subset=['Average Sales', 'Total Sales']))

# Day of week pattern

dow_sales = train_clean.groupby('dayofweek')[TARGET_COL].agg(['mean', 'std']).reset_index()

dow_sales.columns = ['DayOfWeek', 'Average Sales', 'Std Dev']

dow_sales['Day'] = dow_sales['DayOfWeek'].map({

    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',

    4: 'Friday', 5: 'Saturday', 6: 'Sunday'

})



fig = go.Figure()

fig.add_trace(go.Bar(

    x=dow_sales['Day'],

    y=dow_sales['Average Sales'],

    error_y=dict(type='data', array=dow_sales['Std Dev']),

    marker_color='mediumseagreen'

))



fig.update_layout(

    title='Average Sales by Day of Week',

    xaxis_title='Day',

    yaxis_title='Average Sales',

    height=400

)

fig.show()



print('\nDay of Week Sales Summary:')

display(dow_sales[['Day', 'Average Sales', 'Std Dev']].style.background_gradient(cmap='Greens'))

# Prepare data for modeling

# Use last 6 weeks of train data as validation

max_date = train_clean[DATE_COL].max()

validation_start = max_date - timedelta(weeks=6)



train_data = train_clean[train_clean[DATE_COL] < validation_start].copy()

valid_data = train_clean[train_clean[DATE_COL] >= validation_start].copy()



print(f'Training data: {len(train_data)} rows ({train_data[DATE_COL].min()} to {train_data[DATE_COL].max()})')

print(f'Validation data: {len(valid_data)} rows ({valid_data[DATE_COL].min()} to {valid_data[DATE_COL].max()})')

print(f'Test data: {len(test)} rows ({test[DATE_COL].min()} to {test[DATE_COL].max()})')

# Create lag features for ML models

def create_lag_features(df, target_col, store_col, date_col, lags=[1,2,3,7,14,21,28]):

    """Create lag and rolling features for time series ML"""

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



# Apply lag features

if STORE_COL in train_clean.columns:

    train_clean_lag = create_lag_features(train_clean, TARGET_COL, STORE_COL, DATE_COL)

    print('✓ Lag features created')

else:

    train_clean_lag = train_clean.copy()

    print('! No store column - skipping lag features')



# Update train/valid split with lag features

train_ml = train_clean_lag[train_clean_lag[DATE_COL] < validation_start].copy()

valid_ml = train_clean_lag[train_clean_lag[DATE_COL] >= validation_start].copy()

# Prepare data for Prophet (needs 'ds' and 'y' columns)

prophet_train = daily_sales[daily_sales['Date'] < validation_start][['Date', 'Sales']].copy()

prophet_train.columns = ['ds', 'y']



# Train Prophet model

print('Training Prophet model...')

prophet_model = Prophet(

    yearly_seasonality=True,

    weekly_seasonality=True,

    daily_seasonality=False,

    seasonality_mode='multiplicative'

)

prophet_model.fit(prophet_train)



# Make predictions ONLY for validation period

validation_days = len(daily_sales[daily_sales['Date'] >= validation_start])

future_dates = prophet_model.make_future_dataframe(periods=validation_days, freq='D')

prophet_forecast = prophet_model.predict(future_dates)



# Extract validation predictions - get the last 'validation_days' predictions

prophet_valid_pred = prophet_forecast['yhat'].tail(validation_days).values

prophet_valid_actual = daily_sales[daily_sales['Date'] >= validation_start]['Sales'].values



# Calculate metrics

prophet_rmse = np.sqrt(mean_squared_error(prophet_valid_actual, prophet_valid_pred))

prophet_mae = mean_absolute_error(prophet_valid_actual, prophet_valid_pred)



# Calculate MAPE (excluding zero sales days)

mask = prophet_valid_actual != 0

prophet_mape = np.mean(np.abs((prophet_valid_actual[mask] - prophet_valid_pred[mask]) / prophet_valid_actual[mask])) * 100



print(f'\n✓ Prophet Model Results:')

print(f'  RMSE: {prophet_rmse:,.2f}')

print(f'  MAE: {prophet_mae:,.2f}')

print(f'  MAPE: {prophet_mape:.2f}%')

# Visualize Prophet forecast

fig = prophet_model.plot(prophet_forecast)

plt.title('Prophet Forecast with Components')

plt.tight_layout()

plt.show()



# Components

fig = prophet_model.plot_components(prophet_forecast)

plt.tight_layout()

plt.show()

# Prepare features for XGBoost

feature_cols = [col for col in train_ml.columns if col not in 

                [DATE_COL, TARGET_COL, 'Id', 'id', 'ID'] and 

                train_ml[col].dtype in ['int64', 'float64']]



# Remove rows with NaN (from lag features)

train_ml_clean = train_ml.dropna(subset=feature_cols + [TARGET_COL])

valid_ml_clean = valid_ml.dropna(subset=feature_cols + [TARGET_COL])



X_train = train_ml_clean[feature_cols]

y_train = train_ml_clean[TARGET_COL]

X_valid = valid_ml_clean[feature_cols]

y_valid = valid_ml_clean[TARGET_COL]



print(f'XGBoost features ({len(feature_cols)}): {feature_cols[:10]}...')

print(f'Training samples: {len(X_train)}')

print(f'Validation samples: {len(X_valid)}')

# Train XGBoost model

print('Training XGBoost model...')

xgb_model = xgb.XGBRegressor(

    n_estimators=500,

    max_depth=7,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    early_stopping_rounds=50

)



xgb_model.fit(

    X_train, y_train,

    eval_set=[(X_valid, y_valid)],

    verbose=False

)



# Predictions

xgb_valid_pred = xgb_model.predict(X_valid)



# Metrics

xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_valid_pred))

xgb_mae = mean_absolute_error(y_valid, xgb_valid_pred)

# Calculate MAPE (excluding zero sales days)

mask = y_valid != 0

xgb_mape = np.mean(np.abs((y_valid[mask] - xgb_valid_pred[mask]) / y_valid[mask])) * 100

xgb_r2 = r2_score(y_valid, xgb_valid_pred)



print(f'\n✓ XGBoost Model Results:')

print(f'  RMSE: {xgb_rmse:,.2f}')

print(f'  MAE: {xgb_mae:,.2f}')

print(f'  MAPE: {xgb_mape:.2f}%')

print(f'  R²: {xgb_r2:.4f}')

# Feature importance

importance_df = pd.DataFrame({

    'feature': feature_cols,

    'importance': xgb_model.feature_importances_

}).sort_values('importance', ascending=False).head(20)



fig = px.bar(importance_df, x='importance', y='feature', orientation='h',

             title='Top 20 XGBoost Feature Importances',

             labels={'importance': 'Importance Score', 'feature': 'Feature'})

fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})

fig.show()



print('\nTop 10 Most Important Features:')

display(importance_df.head(10).style.background_gradient(cmap='Greens', subset=['importance']))

# Train LightGBM model

print('Training LightGBM model...')

lgb_model = lgb.LGBMRegressor(

    n_estimators=500,

    max_depth=7,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    verbose=-1

)



lgb_model.fit(

    X_train, y_train,

    eval_set=[(X_valid, y_valid)],

    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]

)



# Predictions

lgb_valid_pred = lgb_model.predict(X_valid)



# Metrics

lgb_rmse = np.sqrt(mean_squared_error(y_valid, lgb_valid_pred))

lgb_mae = mean_absolute_error(y_valid, lgb_valid_pred)

# Calculate MAPE (excluding zero sales days)

mask = y_valid != 0

lgb_mape = np.mean(np.abs((y_valid[mask] - lgb_valid_pred[mask]) / y_valid[mask])) * 100

lgb_r2 = r2_score(y_valid, lgb_valid_pred)



print(f'\n✓ LightGBM Model Results:')

print(f'  RMSE: {lgb_rmse:,.2f}')

print(f'  MAE: {lgb_mae:,.2f}')

print(f'  MAPE: {lgb_mape:.2f}%')

print(f'  R²: {lgb_r2:.4f}')

# Model comparison table

comparison_df = pd.DataFrame({

    'Model': ['Prophet', 'XGBoost', 'LightGBM'],

    'RMSE': [prophet_rmse, xgb_rmse, lgb_rmse],

    'MAE': [prophet_mae, xgb_mae, lgb_mae],

    'MAPE (%)': [prophet_mape, xgb_mape, lgb_mape]

})



# Add R² for ML models

comparison_df['R²'] = [np.nan, xgb_r2, lgb_r2]



# Sort by RMSE

comparison_df = comparison_df.sort_values('RMSE')



print('\n' + '='*80)

print('MODEL PERFORMANCE COMPARISON')

print('='*80)

display(comparison_df.style.background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE', 'MAPE (%)'])

                            .background_gradient(cmap='RdYlGn', subset=['R²'])

                            .format({'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'MAPE (%)': '{:.2f}', 'R²': '{:.4f}'}))

# Visual comparison with log scale for all metrics

import numpy as np



# Create log-transformed versions

comparison_df_log = comparison_df.copy()

comparison_df_log['RMSE_log'] = np.log10(comparison_df['RMSE'])

comparison_df_log['MAE_log'] = np.log10(comparison_df['MAE'])

comparison_df_log['MAPE_log'] = np.log10(comparison_df['MAPE (%)'])



fig = make_subplots(

    rows=1, cols=4,

    subplot_titles=('RMSE (Log10)', 'MAE (Log10)', 'MAPE (Log10)', 'R² Score')

)



fig.add_trace(go.Bar(

    x=comparison_df_log['Model'], 

    y=comparison_df_log['RMSE_log'],

    text=comparison_df['RMSE'].round(0),

    textposition='auto',

    marker_color=['#d62728', '#2ca02c', '#ff7f0e']

), row=1, col=1)



fig.add_trace(go.Bar(

    x=comparison_df_log['Model'], 

    y=comparison_df_log['MAE_log'],

    text=comparison_df['MAE'].round(0),

    textposition='auto',

    marker_color=['#d62728', '#2ca02c', '#ff7f0e']

), row=1, col=2)



fig.add_trace(go.Bar(

    x=comparison_df_log['Model'], 

    y=comparison_df_log['MAPE_log'],

    text=comparison_df['MAPE (%)'].round(2),

    textposition='auto',

    marker_color=['#d62728', '#2ca02c', '#ff7f0e']

), row=1, col=3)



fig.add_trace(go.Bar(

    x=comparison_df_log['Model'], 

    y=comparison_df['R²'],

    text=comparison_df['R²'].round(4),

    textposition='auto',

    marker_color=['#d62728', '#2ca02c', '#ff7f0e']

), row=1, col=4)



fig.update_xaxes(title_text='Model', row=1, col=1)

fig.update_xaxes(title_text='Model', row=1, col=2)

fig.update_xaxes(title_text='Model', row=1, col=3)

fig.update_xaxes(title_text='Model', row=1, col=4)



fig.update_yaxes(title_text='Log10(RMSE)', row=1, col=1)

fig.update_yaxes(title_text='Log10(MAE)', row=1, col=2)

fig.update_yaxes(title_text='Log10(MAPE)', row=1, col=3)

fig.update_yaxes(title_text='R² Score', row=1, col=4)



fig.update_layout(height=400, showlegend=False, title_text='Model Performance Metrics Comparison')

fig.show()

# Best model

best_model_name = comparison_df.iloc[0]['Model']

best_rmse = comparison_df.iloc[0]['RMSE']

best_mape = comparison_df.iloc[0]['MAPE (%)']



print('\n' + '='*80)

print(f'🏆 BEST MODEL: {best_model_name}')

print('='*80)

print(f'   RMSE: {best_rmse:,.2f}')

print(f'   MAPE: {best_mape:.2f}%')

print('='*80)

# Aggregate predictions by date for comparison

valid_ml_clean_copy = valid_ml_clean.copy()

valid_ml_clean_copy['XGBoost_Pred'] = xgb_valid_pred

valid_ml_clean_copy['LightGBM_Pred'] = lgb_valid_pred



daily_comparison = valid_ml_clean_copy.groupby(DATE_COL).agg({

    TARGET_COL: 'sum',

    'XGBoost_Pred': 'sum',

    'LightGBM_Pred': 'sum'

}).reset_index()



# Plot

fig = go.Figure()

fig.add_trace(go.Scatter(x=daily_comparison[DATE_COL], y=daily_comparison[TARGET_COL],

                         mode='lines', name='Actual', line=dict(color='black', width=2)))

fig.add_trace(go.Scatter(x=daily_comparison[DATE_COL], y=daily_comparison['XGBoost_Pred'],

                         mode='lines', name='XGBoost', line=dict(dash='dash')))

fig.add_trace(go.Scatter(x=daily_comparison[DATE_COL], y=daily_comparison['LightGBM_Pred'],

                         mode='lines', name='LightGBM', line=dict(dash='dot')))



fig.update_layout(

    title='Validation Period: Actual vs Predicted Sales',

    xaxis_title='Date',

    yaxis_title='Total Sales',

    height=500,

    hovermode='x unified'

)

fig.show()

# Generate 6-week forecast using best model (assuming XGBoost)

print(f'Generating 6-week forecast using {best_model_name}...')



if best_model_name == 'XGBoost':

    best_model = xgb_model

elif best_model_name == 'LightGBM':

    best_model = lgb_model

else:

    best_model = prophet_model



# For ML models, prepare test data

if best_model_name in ['XGBoost', 'LightGBM']:

    # Combine train and test for lag feature generation

    full_data = pd.concat([train_clean_lag, test], ignore_index=True)

    full_data_lag = create_lag_features(full_data, TARGET_COL, STORE_COL, DATE_COL) if STORE_COL in full_data.columns else full_data

    

    # Get test data with features

    test_dates = test[DATE_COL].unique()

    test_ml = full_data_lag[full_data_lag[DATE_COL].isin(test_dates)].copy()

    

    # Prepare features

    X_test = test_ml[feature_cols].fillna(0)  # Fill NaN with 0 for any missing lags

    

    # Predict

    test_predictions = best_model.predict(X_test)

    test_ml['Predicted_Sales'] = test_predictions

    

    # Aggregate by date

    final_forecast = test_ml.groupby(DATE_COL)['Predicted_Sales'].sum().reset_index()

    final_forecast.columns = ['Date', 'Predicted Sales']

    

else:  # Prophet

    # Prophet already has forecast

    test_dates = test[DATE_COL].unique()

    final_forecast = prophet_forecast[prophet_forecast['ds'].isin(test_dates)][['ds', 'yhat']]

    final_forecast.columns = ['Date', 'Predicted Sales']



# Add week number

final_forecast = final_forecast.sort_values('Date').reset_index(drop=True)

final_forecast['Week'] = (final_forecast.index // 7) + 1



print(f'\n✓ 6-week forecast generated')

print(f'\nForecast period: {final_forecast["Date"].min()} to {final_forecast["Date"].max()}')

print(f'Total days: {len(final_forecast)}')

# Weekly summary

weekly_forecast = final_forecast.groupby('Week')['Predicted Sales'].agg([

    ('Total Sales', 'sum'),

    ('Avg Daily Sales', 'mean'),

    ('Min Daily', 'min'),

    ('Max Daily', 'max')

]).reset_index()



print('\n' + '='*80)

print('6-WEEK FORECAST SUMMARY')

print('='*80)

display(weekly_forecast.style.background_gradient(cmap='Blues', subset=['Total Sales', 'Avg Daily Sales'])

                              .format({'Total Sales': '{:,.0f}', 'Avg Daily Sales': '{:,.0f}',

                                       'Min Daily': '{:,.0f}', 'Max Daily': '{:,.0f}'}))

# Diagnostic: Check forecast values

print("="*80)

print("FORECAST DIAGNOSTICS")

print("="*80)

print(f"\nHistorical Sales (last 8 weeks):")

print(f"  Average daily sales: {historical_data['Sales'].mean():,.0f}")

print(f"  Min daily sales: {historical_data['Sales'].min():,.0f}")

print(f"  Max daily sales: {historical_data['Sales'].max():,.0f}")



print(f"\n6-Week Forecast:")

print(f"  Average daily sales: {final_forecast['Predicted Sales'].mean():,.0f}")

print(f"  Min daily sales: {final_forecast['Predicted Sales'].min():,.0f}")

print(f"  Max daily sales: {final_forecast['Predicted Sales'].max():,.0f}")



print(f"\nRatio (Forecast/Historical): {final_forecast['Predicted Sales'].mean() / historical_data['Sales'].mean():.2%}")



# Visualize 6-week forecast

fig = go.Figure()



# Add actual historical data (last 8 weeks for context)

historical_end = train_clean[DATE_COL].max()

historical_start = historical_end - timedelta(weeks=8)

historical_data = daily_sales[(daily_sales['Date'] >= historical_start) & 

                               (daily_sales['Date'] <= historical_end)]



fig.add_trace(go.Scatter(

    x=historical_data['Date'],

    y=historical_data['Sales'],

    mode='lines',

    name='Historical Sales',

    line=dict(color='#1f77b4', width=2)

))



# Add forecast

fig.add_trace(go.Scatter(

    x=final_forecast['Date'],

    y=final_forecast['Predicted Sales'],

    mode='lines',

    name='6-Week Forecast',

    line=dict(color='#ff7f0e', width=2, dash='dash')

))



# Add vertical line at forecast start

forecast_start_date = final_forecast['Date'].min()



fig.add_vline(

    x=forecast_start_date,

    line_dash='dot',

    line_color='gray'

)



fig.add_annotation(

    x=forecast_start_date,

    y=1.02,

    yref='paper',

    text='Forecast Start',

    showarrow=False,

    font=dict(size=10)

)



fig.show()

# Weekly forecast bar chart

fig = go.Figure()

fig.add_trace(go.Bar(

    x=weekly_forecast['Week'],

    y=weekly_forecast['Total Sales'],

    text=weekly_forecast['Total Sales'].apply(lambda x: f'{x:,.0f}'),

    textposition='auto',

    marker_color='lightcoral'

))



fig.update_layout(

    title='Predicted Total Sales by Week',

    xaxis_title='Week',

    yaxis_title='Total Sales',

    height=400

)

fig.show()

# Calculate growth trends

last_4weeks_actual = daily_sales[daily_sales['Date'] >= (daily_sales['Date'].max() - timedelta(weeks=4))]['Sales'].sum()

forecast_6weeks_total = final_forecast['Predicted Sales'].sum()

forecast_first_4weeks = final_forecast[final_forecast['Week'] <= 4]['Predicted Sales'].sum()



growth_rate = ((forecast_first_4weeks - last_4weeks_actual) / last_4weeks_actual) * 100



print('\n' + '='*80)

print('KEY INSIGHTS')

print('='*80)

print(f'\n📊 Forecast Summary:')

print(f'   • Total forecasted sales (6 weeks): {forecast_6weeks_total:,.0f}')

print(f'   • Average daily sales: {final_forecast["Predicted Sales"].mean():,.0f}')

print(f'   • Projected growth vs last 4 weeks: {growth_rate:+.2f}%')



print(f'\n📈 Best Performing Days:')

best_days = dow_sales.nlargest(3, 'Average Sales')[['Day', 'Average Sales']]

for idx, row in best_days.iterrows():

    print(f'   • {row["Day"]}: {row["Average Sales"]:,.0f} avg sales')



print(f'\n🎯 Best Performing Months:')

best_months = monthly_sales.nlargest(3, 'Average Sales')[['Month', 'Average Sales']]

month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 

               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

for idx, row in best_months.iterrows():

    print(f'   • {month_names[row["Month"]]}: {row["Average Sales"]:,.0f} avg sales')



print(f'\n🤖 Model Performance:')

print(f'   • Best model: {best_model_name}')

print(f'   • Prediction accuracy (MAPE): {100 - best_mape:.2f}%')

print(f'   • Average error: {best_mape:.2f}%')



print('\n' + '='*80)

print('RECOMMENDATIONS')

print('='*80)

print('\n1. 📦 Inventory Management:')

print(f'   • Stock up for weeks {weekly_forecast.nlargest(2, "Total Sales")["Week"].values}')

print(f'   • Peak demand expected: {weekly_forecast["Total Sales"].max():,.0f} units')



print('\n2. 💼 Staffing Recommendations:')

print(f'   • Increase staff on: {best_days.iloc[0]["Day"]}s and {best_days.iloc[1]["Day"]}s')

print(f'   • Weekend demand: {"Higher" if dow_sales[dow_sales["DayOfWeek"] >= 5]["Average Sales"].mean() > dow_sales[dow_sales["DayOfWeek"] < 5]["Average Sales"].mean() else "Lower"} than weekdays')



print('\n3. 🎯 Marketing Strategy:')

if growth_rate > 0:

    print(f'   • Capitalize on {growth_rate:.1f}% growth trend')

    print('   • Focus on retention and upselling strategies')

else:

    print(f'   • Address {abs(growth_rate):.1f}% decline with promotions')

    print('   • Consider targeted marketing campaigns')



print('\n4. 📊 Model Deployment:')

print(f'   • Deploy {best_model_name} for production forecasting')

print('   • Retrain weekly with new data')

print('   • Monitor MAPE - alert if exceeds 15%')



print('\n' + '='*80)

# Create outputs directory if it doesn't exist

import os

os.makedirs('outputs', exist_ok=True)



# Save results

final_forecast.to_csv('outputs/6_week_forecast.csv', index=False)

weekly_forecast.to_csv('outputs/weekly_forecast_summary.csv', index=False)

comparison_df.to_csv('outputs/model_comparison.csv', index=False)



print('\n✓ Results saved to outputs/ directory')

print('  - 6_week_forecast.csv')

print('  - weekly_forecast_summary.csv')

print('  - model_comparison.csv')
