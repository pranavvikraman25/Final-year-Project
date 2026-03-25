"""
Stock Market EDA - NSE Indexes and Stocks Analysis
For Stock Forecasting and Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("NSE STOCK MARKET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] LOADING DATA...")

# Load indexes list
indexes_list = pd.read_csv('archive/indexes_df.csv')
print(f"   - Index symbols loaded: {len(indexes_list)} indexes")

# Load NSE indexes historical data
nse_indexes = pd.read_csv('archive/nse_indexes.csv', parse_dates=['Date'])
print(f"   - NSE Indexes data: {nse_indexes.shape[0]:,} rows, {nse_indexes.shape[1]} columns")

# Load stocks data (large file - use chunking for initial analysis)
print("   - Loading stocks data (this may take a moment)...")
stocks_df = pd.read_csv('archive/stocks_df.csv', parse_dates=['Date'])
print(f"   - Stocks data: {stocks_df.shape[0]:,} rows, {stocks_df.shape[1]} columns")

# =============================================================================
# 2. DATA OVERVIEW
# =============================================================================
print("\n" + "=" * 60)
print("[2] DATA OVERVIEW")
print("=" * 60)

print("\n--- NSE Indexes Data ---")
print(f"Columns: {list(nse_indexes.columns)}")
print(f"\nData Types:\n{nse_indexes.dtypes}")
print(f"\nDate Range: {nse_indexes['Date'].min()} to {nse_indexes['Date'].max()}")
print(f"Unique Indexes: {nse_indexes['Index'].nunique()}")
print(f"\nSample Data:")
print(nse_indexes.head())

print("\n--- Stocks Data ---")
print(f"Columns: {list(stocks_df.columns)}")
print(f"\nData Types:\n{stocks_df.dtypes}")
print(f"\nDate Range: {stocks_df['Date'].min()} to {stocks_df['Date'].max()}")
print(f"Unique Stocks: {stocks_df['Stock'].nunique()}")
print(f"\nSample Data:")
print(stocks_df.head())

# =============================================================================
# 3. MISSING VALUES ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[3] MISSING VALUES ANALYSIS")
print("=" * 60)

print("\n--- NSE Indexes Missing Values ---")
missing_indexes = nse_indexes.isnull().sum()
missing_pct_indexes = (missing_indexes / len(nse_indexes) * 100).round(2)
missing_df_indexes = pd.DataFrame({
    'Missing Count': missing_indexes,
    'Missing %': missing_pct_indexes
})
print(missing_df_indexes)

print("\n--- Stocks Missing Values ---")
missing_stocks = stocks_df.isnull().sum()
missing_pct_stocks = (missing_stocks / len(stocks_df) * 100).round(2)
missing_df_stocks = pd.DataFrame({
    'Missing Count': missing_stocks,
    'Missing %': missing_pct_stocks
})
print(missing_df_stocks)

# =============================================================================
# 4. STATISTICAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("[4] STATISTICAL SUMMARY")
print("=" * 60)

print("\n--- NSE Indexes Statistics ---")
print(nse_indexes.describe())

print("\n--- Stocks Statistics ---")
print(stocks_df.describe())

# =============================================================================
# 5. INDEX-WISE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[5] INDEX-WISE ANALYSIS")
print("=" * 60)

index_summary = nse_indexes.groupby('Index').agg({
    'Date': ['min', 'max', 'count'],
    'Close': ['mean', 'min', 'max', 'std'],
    'Volume': 'mean'
}).round(2)
index_summary.columns = ['Start_Date', 'End_Date', 'Records', 'Avg_Close', 'Min_Close', 'Max_Close', 'Std_Close', 'Avg_Volume']
print("\nTop 10 Indexes by Average Close Price:")
print(index_summary.sort_values('Avg_Close', ascending=False).head(10))

# =============================================================================
# 6. STOCK-WISE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[6] STOCK-WISE ANALYSIS")
print("=" * 60)

stock_summary = stocks_df.groupby('Stock').agg({
    'Date': ['min', 'max', 'count'],
    'Close': ['mean', 'min', 'max', 'std'],
    'Volume': 'mean',
    'Change Pct': ['mean', 'std']
}).round(2)
stock_summary.columns = ['Start_Date', 'End_Date', 'Records', 'Avg_Close', 'Min_Close', 'Max_Close', 
                         'Std_Close', 'Avg_Volume', 'Avg_Change_Pct', 'Volatility']
print(f"\nTotal Unique Stocks: {len(stock_summary)}")
print("\nTop 10 Stocks by Average Volume:")
print(stock_summary.sort_values('Avg_Volume', ascending=False).head(10))

print("\nTop 10 Most Volatile Stocks (by Change Pct Std):")
print(stock_summary.sort_values('Volatility', ascending=False).head(10))

# =============================================================================
# 7. TIME-BASED ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[7] TIME-BASED ANALYSIS")
print("=" * 60)

# Add time features to stocks data
stocks_df['Year'] = stocks_df['Date'].dt.year
stocks_df['Month'] = stocks_df['Date'].dt.month
stocks_df['DayOfWeek'] = stocks_df['Date'].dt.dayofweek
stocks_df['Quarter'] = stocks_df['Date'].dt.quarter

# Yearly analysis
yearly_stats = stocks_df.groupby('Year').agg({
    'Stock': 'nunique',
    'Close': 'mean',
    'Volume': 'mean',
    'Change Pct': 'mean'
}).round(2)
yearly_stats.columns = ['Unique_Stocks', 'Avg_Close', 'Avg_Volume', 'Avg_Change_Pct']
print("\nYearly Market Overview:")
print(yearly_stats)

# Monthly seasonality
monthly_returns = stocks_df.groupby('Month')['Change Pct'].mean().round(3)
print("\nMonthly Average Returns (Seasonality):")
print(monthly_returns)

# Day of week analysis
dow_returns = stocks_df.groupby('DayOfWeek')['Change Pct'].mean().round(3)
dow_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
dow_returns.index = dow_returns.index.map(dow_names)
print("\nDay of Week Average Returns:")
print(dow_returns)

# =============================================================================
# 8. NIFTY 50 DEEP DIVE (Primary Index)
# =============================================================================
print("\n" + "=" * 60)
print("[8] NIFTY 50 DEEP DIVE")
print("=" * 60)

nifty50 = nse_indexes[nse_indexes['Index'] == 'NIFTY 50'].copy()
nifty50 = nifty50.sort_values('Date')

print(f"\nNIFTY 50 Data Points: {len(nifty50):,}")
print(f"Date Range: {nifty50['Date'].min()} to {nifty50['Date'].max()}")

# Calculate returns
nifty50['Daily_Return'] = nifty50['Close'].pct_change() * 100
nifty50['Cumulative_Return'] = (1 + nifty50['Daily_Return']/100).cumprod() - 1

print(f"\nNIFTY 50 Performance:")
print(f"   Starting Price: {nifty50['Close'].iloc[0]:.2f}")
print(f"   Ending Price: {nifty50['Close'].iloc[-1]:.2f}")
print(f"   Total Return: {((nifty50['Close'].iloc[-1] / nifty50['Close'].iloc[0]) - 1) * 100:.2f}%")
print(f"   Average Daily Return: {nifty50['Daily_Return'].mean():.4f}%")
print(f"   Daily Return Std (Volatility): {nifty50['Daily_Return'].std():.4f}%")
print(f"   Max Daily Gain: {nifty50['Daily_Return'].max():.2f}%")
print(f"   Max Daily Loss: {nifty50['Daily_Return'].min():.2f}%")

# =============================================================================
# 9. DATA QUALITY CHECKS
# =============================================================================
print("\n" + "=" * 60)
print("[9] DATA QUALITY CHECKS")
print("=" * 60)

# Check for duplicates
print("\n--- Duplicate Check ---")
dup_indexes = nse_indexes.duplicated().sum()
dup_stocks = stocks_df.duplicated().sum()
print(f"Duplicate rows in NSE Indexes: {dup_indexes}")
print(f"Duplicate rows in Stocks: {dup_stocks}")

# Check for negative values
print("\n--- Negative Value Check ---")
neg_close_idx = (nse_indexes['Close'] < 0).sum()
neg_close_stk = (stocks_df['Close'] < 0).sum()
neg_vol_idx = (nse_indexes['Volume'] < 0).sum()
neg_vol_stk = (stocks_df['Volume'] < 0).sum()
print(f"Negative Close prices in Indexes: {neg_close_idx}")
print(f"Negative Close prices in Stocks: {neg_close_stk}")
print(f"Negative Volume in Indexes: {neg_vol_idx}")
print(f"Negative Volume in Stocks: {neg_vol_stk}")

# Check for zero volume days
print("\n--- Zero Volume Check ---")
zero_vol_idx = (nse_indexes['Volume'] == 0).sum()
zero_vol_stk = (stocks_df['Volume'] == 0).sum()
print(f"Zero volume days in Indexes: {zero_vol_idx} ({zero_vol_idx/len(nse_indexes)*100:.2f}%)")
print(f"Zero volume days in Stocks: {zero_vol_stk} ({zero_vol_stk/len(stocks_df)*100:.2f}%)")

# Check OHLC consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
print("\n--- OHLC Consistency Check ---")
ohlc_issues_idx = ((nse_indexes['High'] < nse_indexes['Low']) | 
                   (nse_indexes['High'] < nse_indexes['Open']) |
                   (nse_indexes['High'] < nse_indexes['Close']) |
                   (nse_indexes['Low'] > nse_indexes['Open']) |
                   (nse_indexes['Low'] > nse_indexes['Close'])).sum()
ohlc_issues_stk = ((stocks_df['High'] < stocks_df['Low']) | 
                   (stocks_df['High'] < stocks_df['Open']) |
                   (stocks_df['High'] < stocks_df['Close']) |
                   (stocks_df['Low'] > stocks_df['Open']) |
                   (stocks_df['Low'] > stocks_df['Close'])).sum()
print(f"OHLC inconsistencies in Indexes: {ohlc_issues_idx}")
print(f"OHLC inconsistencies in Stocks: {ohlc_issues_stk}")

# =============================================================================
# 10. OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 60)
print("[10] OUTLIER DETECTION")
print("=" * 60)

# Using IQR method for Change Pct
Q1 = stocks_df['Change Pct'].quantile(0.25)
Q3 = stocks_df['Change Pct'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = stocks_df[(stocks_df['Change Pct'] < lower_bound) | (stocks_df['Change Pct'] > upper_bound)]
print(f"\nChange Pct Outlier Analysis (IQR Method):")
print(f"   Q1: {Q1:.2f}%, Q3: {Q3:.2f}%, IQR: {IQR:.2f}%")
print(f"   Lower Bound: {lower_bound:.2f}%, Upper Bound: {upper_bound:.2f}%")
print(f"   Outlier Count: {len(outliers):,} ({len(outliers)/len(stocks_df)*100:.2f}%)")

print("\nExtreme Daily Changes (>10% or <-10%):")
extreme_changes = stocks_df[abs(stocks_df['Change Pct']) > 10]
print(f"   Count: {len(extreme_changes):,}")
print(f"   Max Gain: {stocks_df['Change Pct'].max():.2f}%")
print(f"   Max Loss: {stocks_df['Change Pct'].min():.2f}%")

# =============================================================================
# 11. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("[11] CORRELATION ANALYSIS")
print("=" * 60)

# Correlation for numeric columns in stocks
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change Pct']
corr_matrix = stocks_df[numeric_cols].corr()
print("\nCorrelation Matrix (Stocks):")
print(corr_matrix.round(3))

# =============================================================================
# 12. RECOMMENDATIONS FOR FORECASTING
# =============================================================================
print("\n" + "=" * 60)
print("[12] RECOMMENDATIONS FOR STOCK FORECASTING")
print("=" * 60)

print("""
Based on the EDA, here are key insights for building a forecasting model:

1. DATA QUALITY:
   - Handle zero volume days (especially in older index data)
   - Address any OHLC inconsistencies before modeling
   - Consider removing or flagging extreme outliers (>10% daily change)

2. FEATURE ENGINEERING SUGGESTIONS:
   - Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
   - Lag features: Previous day's returns, rolling statistics
   - Time features: Day of week, month, quarter (seasonality exists)
   - Volatility measures: Rolling standard deviation, ATR

3. TARGET VARIABLE OPTIONS:
   - Next day's Close price (regression)
   - Next day's direction (classification: up/down)
   - Next day's return percentage

4. MODEL CONSIDERATIONS:
   - Time series models: ARIMA, SARIMA, Prophet
   - ML models: XGBoost, LightGBM, Random Forest
   - Deep Learning: LSTM, GRU, Transformer-based models
   - Ensemble approaches for better generalization

5. VALIDATION STRATEGY:
   - Use time-based train/test split (not random)
   - Walk-forward validation for realistic backtesting
   - Consider multiple time horizons (1-day, 5-day, 30-day)

6. KEY STOCKS/INDEXES FOR INITIAL MODELING:
   - NIFTY 50 (most liquid, longest history)
   - NIFTY BANK (sector-specific)
   - Top volume stocks for individual stock prediction
""")

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
