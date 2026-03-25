"""
Stock Market EDA - Visualizations
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
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("Loading data for visualizations...")

# Load data
nse_indexes = pd.read_csv('archive/nse_indexes.csv', parse_dates=['Date'])
stocks_df = pd.read_csv('archive/stocks_df.csv', parse_dates=['Date'])

# Add time features
stocks_df['Year'] = stocks_df['Date'].dt.year
stocks_df['Month'] = stocks_df['Date'].dt.month
stocks_df['DayOfWeek'] = stocks_df['Date'].dt.dayofweek

# Prepare NIFTY 50 data
nifty50 = nse_indexes[nse_indexes['Index'] == 'NIFTY 50'].copy()
nifty50 = nifty50.sort_values('Date')
nifty50['Daily_Return'] = nifty50['Close'].pct_change() * 100

print("Creating visualizations...")

# =============================================================================
# FIGURE 1: NIFTY 50 Historical Price
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: NIFTY 50 Price History
ax1 = axes[0, 0]
ax1.plot(nifty50['Date'], nifty50['Close'], linewidth=0.8, color='#2E86AB')
ax1.set_title('NIFTY 50 Historical Close Price (1995-Present)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price (INR)')
ax1.grid(True, alpha=0.3)

# Plot 2: NIFTY 50 Daily Returns Distribution
ax2 = axes[0, 1]
returns_clean = nifty50['Daily_Return'].dropna()
ax2.hist(returns_clean, bins=100, edgecolor='black', alpha=0.7, color='#A23B72')
ax2.axvline(returns_clean.mean(), color='red', linestyle='--', label=f'Mean: {returns_clean.mean():.2f}%')
ax2.axvline(0, color='black', linestyle='-', linewidth=2)
ax2.set_title('NIFTY 50 Daily Returns Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Daily Return (%)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.set_xlim(-15, 15)

# Plot 3: Volume over time (NIFTY 50)
ax3 = axes[1, 0]
nifty50_vol = nifty50[nifty50['Volume'] > 0]  # Filter zero volume
ax3.fill_between(nifty50_vol['Date'], nifty50_vol['Volume']/1e6, alpha=0.5, color='#F18F01')
ax3.set_title('NIFTY 50 Trading Volume Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Volume (Millions)')
ax3.grid(True, alpha=0.3)

# Plot 4: Yearly Returns Box Plot
ax4 = axes[1, 1]
nifty50['Year'] = nifty50['Date'].dt.year
yearly_data = nifty50[nifty50['Year'] >= 2000]
yearly_data.boxplot(column='Daily_Return', by='Year', ax=ax4)
ax4.set_title('NIFTY 50 Daily Returns by Year', fontsize=12, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Daily Return (%)')
plt.suptitle('')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_nifty50_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: eda_nifty50_analysis.png")

# =============================================================================
# FIGURE 2: Stock Market Overview
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Number of stocks over time
ax1 = axes[0, 0]
stocks_per_year = stocks_df.groupby('Year')['Stock'].nunique()
ax1.bar(stocks_per_year.index, stocks_per_year.values, color='#2E86AB', edgecolor='black')
ax1.set_title('Number of Unique Stocks Traded per Year', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Stocks')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Average Daily Change % by Month (Seasonality)
ax2 = axes[0, 1]
monthly_returns = stocks_df.groupby('Month')['Change Pct'].mean()
colors = ['#2E86AB' if x >= 0 else '#E94F37' for x in monthly_returns.values]
ax2.bar(monthly_returns.index, monthly_returns.values, color=colors, edgecolor='black')
ax2.axhline(0, color='black', linewidth=1)
ax2.set_title('Average Daily Returns by Month (Seasonality)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Daily Return (%)')
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Day of Week Effect
ax3 = axes[1, 0]
dow_returns = stocks_df.groupby('DayOfWeek')['Change Pct'].mean()
# Filter to only weekdays (0-4)
dow_returns = dow_returns[dow_returns.index.isin([0, 1, 2, 3, 4])]
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
colors = ['#2E86AB' if x >= 0 else '#E94F37' for x in dow_returns.values]
ax3.bar(dow_names, dow_returns.values, color=colors, edgecolor='black')
ax3.axhline(0, color='black', linewidth=1)
ax3.set_title('Average Daily Returns by Day of Week', fontsize=12, fontweight='bold')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Average Daily Return (%)')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution of Daily Change %
ax4 = axes[1, 1]
change_pct_clean = stocks_df['Change Pct'].dropna()
change_pct_clipped = change_pct_clean.clip(-20, 20)  # Clip for better visualization
ax4.hist(change_pct_clipped, bins=100, edgecolor='black', alpha=0.7, color='#A23B72')
ax4.axvline(0, color='black', linestyle='-', linewidth=2)
ax4.axvline(change_pct_clean.mean(), color='red', linestyle='--', 
            label=f'Mean: {change_pct_clean.mean():.3f}%')
ax4.set_title('Distribution of Daily Price Changes (All Stocks)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Daily Change (%)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.set_xlim(-20, 20)

plt.tight_layout()
plt.savefig('eda_market_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: eda_market_overview.png")

# =============================================================================
# FIGURE 3: Correlation and Volatility Analysis
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation Heatmap
ax1 = axes[0, 0]
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change Pct']
corr_matrix = stocks_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax1,
            fmt='.2f', square=True, linewidths=0.5)
ax1.set_title('Correlation Matrix (Stock Features)', fontsize=12, fontweight='bold')

# Plot 2: Top 20 Most Volatile Stocks
ax2 = axes[0, 1]
stock_volatility = stocks_df.groupby('Stock')['Change Pct'].std().sort_values(ascending=False).head(20)
ax2.barh(range(len(stock_volatility)), stock_volatility.values, color='#E94F37', edgecolor='black')
ax2.set_yticks(range(len(stock_volatility)))
ax2.set_yticklabels(stock_volatility.index)
ax2.set_title('Top 20 Most Volatile Stocks (by Std of Daily Change)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Standard Deviation of Daily Change (%)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Top 20 Highest Volume Stocks
ax3 = axes[1, 0]
stock_volume = stocks_df.groupby('Stock')['Volume'].mean().sort_values(ascending=False).head(20)
ax3.barh(range(len(stock_volume)), stock_volume.values/1e6, color='#2E86AB', edgecolor='black')
ax3.set_yticks(range(len(stock_volume)))
ax3.set_yticklabels(stock_volume.index)
ax3.set_title('Top 20 Stocks by Average Daily Volume', fontsize=12, fontweight='bold')
ax3.set_xlabel('Average Volume (Millions)')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Volume vs Volatility Scatter
ax4 = axes[1, 1]
stock_stats = stocks_df.groupby('Stock').agg({
    'Volume': 'mean',
    'Change Pct': 'std'
}).dropna()
ax4.scatter(stock_stats['Volume']/1e6, stock_stats['Change Pct'], alpha=0.5, s=20, color='#2E86AB')
ax4.set_title('Volume vs Volatility (All Stocks)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Average Volume (Millions)')
ax4.set_ylabel('Volatility (Std of Daily Change %)')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_correlation_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: eda_correlation_volatility.png")

# =============================================================================
# FIGURE 4: Multiple Index Comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Select key indexes for comparison
key_indexes = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'NIFTY PHARMA', 'NIFTY AUTO']
available_indexes = nse_indexes['Index'].unique()
key_indexes = [idx for idx in key_indexes if idx in available_indexes]

# Plot 1: Normalized Price Comparison (2015 onwards)
ax1 = axes[0, 0]
for idx in key_indexes:
    idx_data = nse_indexes[(nse_indexes['Index'] == idx) & (nse_indexes['Date'] >= '2015-01-01')].copy()
    if len(idx_data) > 0:
        idx_data = idx_data.sort_values('Date')
        normalized = idx_data['Close'] / idx_data['Close'].iloc[0] * 100
        ax1.plot(idx_data['Date'], normalized, label=idx, linewidth=1.2)
ax1.set_title('Normalized Index Performance (Base=100, 2015 onwards)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Index Volatility Comparison
ax2 = axes[0, 1]
index_volatility = []
for idx in nse_indexes['Index'].unique():
    idx_data = nse_indexes[nse_indexes['Index'] == idx].copy()
    idx_data['Return'] = idx_data['Close'].pct_change() * 100
    vol = idx_data['Return'].std()
    if not np.isnan(vol):
        index_volatility.append({'Index': idx, 'Volatility': vol})

vol_df = pd.DataFrame(index_volatility).sort_values('Volatility', ascending=False).head(15)
ax2.barh(range(len(vol_df)), vol_df['Volatility'].values, color='#E94F37', edgecolor='black')
ax2.set_yticks(range(len(vol_df)))
ax2.set_yticklabels(vol_df['Index'].values)
ax2.set_title('Top 15 Most Volatile Indexes', fontsize=12, fontweight='bold')
ax2.set_xlabel('Daily Return Volatility (%)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Data Availability by Index
ax3 = axes[1, 0]
index_records = nse_indexes.groupby('Index').size().sort_values(ascending=False).head(20)
ax3.barh(range(len(index_records)), index_records.values, color='#2E86AB', edgecolor='black')
ax3.set_yticks(range(len(index_records)))
ax3.set_yticklabels(index_records.index)
ax3.set_title('Data Points Available by Index (Top 20)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of Records')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: NIFTY 50 Rolling Volatility
ax4 = axes[1, 1]
nifty50_recent = nifty50[nifty50['Date'] >= '2010-01-01'].copy()
nifty50_recent['Rolling_Vol_30'] = nifty50_recent['Daily_Return'].rolling(30).std()
nifty50_recent['Rolling_Vol_90'] = nifty50_recent['Daily_Return'].rolling(90).std()
ax4.plot(nifty50_recent['Date'], nifty50_recent['Rolling_Vol_30'], 
         label='30-day Rolling Vol', linewidth=1, alpha=0.8)
ax4.plot(nifty50_recent['Date'], nifty50_recent['Rolling_Vol_90'], 
         label='90-day Rolling Vol', linewidth=1.5, color='red')
ax4.set_title('NIFTY 50 Rolling Volatility (2010 onwards)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Rolling Volatility (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_index_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: eda_index_comparison.png")

# =============================================================================
# FIGURE 5: Time Series Decomposition Prep
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: NIFTY 50 with Moving Averages
ax1 = axes[0, 0]
nifty50_ma = nifty50[nifty50['Date'] >= '2015-01-01'].copy()
nifty50_ma['SMA_50'] = nifty50_ma['Close'].rolling(50).mean()
nifty50_ma['SMA_200'] = nifty50_ma['Close'].rolling(200).mean()
ax1.plot(nifty50_ma['Date'], nifty50_ma['Close'], label='Close', linewidth=0.8, alpha=0.7)
ax1.plot(nifty50_ma['Date'], nifty50_ma['SMA_50'], label='50-day SMA', linewidth=1.5)
ax1.plot(nifty50_ma['Date'], nifty50_ma['SMA_200'], label='200-day SMA', linewidth=1.5)
ax1.set_title('NIFTY 50 with Moving Averages (2015 onwards)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (INR)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Autocorrelation of Returns
ax2 = axes[0, 1]
returns = nifty50['Daily_Return'].dropna()
lags = 30
autocorr = [returns.autocorr(lag=i) for i in range(1, lags+1)]
ax2.bar(range(1, lags+1), autocorr, color='#2E86AB', edgecolor='black')
ax2.axhline(0, color='black', linewidth=1)
ax2.axhline(1.96/np.sqrt(len(returns)), color='red', linestyle='--', alpha=0.7)
ax2.axhline(-1.96/np.sqrt(len(returns)), color='red', linestyle='--', alpha=0.7)
ax2.set_title('NIFTY 50 Returns Autocorrelation', fontsize=12, fontweight='bold')
ax2.set_xlabel('Lag (Days)')
ax2.set_ylabel('Autocorrelation')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Price Range (High-Low) Analysis
ax3 = axes[1, 0]
nifty50_range = nifty50[nifty50['Date'] >= '2015-01-01'].copy()
nifty50_range['Range_Pct'] = (nifty50_range['High'] - nifty50_range['Low']) / nifty50_range['Close'] * 100
ax3.plot(nifty50_range['Date'], nifty50_range['Range_Pct'].rolling(20).mean(), 
         linewidth=1, color='#E94F37')
ax3.set_title('NIFTY 50 Daily Range % (20-day Rolling Average)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Daily Range (%)')
ax3.grid(True, alpha=0.3)

# Plot 4: Cumulative Returns
ax4 = axes[1, 1]
nifty50_cum = nifty50[nifty50['Date'] >= '2010-01-01'].copy()
nifty50_cum['Cumulative_Return'] = (1 + nifty50_cum['Daily_Return'].fillna(0)/100).cumprod() - 1
ax4.fill_between(nifty50_cum['Date'], nifty50_cum['Cumulative_Return']*100, 
                  alpha=0.5, color='#2E86AB')
ax4.plot(nifty50_cum['Date'], nifty50_cum['Cumulative_Return']*100, 
         linewidth=1, color='#2E86AB')
ax4.axhline(0, color='black', linewidth=1)
ax4.set_title('NIFTY 50 Cumulative Returns (2010 onwards)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Cumulative Return (%)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_timeseries_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: eda_timeseries_analysis.png")

print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("   1. eda_nifty50_analysis.png")
print("   2. eda_market_overview.png")
print("   3. eda_correlation_volatility.png")
print("   4. eda_index_comparison.png")
print("   5. eda_timeseries_analysis.png")
