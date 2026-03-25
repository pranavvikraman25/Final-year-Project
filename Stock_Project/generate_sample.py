import pandas as pd
import os

NIFTY50 = [
    'RELIANCE','TCS','HDFCBANK','BHARTIARTL','ICICIBANK','SBIN',
    'HINDUNILVR','ITC','LT','KOTAKBANK','AXISBANK','BAJFINANCE','MARUTI',
    'ASIANPAINT','HCLTECH','WIPRO','NESTLEIND','TITAN','BAJAJFINSV',
    'NTPC','SUNPHARMA','TATAMOTORS','TATASTEEL','JSWSTEEL','CIPLA',
    'DRREDDY','HINDALCO','TECHM','ONGC','COALINDIA','POWERGRID',
    'ADANIENT','ADANIPORTS','BRITANNIA','EICHERMOT','INDUSINDBK',
    'SBILIFE','TATACONSUM','HEROMOTOCO','DIVISLAB','GRASIM','BPCL',
    'APOLLOHOSP','UPL','INFY'
]

print("Reading stocks_df.csv...")
df = pd.read_csv('archive/stocks_df.csv', parse_dates=['Date'])
print("Total rows:", len(df))
print("Date range:", df['Date'].min(), "to", df['Date'].max())

# Last 3 years of available data
cutoff = df['Date'].max() - pd.DateOffset(years=3)
sample = df[df['Stock'].isin(NIFTY50) & (df['Date'] >= cutoff)]
print("Filtered rows:", len(sample))
print("Stocks found:", sample['Stock'].nunique())

os.makedirs('archive/sample', exist_ok=True)
sample.to_csv('archive/sample/stocks_sample.csv', index=False)
sz = os.path.getsize('archive/sample/stocks_sample.csv') / 1e6
print("Size:", round(sz, 2), "MB")

# Also copy indexes
idx_path = 'archive/nse_indexes.csv'
if os.path.exists(idx_path):
    idx = pd.read_csv(idx_path, parse_dates=['Date'])
    idx_sample = idx[idx['Date'] >= cutoff]
    idx_sample.to_csv('archive/sample/indexes_sample.csv', index=False)
    isz = os.path.getsize('archive/sample/indexes_sample.csv') / 1e6
    print("Index size:", round(isz, 2), "MB")

print("Done!")
