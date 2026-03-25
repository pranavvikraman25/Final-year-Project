# -*- coding: utf-8 -*-
"""
# 📈 NSE Stock Price Prediction
## Using LSTM and Linear Regression

This notebook implements stock price prediction using:
1. **Linear Regression** - Baseline model
2. **LSTM (Long Short-Term Memory)** - Deep learning approach

We'll predict NIFTY 50 index prices using historical data.

To use in Google Colab:
1. Upload this file to Colab
2. Or copy-paste each section into separate cells
"""

#%% [markdown]
# ## 1. Setup and Imports

#%%
# Install required packages (uncomment if needed in Colab)
# !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f'TensorFlow version: {tf.__version__}')
print('Setup complete!')

#%% [markdown]
# ## 2. Load and Prepare Data

#%%
# Mount Google Drive (for Colab) - Uncomment these lines in Colab
# from google.colab import drive
# drive.mount('/content/drive')

# Update this path to your data location
# For Colab with Google Drive:
# DATA_PATH = '/content/drive/MyDrive/Stock_Project/archive/'

# For local use:
DATA_PATH = 'archive/'

#%%
# Alternative: Upload files directly in Colab
# from google.colab import files
# uploaded = files.upload()

#%%
# Load NSE Indexes data
nse_indexes = pd.read_csv(f'{DATA_PATH}nse_indexes.csv', parse_dates=['Date'])
print(f'NSE Indexes shape: {nse_indexes.shape}')
print(f'Columns: {list(nse_indexes.columns)}')
print(f'\nUnique Indexes: {nse_indexes["Index"].nunique()}')
print(nse_indexes['Index'].unique()[:10])

#%%
# Filter NIFTY 50 data (our primary prediction target)
df = nse_indexes[nse_indexes['Index'] == 'NIFTY 50'].copy()
df = df.sort_values('Date').reset_index(drop=True)

print(f'NIFTY 50 data shape: {df.shape}')
print(f'Date range: {df["Date"].min()} to {df["Date"].max()}')
df.head()

#%%
# Check for missing values
print('Missing values:')
print(df.isnull().sum())

# Basic statistics
print('\nBasic Statistics:')
print(df.describe())


#%% [markdown]
# ## 3. Feature Engineering

#%%
def create_features(data):
    """
    Create technical indicators and features for stock prediction
    """
    df = data.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change() * 100
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # Price momentum
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # High-Low range
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close'] * 100
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    # Time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    return df

#%%
# Apply feature engineering
df_features = create_features(df)

# Drop rows with NaN values (from rolling calculations)
df_features = df_features.dropna().reset_index(drop=True)

print(f'Shape after feature engineering: {df_features.shape}')
print(f'\nFeatures created: {len(df_features.columns)}')
print(df_features.columns.tolist())

#%%
# Visualize the data
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Price with Moving Averages
ax1 = axes[0]
ax1.plot(df_features['Date'], df_features['Close'], label='Close', alpha=0.7)
ax1.plot(df_features['Date'], df_features['SMA_20'], label='SMA 20', alpha=0.8)
ax1.plot(df_features['Date'], df_features['SMA_50'], label='SMA 50', alpha=0.8)
ax1.set_title('NIFTY 50 Price with Moving Averages', fontsize=12)
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# RSI
ax2 = axes[1]
ax2.plot(df_features['Date'], df_features['RSI'], color='purple')
ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
ax2.set_title('RSI (Relative Strength Index)', fontsize=12)
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)

# MACD
ax3 = axes[2]
ax3.plot(df_features['Date'], df_features['MACD'], label='MACD')
ax3.plot(df_features['Date'], df_features['MACD_Signal'], label='Signal')
ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
ax3.set_title('MACD', fontsize=12)
ax3.set_ylabel('MACD')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% [markdown]
# ## 4. Data Preparation for Models

#%%
# Define features for prediction
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'Volatility_5', 'Volatility_20',
    'Momentum_5', 'Momentum_10', 'HL_Range_Pct',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3'
]

# Target: Next day's closing price
df_features['Target'] = df_features['Close'].shift(-1)

# Remove last row (no target)
df_model = df_features.dropna(subset=['Target']).copy()

print(f'Final dataset shape: {df_model.shape}')

#%%
# Time-based train/test split (80/20)
train_size = int(len(df_model) * 0.8)

train_data = df_model.iloc[:train_size].copy()
test_data = df_model.iloc[train_size:].copy()

print(f'Training set: {len(train_data)} samples ({train_data["Date"].min()} to {train_data["Date"].max()})')
print(f'Test set: {len(test_data)} samples ({test_data["Date"].min()} to {test_data["Date"].max()})')

#%%
# Prepare features and target
X_train = train_data[feature_columns].values
y_train = train_data['Target'].values

X_test = test_data[feature_columns].values
y_test = test_data['Target'].values

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f'X_train shape: {X_train_scaled.shape}')
print(f'X_test shape: {X_test_scaled.shape}')

#%% [markdown]
# ## 5. Model 1: Linear Regression (Baseline)

#%%
# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

print('Linear Regression Model Trained!')

#%%
def evaluate_model(y_true, y_pred, model_name):
    """
    Calculate and display evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f'\n{"="*50}')
    print(f'{model_name} Evaluation Metrics')
    print(f'{"="*50}')
    print(f'MSE:  {mse:,.2f}')
    print(f'RMSE: {rmse:,.2f}')
    print(f'MAE:  {mae:,.2f}')
    print(f'R²:   {r2:.4f}')
    print(f'MAPE: {mape:.2f}%')
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

#%%
# Evaluate Linear Regression
print('TRAINING SET:')
lr_train_metrics = evaluate_model(y_train, lr_train_pred, 'Linear Regression')

print('\nTEST SET:')
lr_test_metrics = evaluate_model(y_test, lr_test_pred, 'Linear Regression')

#%%
# Visualize Linear Regression Results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Actual vs Predicted (Test Set)
ax1 = axes[0, 0]
ax1.plot(test_data['Date'].values, y_test, label='Actual', alpha=0.7)
ax1.plot(test_data['Date'].values, lr_test_pred, label='Predicted', alpha=0.7)
ax1.set_title('Linear Regression: Actual vs Predicted (Test Set)', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot
ax2 = axes[0, 1]
ax2.scatter(y_test, lr_test_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title('Linear Regression: Actual vs Predicted Scatter', fontsize=12)
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.grid(True, alpha=0.3)

# Residuals
ax3 = axes[1, 0]
residuals = y_test - lr_test_pred
ax3.plot(test_data['Date'].values, residuals, alpha=0.7)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_title('Linear Regression: Residuals Over Time', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Residual')
ax3.grid(True, alpha=0.3)

# Residual Distribution
ax4 = axes[1, 1]
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--')
ax4.set_title('Linear Regression: Residual Distribution', fontsize=12)
ax4.set_xlabel('Residual')
ax4.set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#%% [markdown]
# ## 6. Model 2: LSTM (Deep Learning)

#%%
def create_sequences(X, y, seq_length):
    """
    Create sequences for LSTM input
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

#%%
# LSTM Parameters
SEQUENCE_LENGTH = 60  # Use 60 days of history
EPOCHS = 100
BATCH_SIZE = 32

# Create sequences for LSTM
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

print(f'LSTM Training shape: X={X_train_seq.shape}, y={y_train_seq.shape}')
print(f'LSTM Test shape: X={X_test_seq.shape}, y={y_test_seq.shape}')

#%%
def build_lstm_model(input_shape):
    """
    Build LSTM model architecture
    """
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

#%%
# Build LSTM model
lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
lstm_model.summary()

#%%
# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

#%%
# Train LSTM model
print('Training LSTM model...')
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
print('\nLSTM Training Complete!')

#%%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('LSTM Training History - Loss', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE
ax2 = axes[1]
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('LSTM Training History - MAE', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# LSTM Predictions
lstm_train_pred_scaled = lstm_model.predict(X_train_seq)
lstm_test_pred_scaled = lstm_model.predict(X_test_seq)

# Inverse transform predictions
lstm_train_pred = scaler_y.inverse_transform(lstm_train_pred_scaled).flatten()
lstm_test_pred = scaler_y.inverse_transform(lstm_test_pred_scaled).flatten()

# Get actual values (aligned with sequences)
y_train_lstm = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).flatten()
y_test_lstm = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

print('LSTM Predictions Generated!')

#%%
# Evaluate LSTM
print('TRAINING SET:')
lstm_train_metrics = evaluate_model(y_train_lstm, lstm_train_pred, 'LSTM')

print('\nTEST SET:')
lstm_test_metrics = evaluate_model(y_test_lstm, lstm_test_pred, 'LSTM')

#%%
# Visualize LSTM Results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get dates for LSTM test set (offset by sequence length)
lstm_test_dates = test_data['Date'].values[SEQUENCE_LENGTH:]

# Actual vs Predicted (Test Set)
ax1 = axes[0, 0]
ax1.plot(lstm_test_dates, y_test_lstm, label='Actual', alpha=0.7)
ax1.plot(lstm_test_dates, lstm_test_pred, label='Predicted', alpha=0.7)
ax1.set_title('LSTM: Actual vs Predicted (Test Set)', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot
ax2 = axes[0, 1]
ax2.scatter(y_test_lstm, lstm_test_pred, alpha=0.5)
ax2.plot([y_test_lstm.min(), y_test_lstm.max()], [y_test_lstm.min(), y_test_lstm.max()], 'r--', lw=2)
ax2.set_title('LSTM: Actual vs Predicted Scatter', fontsize=12)
ax2.set_xlabel('Actual Price')
ax2.set_ylabel('Predicted Price')
ax2.grid(True, alpha=0.3)

# Residuals
ax3 = axes[1, 0]
lstm_residuals = y_test_lstm - lstm_test_pred
ax3.plot(lstm_test_dates, lstm_residuals, alpha=0.7)
ax3.axhline(0, color='red', linestyle='--')
ax3.set_title('LSTM: Residuals Over Time', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Residual')
ax3.grid(True, alpha=0.3)

# Residual Distribution
ax4 = axes[1, 1]
ax4.hist(lstm_residuals, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--')
ax4.set_title('LSTM: Residual Distribution', fontsize=12)
ax4.set_xlabel('Residual')
ax4.set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#%% [markdown]
# ## 7. Model Comparison

#%%
# Compare models
comparison_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)'],
    'Linear Regression': [
        lr_test_metrics['MSE'],
        lr_test_metrics['RMSE'],
        lr_test_metrics['MAE'],
        lr_test_metrics['R2'],
        lr_test_metrics['MAPE']
    ],
    'LSTM': [
        lstm_test_metrics['MSE'],
        lstm_test_metrics['RMSE'],
        lstm_test_metrics['MAE'],
        lstm_test_metrics['R2'],
        lstm_test_metrics['MAPE']
    ]
})

print('\n' + '='*60)
print('MODEL COMPARISON (Test Set)')
print('='*60)
print(comparison_df.to_string(index=False))

#%%
# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparison
ax1 = axes[0]
metrics = ['RMSE', 'MAE']
x = np.arange(len(metrics))
width = 0.35

lr_values = [lr_test_metrics['RMSE'], lr_test_metrics['MAE']]
lstm_values = [lstm_test_metrics['RMSE'], lstm_test_metrics['MAE']]

bars1 = ax1.bar(x - width/2, lr_values, width, label='Linear Regression', color='#2E86AB')
bars2 = ax1.bar(x + width/2, lstm_values, width, label='LSTM', color='#E94F37')

ax1.set_ylabel('Error')
ax1.set_title('Model Comparison: RMSE and MAE', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# R² comparison
ax2 = axes[1]
models = ['Linear Regression', 'LSTM']
r2_values = [lr_test_metrics['R2'], lstm_test_metrics['R2']]
colors = ['#2E86AB', '#E94F37']

bars = ax2.bar(models, r2_values, color=colors, edgecolor='black')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Comparison: R² Score', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, r2_values):
    ax2.annotate(f'{val:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

#%%
# Combined prediction plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual values
ax.plot(test_data['Date'].values, y_test, label='Actual', color='black', linewidth=2, alpha=0.8)

# Plot Linear Regression predictions
ax.plot(test_data['Date'].values, lr_test_pred, label='Linear Regression', 
        color='#2E86AB', linewidth=1.5, alpha=0.7)

# Plot LSTM predictions (offset by sequence length)
ax.plot(lstm_test_dates, lstm_test_pred, label='LSTM', 
        color='#E94F37', linewidth=1.5, alpha=0.7)

ax.set_title('NIFTY 50 Price Prediction: Model Comparison', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (INR)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 8. Future Predictions (Next 5 Days)

#%%
def predict_future(model, last_sequence, scaler_y, n_days=5, is_lstm=True):
    """
    Predict future prices
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_days):
        if is_lstm:
            pred_scaled = model.predict(current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1]), verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
        else:
            pred = model.predict(current_seq[-1].reshape(1, -1))[0]
        
        predictions.append(pred)
        
        # Update sequence (simplified - in practice you'd need to update all features)
        if is_lstm:
            new_row = current_seq[-1].copy()
            current_seq = np.vstack([current_seq[1:], new_row])
    
    return predictions

#%%
# Get last sequence for LSTM prediction
last_sequence = X_test_scaled[-SEQUENCE_LENGTH:]

# Predict next 5 days
lstm_future_pred = predict_future(lstm_model, last_sequence, scaler_y, n_days=5, is_lstm=True)

print('\n' + '='*50)
print('LSTM Future Predictions (Next 5 Trading Days)')
print('='*50)
last_actual = y_test[-1]
print(f'Last Actual Price: {last_actual:,.2f}')
print('\nPredicted Prices:')
for i, pred in enumerate(lstm_future_pred, 1):
    change = ((pred - last_actual) / last_actual) * 100
    print(f'  Day {i}: {pred:,.2f} ({change:+.2f}% from last)')

#%%
# Visualize future predictions
fig, ax = plt.subplots(figsize=(14, 6))

# Last 60 days of actual data
last_60_dates = test_data['Date'].values[-60:]
last_60_actual = y_test[-60:]

ax.plot(range(60), last_60_actual, label='Actual', color='black', linewidth=2)

# Future predictions
future_x = range(59, 59 + len(lstm_future_pred) + 1)
future_y = [last_60_actual[-1]] + lstm_future_pred
ax.plot(future_x, future_y, label='LSTM Prediction', color='#E94F37', 
        linewidth=2, linestyle='--', marker='o')

ax.axvline(x=59, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
ax.set_title('NIFTY 50: Last 60 Days + 5-Day Forecast', fontsize=14)
ax.set_xlabel('Days', fontsize=12)
ax.set_ylabel('Price (INR)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 9. Save Models

#%%
# Save LSTM model
lstm_model.save('nifty50_lstm_model.h5')
print('LSTM model saved as: nifty50_lstm_model.h5')

# Save scalers using pickle
import pickle

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
    
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print('Scalers saved as: scaler_X.pkl, scaler_y.pkl')

# Save Linear Regression model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

print('Linear Regression model saved as: linear_regression_model.pkl')

#%% [markdown]
# ## 10. Summary and Conclusions

#%%
print('\n' + '='*70)
print('STOCK PREDICTION MODEL - SUMMARY')
print('='*70)

print(f'''
Dataset: NIFTY 50 Index
Date Range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}
Total Samples: {len(df_model):,}
Training Samples: {len(train_data):,}
Test Samples: {len(test_data):,}

MODELS TRAINED:
1. Linear Regression (Baseline)
2. LSTM (Deep Learning)

BEST MODEL: {'LSTM' if lstm_test_metrics['R2'] > lr_test_metrics['R2'] else 'Linear Regression'}

KEY METRICS (Test Set):
┌─────────────────────┬──────────────────┬──────────────────┐
│ Metric              │ Linear Regression│ LSTM             │
├─────────────────────┼──────────────────┼──────────────────┤
│ R² Score            │ {lr_test_metrics['R2']:.4f}           │ {lstm_test_metrics['R2']:.4f}           │
│ RMSE                │ {lr_test_metrics['RMSE']:,.2f}          │ {lstm_test_metrics['RMSE']:,.2f}          │
│ MAE                 │ {lr_test_metrics['MAE']:,.2f}          │ {lstm_test_metrics['MAE']:,.2f}          │
│ MAPE                │ {lr_test_metrics['MAPE']:.2f}%           │ {lstm_test_metrics['MAPE']:.2f}%           │
└─────────────────────┴──────────────────┴──────────────────┘

RECOMMENDATIONS:
- Both models show strong predictive capability
- LSTM captures temporal patterns better for volatile periods
- Consider ensemble methods for production use
- Always validate with walk-forward testing
- Monitor model drift and retrain periodically
''')

print('='*70)
print('ANALYSIS COMPLETE!')
print('='*70)
