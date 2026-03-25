# -*- coding: utf-8 -*-
"""
# 📈 Advanced Stock Prediction System
## Regime-Aware Ensemble Forecasting with GRU & Temporal Fusion Transformer

Features:
- **GRU (Gated Recurrent Unit)** - Fast sequential processing
- **Temporal Fusion Transformer** - Attention-based forecasting
- **Market Regime Detection** - Bull/Bear/Sideways classification
- **Ensemble Forecasting** - Combined predictions with uncertainty
- **Trading Signals** - Buy/Sell/Hold with confidence levels

Upload to Google Colab and run all cells.
"""

#%% [markdown]
# ## 1. Setup and Installation

#%%
# Install required packages
!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow plotly -q

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    GRU, LSTM, Dense, Dropout, Input, 
    Bidirectional, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Plotly for interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
tf.random.set_seed(42)

print(f'TensorFlow version: {tf.__version__}')
print('✅ Setup complete!')

#%% [markdown]
# ## 2. Load Data

#%%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update path to your data
DATA_PATH = '/content/drive/MyDrive/Stock_Project/archive/'

#%%
# Load NSE Indexes data
nse_indexes = pd.read_csv(f'{DATA_PATH}nse_indexes.csv', parse_dates=['Date'])
print(f'Loaded {len(nse_indexes):,} records')
print(f'Indexes available: {nse_indexes["Index"].nunique()}')

# Filter NIFTY 50
df = nse_indexes[nse_indexes['Index'] == 'NIFTY 50'].copy()
df = df.sort_values('Date').reset_index(drop=True)
print(f'\nNIFTY 50: {len(df):,} records')
print(f'Date range: {df["Date"].min()} to {df["Date"].max()}')
df.head()


#%% [markdown]
# ## 3. Advanced Feature Engineering

#%%
def create_advanced_features(data):
    """
    Create comprehensive technical indicators and features
    """
    df = data.copy()
    
    # Basic returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # Momentum indicators
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Rate of Change
    df['ROC_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-10) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Williams %R
    df['Williams_R'] = (high_14 - df['Close']) / (high_14 - low_14 + 1e-10) * -100
    
    # Price position in range
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close']
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    # Time features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

#%%
# Apply feature engineering
df_features = create_advanced_features(df)
df_features = df_features.dropna().reset_index(drop=True)

print(f'Features created: {len(df_features.columns)}')
print(f'Dataset shape: {df_features.shape}')
df_features.head()

#%%
# Define feature columns for models
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'BB_Position', 'Volatility_5', 'Volatility_20',
    'ATR_Pct', 'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Stoch_K', 'Stoch_D', 'Williams_R',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3'
]

print(f'Using {len(FEATURE_COLUMNS)} features for prediction')


#%% [markdown]
# ## 4. Market Regime Detection

#%%
class MarketRegimeDetector:
    """
    Detects market regimes: Bull, Bear, Sideways
    Uses trend, volatility, and momentum indicators
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def detect_regime(self, df):
        """
        Detect current market regime
        Returns: regime ('bull', 'bear', 'sideways'), confidence (0-1)
        """
        if len(df) < self.lookback + 10:
            return 'sideways', 0.5
        
        recent = df.tail(self.lookback)
        
        # Calculate indicators
        sma_ratio = recent['Close'].iloc[-1] / recent['Close'].mean()
        price_momentum = (recent['Close'].iloc[-1] / recent['Close'].iloc[0]) - 1
        volatility = recent['Close'].pct_change().std()
        rsi = recent['RSI'].iloc[-1] if 'RSI' in recent.columns else 50
        
        # Scoring
        bull_score = 0
        bear_score = 0
        
        # Trend
        if sma_ratio > 1.02:
            bull_score += 2
        elif sma_ratio < 0.98:
            bear_score += 2
        
        # Momentum
        if price_momentum > 0.05:
            bull_score += 1.5
        elif price_momentum < -0.05:
            bear_score += 1.5
        
        # RSI
        if rsi > 60:
            bull_score += 1
        elif rsi < 40:
            bear_score += 1
        
        # High volatility reduces confidence
        vol_penalty = min(volatility * 20, 0.3)
        
        total = bull_score + bear_score + 0.1
        
        if bull_score > bear_score + 1:
            regime = 'bull'
            confidence = min((bull_score / total) * (1 - vol_penalty), 0.95)
        elif bear_score > bull_score + 1:
            regime = 'bear'
            confidence = min((bear_score / total) * (1 - vol_penalty), 0.95)
        else:
            regime = 'sideways'
            confidence = 0.6
        
        return regime, confidence
    
    def get_regime_history(self, df, window=20):
        """Get regime for rolling windows"""
        regimes = []
        confidences = []
        
        for i in range(self.lookback + window, len(df)):
            regime, conf = self.detect_regime(df.iloc[:i])
            regimes.append(regime)
            confidences.append(conf)
        
        return regimes, confidences

#%%
# Test regime detection
regime_detector = MarketRegimeDetector(lookback=20)
current_regime, regime_confidence = regime_detector.detect_regime(df_features)

print(f'Current Market Regime: {current_regime.upper()}')
print(f'Confidence: {regime_confidence:.1%}')

# Get regime history for visualization
regimes, confidences = regime_detector.get_regime_history(df_features.tail(500))
print(f'\nRegime distribution (last 500 days):')
print(pd.Series(regimes).value_counts())

#%% [markdown]
# ## 5. Data Preparation

#%%
# Prepare features and target
df_features['Target'] = df_features['Close'].shift(-1)
df_model = df_features.dropna(subset=['Target']).copy()

X = df_model[FEATURE_COLUMNS].values
y = df_model['Target'].values

# Time-based split (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f'Training samples: {len(X_train):,}')
print(f'Test samples: {len(X_test):,}')

#%%
# Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

#%%
# Create sequences for RNN models
SEQUENCE_LENGTH = 60

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)

print(f'Sequence shape: {X_train_seq.shape}')


#%% [markdown]
# ## 6. Model 1: GRU (Gated Recurrent Unit)

#%%
def build_gru_model(seq_length, n_features):
    """
    Build Bidirectional GRU model
    """
    model = Sequential([
        Bidirectional(GRU(64, return_sequences=True), 
                     input_shape=(seq_length, n_features)),
        Dropout(0.2),
        
        GRU(32, return_sequences=False),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    
    return model

#%%
# Build and train GRU model
print('Building GRU model...')
gru_model = build_gru_model(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
gru_model.summary()

#%%
# Train GRU
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print('\nTraining GRU model...')
gru_history = gru_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
print('✅ GRU training complete!')

#%%
# Plot GRU training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(gru_history.history['loss'], label='Train Loss')
axes[0].plot(gru_history.history['val_loss'], label='Val Loss')
axes[0].set_title('GRU Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(gru_history.history['mae'], label='Train MAE')
axes[1].plot(gru_history.history['val_mae'], label='Val MAE')
axes[1].set_title('GRU Training MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 7. Model 2: Temporal Fusion Transformer (Simplified)

#%%
def build_temporal_fusion_model(seq_length, n_features, d_model=64, n_heads=4):
    """
    Build Temporal Fusion Transformer-inspired model
    Combines attention mechanism with temporal processing
    """
    inputs = Input(shape=(seq_length, n_features))
    
    # Initial projection
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads
    )(x, x)
    x = LayerNormalization()(x + attention_output)
    
    # Temporal processing with GRU
    x = GRU(d_model, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # Second attention layer
    attention_output2 = MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads
    )(x, x)
    x = LayerNormalization()(x + attention_output2)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='huber',
        metrics=['mae']
    )
    
    return model

#%%
# Build and train TFT model
print('Building Temporal Fusion model...')
tft_model = build_temporal_fusion_model(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
tft_model.summary()

#%%
# Train TFT
print('\nTraining Temporal Fusion model...')
tft_history = tft_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
print('✅ Temporal Fusion training complete!')

#%%
# Plot TFT training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(tft_history.history['loss'], label='Train Loss')
axes[0].plot(tft_history.history['val_loss'], label='Val Loss')
axes[0].set_title('Temporal Fusion Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(tft_history.history['mae'], label='Train MAE')
axes[1].plot(tft_history.history['val_mae'], label='Val MAE')
axes[1].set_title('Temporal Fusion Training MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%% [markdown]
# ## 8. Model Evaluation

#%%
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f'\n{model_name} Metrics:')
    print(f'  RMSE: {rmse:,.2f}')
    print(f'  MAE:  {mae:,.2f}')
    print(f'  R²:   {r2:.4f}')
    print(f'  MAPE: {mape:.2f}%')
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

#%%
# GRU predictions
gru_pred_scaled = gru_model.predict(X_test_seq, verbose=0)
gru_pred = scaler_y.inverse_transform(gru_pred_scaled).flatten()
y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

gru_metrics = evaluate_model(y_test_actual, gru_pred, 'GRU')

#%%
# TFT predictions
tft_pred_scaled = tft_model.predict(X_test_seq, verbose=0)
tft_pred = scaler_y.inverse_transform(tft_pred_scaled).flatten()

tft_metrics = evaluate_model(y_test_actual, tft_pred, 'Temporal Fusion')

#%%
# Comparison chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

test_dates = df_model['Date'].iloc[train_size + SEQUENCE_LENGTH:].values

# GRU predictions
ax1 = axes[0, 0]
ax1.plot(test_dates, y_test_actual, label='Actual', alpha=0.7)
ax1.plot(test_dates, gru_pred, label='GRU Predicted', alpha=0.7)
ax1.set_title('GRU: Actual vs Predicted')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# TFT predictions
ax2 = axes[0, 1]
ax2.plot(test_dates, y_test_actual, label='Actual', alpha=0.7)
ax2.plot(test_dates, tft_pred, label='TFT Predicted', alpha=0.7)
ax2.set_title('Temporal Fusion: Actual vs Predicted')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Scatter plots
ax3 = axes[1, 0]
ax3.scatter(y_test_actual, gru_pred, alpha=0.5)
ax3.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--')
ax3.set_title(f'GRU Scatter (R²={gru_metrics["R2"]:.4f})')
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.scatter(y_test_actual, tft_pred, alpha=0.5)
ax4.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--')
ax4.set_title(f'TFT Scatter (R²={tft_metrics["R2"]:.4f})')
ax4.set_xlabel('Actual')
ax4.set_ylabel('Predicted')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 9. Ensemble Forecasting with Uncertainty

#%%
class EnsembleForecaster:
    """
    Ensemble model with regime-aware weighting and uncertainty estimation
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or {name: 1.0 for name in models.keys()}
        self.regime_detector = MarketRegimeDetector()
    
    def get_regime_weights(self, regime):
        """Adjust weights based on market regime"""
        adjustments = {
            'bull': {'GRU': 1.2, 'TFT': 1.0},
            'bear': {'GRU': 1.0, 'TFT': 1.2},
            'sideways': {'GRU': 1.0, 'TFT': 1.1}
        }
        return adjustments.get(regime, {})
    
    def predict_with_uncertainty(self, X_seq, df_for_regime=None, n_samples=50):
        """
        Make ensemble prediction with uncertainty estimation
        Uses Monte Carlo Dropout for uncertainty
        """
        predictions = {}
        
        # Detect regime
        regime = 'sideways'
        regime_conf = 0.5
        if df_for_regime is not None:
            regime, regime_conf = self.regime_detector.detect_regime(df_for_regime)
        
        regime_weights = self.get_regime_weights(regime)
        
        # Get predictions from each model with MC Dropout
        for name, model in self.models.items():
            preds = []
            for _ in range(n_samples):
                pred = model(X_seq, training=True)  # Enable dropout
                preds.append(pred.numpy()[0, 0])
            
            preds = np.array(preds)
            predictions[name] = {
                'mean': preds.mean(),
                'std': preds.std(),
                'predictions': preds
            }
        
        # Weighted ensemble
        total_weight = 0
        weighted_pred = 0
        all_preds = []
        
        for name, pred_data in predictions.items():
            base_weight = self.weights.get(name, 1.0)
            regime_adj = regime_weights.get(name, 1.0)
            final_weight = base_weight * regime_adj
            
            weighted_pred += pred_data['mean'] * final_weight
            total_weight += final_weight
            all_preds.extend(pred_data['predictions'])
        
        ensemble_mean = weighted_pred / total_weight
        ensemble_std = np.std(all_preds)
        
        return {
            'prediction': ensemble_mean,
            'std': ensemble_std,
            'ci_lower': ensemble_mean - 1.96 * ensemble_std,
            'ci_upper': ensemble_mean + 1.96 * ensemble_std,
            'regime': regime,
            'regime_confidence': regime_conf,
            'individual': predictions
        }

#%%
# Create ensemble
ensemble = EnsembleForecaster(
    models={'GRU': gru_model, 'TFT': tft_model},
    weights={'GRU': 1.0, 'TFT': 1.0}
)

# Test ensemble prediction
test_seq = X_test_seq[-1:].copy()
ensemble_result = ensemble.predict_with_uncertainty(
    test_seq, 
    df_for_regime=df_features,
    n_samples=100
)

# Inverse transform
pred_price = scaler_y.inverse_transform([[ensemble_result['prediction']]])[0, 0]
std_price = ensemble_result['std'] * (scaler_y.data_max_ - scaler_y.data_min_)[0]

print('\n' + '='*50)
print('ENSEMBLE PREDICTION WITH UNCERTAINTY')
print('='*50)
print(f"Predicted Price: ₹{pred_price:,.2f}")
print(f"Uncertainty (Std): ₹{std_price:,.2f}")
print(f"95% CI: ₹{pred_price - 1.96*std_price:,.2f} - ₹{pred_price + 1.96*std_price:,.2f}")
print(f"Market Regime: {ensemble_result['regime'].upper()}")
print(f"Regime Confidence: {ensemble_result['regime_confidence']:.1%}")


#%% [markdown]
# ## 10. Trading Signal Generator

#%%
class TradingSignalGenerator:
    """
    Generate trading signals with confidence levels
    Based on predictions, regime, and technical indicators
    """
    
    def __init__(self, threshold_buy=0.02, threshold_sell=-0.02):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
    
    def generate_signal(self, current_price, predicted_price, uncertainty, 
                       regime, regime_confidence, rsi=50, macd_hist=0):
        """
        Generate trading signal with confidence
        
        Returns:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            reasoning: explanation
        """
        expected_return = (predicted_price - current_price) / current_price
        
        # Uncertainty-adjusted return
        risk_adjusted_return = expected_return - (uncertainty / current_price)
        
        # Base signal from prediction
        if expected_return > self.threshold_buy:
            base_signal = 'BUY'
            base_conf = min(expected_return / 0.05, 1.0)
        elif expected_return < self.threshold_sell:
            base_signal = 'SELL'
            base_conf = min(abs(expected_return) / 0.05, 1.0)
        else:
            base_signal = 'HOLD'
            base_conf = 0.5
        
        # Regime adjustment
        regime_multiplier = 1.0
        if regime == 'bull' and base_signal == 'BUY':
            regime_multiplier = 1.2
        elif regime == 'bear' and base_signal == 'SELL':
            regime_multiplier = 1.2
        elif regime == 'bull' and base_signal == 'SELL':
            regime_multiplier = 0.7
        elif regime == 'bear' and base_signal == 'BUY':
            regime_multiplier = 0.7
        
        # RSI adjustment
        rsi_multiplier = 1.0
        if rsi > 70 and base_signal == 'BUY':
            rsi_multiplier = 0.7  # Overbought
        elif rsi < 30 and base_signal == 'SELL':
            rsi_multiplier = 0.7  # Oversold
        elif rsi < 30 and base_signal == 'BUY':
            rsi_multiplier = 1.2  # Good buy opportunity
        elif rsi > 70 and base_signal == 'SELL':
            rsi_multiplier = 1.2  # Good sell opportunity
        
        # MACD confirmation
        macd_multiplier = 1.0
        if macd_hist > 0 and base_signal == 'BUY':
            macd_multiplier = 1.1
        elif macd_hist < 0 and base_signal == 'SELL':
            macd_multiplier = 1.1
        
        # Final confidence
        final_conf = min(base_conf * regime_multiplier * rsi_multiplier * macd_multiplier * regime_confidence, 0.95)
        
        # Build reasoning
        reasons = []
        reasons.append(f"Expected return: {expected_return:.2%}")
        reasons.append(f"Regime: {regime} ({regime_confidence:.0%})")
        if rsi < 30:
            reasons.append("RSI indicates oversold")
        elif rsi > 70:
            reasons.append("RSI indicates overbought")
        if macd_hist > 0:
            reasons.append("MACD bullish")
        elif macd_hist < 0:
            reasons.append("MACD bearish")
        
        return {
            'signal': base_signal,
            'confidence': final_conf,
            'expected_return': expected_return,
            'risk_adjusted_return': risk_adjusted_return,
            'reasoning': reasons
        }

#%%
# Generate trading signal
signal_generator = TradingSignalGenerator()

current_price = df_features['Close'].iloc[-1]
rsi_current = df_features['RSI'].iloc[-1]
macd_hist_current = df_features['MACD_Hist'].iloc[-1]

signal_result = signal_generator.generate_signal(
    current_price=current_price,
    predicted_price=pred_price,
    uncertainty=std_price,
    regime=ensemble_result['regime'],
    regime_confidence=ensemble_result['regime_confidence'],
    rsi=rsi_current,
    macd_hist=macd_hist_current
)

print('\n' + '='*50)
print('TRADING SIGNAL')
print('='*50)
print(f"Signal: {signal_result['signal']}")
print(f"Confidence: {signal_result['confidence']:.1%}")
print(f"Expected Return: {signal_result['expected_return']:.2%}")
print(f"Risk-Adjusted Return: {signal_result['risk_adjusted_return']:.2%}")
print(f"\nReasoning:")
for reason in signal_result['reasoning']:
    print(f"  • {reason}")



#%% [markdown]
# ## 11. Multi-Day Forecasting with Uncertainty

#%%
def multi_day_forecast(models, scaler_X, scaler_y, last_sequence, df_features, 
                       n_days=7, n_samples=50):
    """
    Generate multi-day forecast with uncertainty bands
    """
    forecasts = []
    current_seq = last_sequence.copy()
    
    regime_detector = MarketRegimeDetector()
    
    for day in range(n_days):
        day_preds = {'GRU': [], 'TFT': []}
        
        # MC Dropout predictions
        for _ in range(n_samples):
            for name, model in models.items():
                pred = model(current_seq, training=True)
                day_preds[name].append(pred.numpy()[0, 0])
        
        # Ensemble
        all_preds = day_preds['GRU'] + day_preds['TFT']
        mean_pred = np.mean(all_preds)
        std_pred = np.std(all_preds)
        
        # Inverse transform
        pred_price = scaler_y.inverse_transform([[mean_pred]])[0, 0]
        std_price = std_pred * (scaler_y.data_max_ - scaler_y.data_min_)[0]
        
        forecasts.append({
            'day': day + 1,
            'prediction': pred_price,
            'std': std_price,
            'ci_lower': pred_price - 1.96 * std_price,
            'ci_upper': pred_price + 1.96 * std_price
        })
        
        # Update sequence for next prediction (simplified)
        new_features = current_seq[0, -1, :].copy()
        new_features[3] = mean_pred  # Update Close price
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_features
    
    return forecasts

#%%
# Generate 7-day forecast
models_dict = {'GRU': gru_model, 'TFT': tft_model}
last_seq = X_test_seq[-1:].copy()

print('Generating 7-day forecast...')
forecasts = multi_day_forecast(
    models_dict, scaler_X, scaler_y, last_seq, df_features, 
    n_days=7, n_samples=100
)

print('\n' + '='*50)
print('7-DAY FORECAST')
print('='*50)
print(f"{'Day':<6} {'Prediction':>12} {'95% CI Lower':>14} {'95% CI Upper':>14}")
print('-' * 50)
for f in forecasts:
    print(f"Day {f['day']:<3} ₹{f['prediction']:>10,.2f}   ₹{f['ci_lower']:>10,.2f}   ₹{f['ci_upper']:>10,.2f}")



#%% [markdown]
# ## 12. Interactive Visualizations

#%%
# Create comprehensive forecast visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Historical Prices with Predictions',
        'Model Comparison',
        '7-Day Forecast with Uncertainty',
        'Prediction Error Distribution',
        'Market Regime Analysis',
        'Trading Signal Dashboard'
    ),
    specs=[
        [{"colspan": 2}, None],
        [{}, {}],
        [{}, {}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Historical with predictions
hist_dates = df_model['Date'].iloc[train_size + SEQUENCE_LENGTH:].values
fig.add_trace(
    go.Scatter(x=hist_dates, y=y_test_actual, name='Actual', 
               line=dict(color='#3b82f6', width=1)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=hist_dates, y=gru_pred, name='GRU', 
               line=dict(color='#10b981', width=1, dash='dot')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=hist_dates, y=tft_pred, name='TFT', 
               line=dict(color='#f59e0b', width=1, dash='dot')),
    row=1, col=1
)

# 2. Model comparison bar chart
metrics_df = pd.DataFrame({
    'Model': ['GRU', 'TFT'],
    'RMSE': [gru_metrics['RMSE'], tft_metrics['RMSE']],
    'MAE': [gru_metrics['MAE'], tft_metrics['MAE']],
    'R2': [gru_metrics['R2'], tft_metrics['R2']]
})

fig.add_trace(
    go.Bar(x=['GRU', 'TFT'], y=[gru_metrics['R2'], tft_metrics['R2']], 
           name='R² Score', marker_color=['#10b981', '#f59e0b']),
    row=2, col=1
)

# 3. 7-day forecast
forecast_days = [f"Day {f['day']}" for f in forecasts]
forecast_preds = [f['prediction'] for f in forecasts]
forecast_lower = [f['ci_lower'] for f in forecasts]
forecast_upper = [f['ci_upper'] for f in forecasts]

fig.add_trace(
    go.Scatter(x=forecast_days, y=forecast_upper, fill=None, mode='lines',
               line=dict(color='rgba(59, 130, 246, 0.3)'), name='Upper CI'),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=forecast_days, y=forecast_lower, fill='tonexty', mode='lines',
               line=dict(color='rgba(59, 130, 246, 0.3)'), name='95% CI'),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=forecast_days, y=forecast_preds, mode='lines+markers',
               line=dict(color='#3b82f6', width=2), name='Forecast',
               marker=dict(size=8)),
    row=2, col=2
)

# 4. Error distribution
gru_errors = y_test_actual - gru_pred
tft_errors = y_test_actual - tft_pred

fig.add_trace(
    go.Histogram(x=gru_errors, name='GRU Errors', opacity=0.7,
                marker_color='#10b981', nbinsx=50),
    row=3, col=1
)
fig.add_trace(
    go.Histogram(x=tft_errors, name='TFT Errors', opacity=0.7,
                marker_color='#f59e0b', nbinsx=50),
    row=3, col=1
)

# 5. Regime indicator
regime_colors = {'bull': '#10b981', 'bear': '#ef4444', 'sideways': '#f59e0b'}
current_regime = ensemble_result['regime']

fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=ensemble_result['regime_confidence'] * 100,
        title={'text': f"Regime: {current_regime.upper()}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': regime_colors[current_regime]},
            'steps': [
                {'range': [0, 33], 'color': '#fee2e2'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#d1fae5'}
            ]
        }
    ),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=900,
    showlegend=True,
    title_text='📈 Advanced Stock Prediction Dashboard',
    title_x=0.5,
    template='plotly_white'
)

fig.show()

#%%
# Trading Signal Visualization
signal_color = {'BUY': '#10b981', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}

fig_signal = go.Figure()

# Signal gauge
fig_signal.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=signal_result['confidence'] * 100,
    title={'text': f"Signal: {signal_result['signal']}", 'font': {'size': 24}},
    delta={'reference': 50, 'increasing': {'color': '#10b981'}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1},
        'bar': {'color': signal_color[signal_result['signal']]},
        'bgcolor': 'white',
        'borderwidth': 2,
        'bordercolor': 'gray',
        'steps': [
            {'range': [0, 30], 'color': '#fee2e2'},
            {'range': [30, 70], 'color': '#fef3c7'},
            {'range': [70, 100], 'color': '#d1fae5'}
        ],
        'threshold': {
            'line': {'color': 'red', 'width': 4},
            'thickness': 0.75,
            'value': 70
        }
    }
))

fig_signal.update_layout(
    height=400,
    title_text='🎯 Trading Signal Confidence',
    title_x=0.5
)

fig_signal.show()



#%% [markdown]
# ## 13. Save Models

#%%
# Save models to Google Drive
SAVE_PATH = '/content/drive/MyDrive/Stock_Project/'

# Save GRU model
gru_model.save(f'{SAVE_PATH}gru_model.h5')
print('✅ GRU model saved')

# Save TFT model
tft_model.save(f'{SAVE_PATH}tft_model.h5')
print('✅ Temporal Fusion model saved')

# Save scalers
import pickle

with open(f'{SAVE_PATH}advanced_scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
print('✅ Feature scaler saved')

with open(f'{SAVE_PATH}advanced_scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print('✅ Target scaler saved')

# Save model configuration
config = {
    'sequence_length': SEQUENCE_LENGTH,
    'feature_columns': FEATURE_COLUMNS,
    'train_size': train_size,
    'gru_metrics': gru_metrics,
    'tft_metrics': tft_metrics
}

with open(f'{SAVE_PATH}model_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print('✅ Model configuration saved')

print(f'\n📁 All models saved to: {SAVE_PATH}')


#%% [markdown]
# ## 14. Summary & Results

#%%
print('='*60)
print('📊 ADVANCED STOCK PREDICTION SYSTEM - SUMMARY')
print('='*60)

print('\n🔧 MODELS TRAINED:')
print(f'  1. GRU (Bidirectional) - R²: {gru_metrics["R2"]:.4f}, MAPE: {gru_metrics["MAPE"]:.2f}%')
print(f'  2. Temporal Fusion Transformer - R²: {tft_metrics["R2"]:.4f}, MAPE: {tft_metrics["MAPE"]:.2f}%')

print('\n📈 MARKET ANALYSIS:')
print(f'  Current Regime: {ensemble_result["regime"].upper()}')
print(f'  Regime Confidence: {ensemble_result["regime_confidence"]:.1%}')

print('\n🎯 ENSEMBLE PREDICTION:')
print(f'  Current Price: ₹{current_price:,.2f}')
print(f'  Predicted Price: ₹{pred_price:,.2f}')
print(f'  Expected Change: {((pred_price - current_price) / current_price * 100):+.2f}%')
print(f'  Uncertainty: ±₹{std_price:,.2f}')

print('\n💹 TRADING SIGNAL:')
print(f'  Signal: {signal_result["signal"]}')
print(f'  Confidence: {signal_result["confidence"]:.1%}')

print('\n📅 7-DAY FORECAST:')
for f in forecasts:
    change = ((f['prediction'] - current_price) / current_price * 100)
    print(f'  Day {f["day"]}: ₹{f["prediction"]:,.2f} ({change:+.2f}%)')

print('\n✅ FEATURES:')
print('  • Regime-aware ensemble forecasting')
print('  • Monte Carlo Dropout uncertainty estimation')
print('  • Multi-day forecasting with confidence intervals')
print('  • Trading signals with technical indicator confirmation')
print('  • Interactive visualizations')

print('\n' + '='*60)
print('🚀 System ready for deployment!')
print('='*60)

