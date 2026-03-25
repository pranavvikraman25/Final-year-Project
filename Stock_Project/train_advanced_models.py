"""
Train Advanced Stock Prediction Models
- GRU Model
- Temporal Fusion Transformer
- Ensemble with Regime Detection
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from advanced_models import (
    GRUPredictor, 
    TemporalFusionBlock,
    MarketRegimeDetector,
    EnsembleForecaster,
    TradingSignalGenerator,
    create_advanced_features,
    ADVANCED_FEATURE_COLUMNS
)

print("=" * 60)
print("ADVANCED STOCK PREDICTION MODEL TRAINING")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
nse_indexes = pd.read_csv('archive/nse_indexes.csv', parse_dates=['Date'])
df = nse_indexes[nse_indexes['Index'] == 'NIFTY 50'].copy()
df = df.sort_values('Date').reset_index(drop=True)
print(f"    Loaded {len(df)} records for NIFTY 50")

# Create features
print("\n[2] Creating advanced features...")
df_features = create_advanced_features(df)
print(f"    Created {len(ADVANCED_FEATURE_COLUMNS)} features")
print(f"    Dataset shape after feature engineering: {df_features.shape}")

# Prepare data
print("\n[3] Preparing training data...")
X = df_features[ADVANCED_FEATURE_COLUMNS].values
y = df_features['Close'].shift(-1).dropna().values
X = X[:-1]  # Align with target

# Train/test split (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")

# Train GRU Model
print("\n[4] Training GRU Model...")
gru_model = GRUPredictor(seq_length=60, n_features=len(ADVANCED_FEATURE_COLUMNS))
gru_history = gru_model.fit(X_train, y_train, epochs=50, batch_size=32)
print("    GRU Model trained!")

# Save GRU model
gru_model.model.save('gru_model.h5')
with open('gru_scaler_X.pkl', 'wb') as f:
    pickle.dump(gru_model.scaler_X, f)
with open('gru_scaler_y.pkl', 'wb') as f:
    pickle.dump(gru_model.scaler_y, f)
print("    GRU Model saved!")

# Train Temporal Fusion Model
print("\n[5] Training Temporal Fusion Model...")
tft_model = TemporalFusionBlock(seq_length=60, n_features=len(ADVANCED_FEATURE_COLUMNS))
tft_history = tft_model.fit(X_train, y_train, epochs=50, batch_size=32)
print("    Temporal Fusion Model trained!")

# Save TFT model
tft_model.model.save('tft_model.h5')
with open('tft_scaler_X.pkl', 'wb') as f:
    pickle.dump(tft_model.scaler_X, f)
with open('tft_scaler_y.pkl', 'wb') as f:
    pickle.dump(tft_model.scaler_y, f)
print("    Temporal Fusion Model saved!")

# Evaluate models
print("\n[6] Evaluating models...")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n    {name}:")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      MAE:  {mae:.2f}")
    print(f"      R²:   {r2:.4f}")
    print(f"      MAPE: {mape:.2f}%")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

# GRU predictions on test set
gru_preds = []
for i in range(60, len(X_test)):
    pred = gru_model.predict(X_test[:i+1])
    gru_preds.append(pred)
gru_metrics = evaluate(y_test[60:], gru_preds, "GRU Model")

# TFT predictions on test set
tft_preds = []
for i in range(60, len(X_test)):
    pred = tft_model.predict(X_test[:i+1])
    tft_preds.append(pred)
tft_metrics = evaluate(y_test[60:], tft_preds, "Temporal Fusion Model")

# Test Regime Detection
print("\n[7] Testing Regime Detection...")
regime_detector = MarketRegimeDetector()
regime, confidence = regime_detector.detect_regime(df_features)
print(f"    Current Market Regime: {regime.upper()}")
print(f"    Confidence: {confidence:.2%}")

# Test Trading Signal Generator
print("\n[8] Testing Trading Signal Generator...")
signal_gen = TradingSignalGenerator(risk_tolerance='moderate')

# Get latest prediction
latest_pred = gru_model.predict(X_test)
current_price = df_features['Close'].iloc[-1]

signal = signal_gen.generate_signal(
    current_price=current_price,
    prediction_data={
        'prediction': latest_pred,
        'std': abs(latest_pred - current_price) * 0.1,
        'confidence': 0.75
    },
    regime=regime
)

print(f"\n    Current Price: ₹{current_price:,.2f}")
print(f"    Predicted Price: ₹{latest_pred:,.2f}")
print(f"    Signal: {signal['signal']}")
print(f"    Signal Strength: {signal['strength']:.2%}")
print(f"    Expected Return: {signal['expected_return']*100:.2f}%")
print(f"    Reasoning: {', '.join(signal['reasoning'])}")

# Save configuration
config = {
    'feature_columns': ADVANCED_FEATURE_COLUMNS,
    'seq_length': 60,
    'gru_metrics': gru_metrics,
    'tft_metrics': tft_metrics,
    'trained_date': datetime.now().isoformat()
}
with open('advanced_models_config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nSaved files:")
print("  - gru_model.h5")
print("  - gru_scaler_X.pkl, gru_scaler_y.pkl")
print("  - tft_model.h5")
print("  - tft_scaler_X.pkl, tft_scaler_y.pkl")
print("  - advanced_models_config.pkl")
