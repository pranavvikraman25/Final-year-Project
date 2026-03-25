"""
Advanced Stock Prediction Models
- GRU (Gated Recurrent Unit)
- Regime Detection
- Ensemble Forecasting with Uncertainty Estimation
- Decision-Oriented Outputs
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        GRU, LSTM, Dense, Dropout, Input, 
        Bidirectional, Attention, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class MarketRegimeDetector:
    """
    Detects market regimes: Bull, Bear, Sideways
    Uses volatility, trend, and momentum indicators
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.regime_model = None
        self.scaler = StandardScaler()
        
    def calculate_regime_features(self, df):
        """Calculate features for regime detection"""
        features = pd.DataFrame(index=df.index)
        
        # Trend indicators
        features['sma_ratio'] = df['Close'] / df['Close'].rolling(self.lookback).mean()
        features['price_momentum'] = df['Close'].pct_change(self.lookback)
        
        # Volatility
        features['volatility'] = df['Close'].pct_change().rolling(self.lookback).std()
        features['atr'] = (df['High'] - df['Low']).rolling(self.lookback).mean() / df['Close']
        
        # Volume trend
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(self.lookback).mean()
        else:
            features['volume_ratio'] = 1.0
        
        # RSI for momentum
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features.dropna()
    
    def detect_regime(self, df):
        """
        Detect current market regime
        Returns: 'bull', 'bear', or 'sideways'
        """
        features = self.calculate_regime_features(df)
        
        if len(features) == 0:
            return 'sideways', 0.5
        
        latest = features.iloc[-1]
        
        # Rule-based regime detection with confidence
        bull_score = 0
        bear_score = 0
        
        # Trend analysis
        if latest['sma_ratio'] > 1.02:
            bull_score += 2
        elif latest['sma_ratio'] < 0.98:
            bear_score += 2
            
        # Momentum
        if latest['price_momentum'] > 0.05:
            bull_score += 1.5
        elif latest['price_momentum'] < -0.05:
            bear_score += 1.5
            
        # RSI
        if latest['rsi'] > 60:
            bull_score += 1
        elif latest['rsi'] < 40:
            bear_score += 1
            
        # Volatility adjustment
        if latest['volatility'] > 0.02:
            # High volatility reduces confidence
            bull_score *= 0.8
            bear_score *= 0.8
        
        total_score = bull_score + bear_score + 0.1
        
        if bull_score > bear_score + 1:
            regime = 'bull'
            confidence = min(bull_score / total_score, 0.95)
        elif bear_score > bull_score + 1:
            regime = 'bear'
            confidence = min(bear_score / total_score, 0.95)
        else:
            regime = 'sideways'
            confidence = 1 - (abs(bull_score - bear_score) / total_score)
        
        return regime, confidence
    
    def get_regime_history(self, df, window=5):
        """Get regime for each point in history"""
        regimes = []
        confidences = []
        
        for i in range(self.lookback + window, len(df)):
            subset = df.iloc[:i]
            regime, conf = self.detect_regime(subset)
            regimes.append(regime)
            confidences.append(conf)
        
        return regimes, confidences


class GRUPredictor:
    """
    GRU-based stock price predictor
    Faster than LSTM with similar performance
    """
    
    def __init__(self, seq_length=60, n_features=27):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def build_model(self):
        """Build GRU model architecture"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = Sequential([
            # Bidirectional GRU for better context
            Bidirectional(GRU(64, return_sequences=True), 
                         input_shape=(self.seq_length, self.n_features)),
            Dropout(0.2),
            
            GRU(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def create_sequences(self, X, y):
        """Create sequences for GRU input"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.1):
        """Train the GRU model"""
        if self.model is None:
            self.build_model()
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X, return_sequences=False):
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X)
        
        if len(X_scaled) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} samples")
        
        # Use last sequence for prediction
        X_seq = X_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
        
        pred_scaled = self.model.predict(X_seq, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        return pred
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """
        Monte Carlo Dropout for uncertainty estimation
        Returns prediction with confidence interval
        """
        X_scaled = self.scaler_X.transform(X)
        X_seq = X_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
        
        # Enable dropout during inference for MC Dropout
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X_seq, training=True)  # Enable dropout
            predictions.append(pred.numpy()[0, 0])
        
        predictions = np.array(predictions)
        pred_mean = self.scaler_y.inverse_transform([[predictions.mean()]])[0, 0]
        pred_std = predictions.std() * (self.scaler_y.data_max_ - self.scaler_y.data_min_)
        
        return {
            'prediction': pred_mean,
            'std': pred_std[0],
            'ci_lower': pred_mean - 1.96 * pred_std[0],
            'ci_upper': pred_mean + 1.96 * pred_std[0],
            'confidence': max(0, 1 - pred_std[0] / pred_mean) if pred_mean > 0 else 0.5
        }


class TemporalFusionBlock:
    """
    Simplified Temporal Fusion Transformer-inspired architecture
    Combines attention mechanism with temporal processing
    """
    
    def __init__(self, seq_length=60, n_features=27, d_model=64, n_heads=4):
        self.seq_length = seq_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
    
    def build_model(self):
        """Build Temporal Fusion model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        # Initial projection
        x = Dense(self.d_model)(inputs)
        x = LayerNormalization()(x)
        
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads
        )(x, x)
        x = LayerNormalization()(x + attention_output)
        
        # Temporal processing with GRU
        x = GRU(self.d_model, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        
        # Second attention layer
        attention_output2 = MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads
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
        
        self.model = model
        return model
    
    def create_sequences(self, X, y):
        """Create sequences"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.1):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make prediction"""
        X_scaled = self.scaler_X.transform(X)
        X_seq = X_scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
        
        pred_scaled = self.model.predict(X_seq, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
        
        return pred


class EnsembleForecaster:
    """
    Ensemble model combining multiple predictors
    with regime-aware weighting and uncertainty estimation
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.regime_detector = MarketRegimeDetector()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
    
    def get_regime_weights(self, regime):
        """Adjust model weights based on market regime"""
        regime_adjustments = {
            'bull': {'LSTM': 1.2, 'GRU': 1.1, 'TFT': 1.0, 'Linear': 0.8},
            'bear': {'LSTM': 1.0, 'GRU': 1.2, 'TFT': 1.1, 'Linear': 0.9},
            'sideways': {'LSTM': 0.9, 'GRU': 1.0, 'TFT': 1.2, 'Linear': 1.1}
        }
        return regime_adjustments.get(regime, {})
    
    def predict(self, X, df_for_regime=None):
        """
        Make ensemble prediction with uncertainty
        """
        predictions = {}
        
        # Detect regime if data provided
        regime = 'sideways'
        regime_confidence = 0.5
        if df_for_regime is not None:
            regime, regime_confidence = self.regime_detector.detect_regime(df_for_regime)
        
        regime_weights = self.get_regime_weights(regime)
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    pred_result = model.predict_with_uncertainty(X)
                    predictions[name] = pred_result
                else:
                    pred = model.predict(X) if hasattr(model, 'predict') else None
                    if pred is not None:
                        predictions[name] = {
                            'prediction': pred,
                            'std': 0,
                            'confidence': 0.7
                        }
            except Exception as e:
                print(f"Error in {name}: {e}")
                continue
        
        if not predictions:
            return None
        
        # Weighted ensemble
        total_weight = 0
        weighted_pred = 0
        uncertainties = []
        
        for name, pred_data in predictions.items():
            base_weight = self.weights.get(name, 1.0)
            regime_adj = regime_weights.get(name, 1.0)
            conf_adj = pred_data.get('confidence', 0.7)
            
            final_weight = base_weight * regime_adj * conf_adj
            weighted_pred += pred_data['prediction'] * final_weight
            total_weight += final_weight
            uncertainties.append(pred_data.get('std', 0))
        
        ensemble_pred = weighted_pred / total_weight if total_weight > 0 else 0
        ensemble_std = np.mean(uncertainties) if uncertainties else 0
        
        return {
            'prediction': ensemble_pred,
            'std': ensemble_std,
            'ci_lower': ensemble_pred - 1.96 * ensemble_std,
            'ci_upper': ensemble_pred + 1.96 * ensemble_std,
            'regime': regime,
            'regime_confidence': regime_confidence,
            'individual_predictions': predictions
        }


class TradingSignalGenerator:
    """
    Generates trading signals based on predictions and market conditions
    Outputs: BUY, SELL, HOLD with confidence levels
    """
    
    def __init__(self, risk_tolerance='moderate'):
        self.risk_tolerance = risk_tolerance
        self.thresholds = self._get_thresholds()
    
    def _get_thresholds(self):
        """Get signal thresholds based on risk tolerance"""
        thresholds = {
            'conservative': {
                'buy_threshold': 0.03,   # 3% expected gain
                'sell_threshold': -0.02,  # 2% expected loss
                'confidence_min': 0.7
            },
            'moderate': {
                'buy_threshold': 0.02,
                'sell_threshold': -0.015,
                'confidence_min': 0.6
            },
            'aggressive': {
                'buy_threshold': 0.01,
                'sell_threshold': -0.01,
                'confidence_min': 0.5
            }
        }
        return thresholds.get(self.risk_tolerance, thresholds['moderate'])
    
    def generate_signal(self, current_price, prediction_data, regime='sideways'):
        """
        Generate trading signal
        
        Args:
            current_price: Current stock price
            prediction_data: Dict with 'prediction', 'std', 'confidence'
            regime: Current market regime
        
        Returns:
            Dict with signal, confidence, reasoning
        """
        pred_price = prediction_data['prediction']
        uncertainty = prediction_data.get('std', 0)
        confidence = prediction_data.get('confidence', 0.5)
        
        expected_return = (pred_price - current_price) / current_price
        
        # Adjust for uncertainty
        risk_adjusted_return = expected_return - (uncertainty / current_price)
        
        # Regime adjustments
        regime_multiplier = {
            'bull': 1.1,    # More aggressive in bull market
            'bear': 0.8,    # More conservative in bear market
            'sideways': 1.0
        }.get(regime, 1.0)
        
        adjusted_return = risk_adjusted_return * regime_multiplier
        
        # Generate signal
        signal = 'HOLD'
        signal_strength = 0.5
        reasoning = []
        
        if adjusted_return > self.thresholds['buy_threshold']:
            if confidence >= self.thresholds['confidence_min']:
                signal = 'BUY'
                signal_strength = min(0.95, 0.5 + adjusted_return * 10)
                reasoning.append(f"Expected return: {expected_return*100:.2f}%")
                reasoning.append(f"Risk-adjusted return: {adjusted_return*100:.2f}%")
            else:
                signal = 'WEAK BUY'
                signal_strength = 0.4
                reasoning.append("Positive outlook but low confidence")
                
        elif adjusted_return < self.thresholds['sell_threshold']:
            if confidence >= self.thresholds['confidence_min']:
                signal = 'SELL'
                signal_strength = min(0.95, 0.5 + abs(adjusted_return) * 10)
                reasoning.append(f"Expected loss: {expected_return*100:.2f}%")
            else:
                signal = 'WEAK SELL'
                signal_strength = 0.4
                reasoning.append("Negative outlook but low confidence")
        else:
            signal = 'HOLD'
            signal_strength = 0.5
            reasoning.append("No clear directional signal")
        
        # Add regime context
        reasoning.append(f"Market regime: {regime}")
        
        return {
            'signal': signal,
            'strength': signal_strength,
            'expected_return': expected_return,
            'risk_adjusted_return': adjusted_return,
            'confidence': confidence,
            'reasoning': reasoning,
            'target_price': pred_price,
            'stop_loss': current_price * (1 + self.thresholds['sell_threshold']),
            'take_profit': pred_price
        }
    
    def generate_position_size(self, signal_data, portfolio_value, max_position_pct=0.1):
        """
        Calculate recommended position size based on signal strength
        """
        if signal_data['signal'] in ['HOLD', 'WEAK BUY', 'WEAK SELL']:
            return 0
        
        base_position = portfolio_value * max_position_pct
        
        # Adjust by signal strength and confidence
        adjusted_position = base_position * signal_data['strength'] * signal_data['confidence']
        
        return round(adjusted_position, 2)


def create_advanced_features(df):
    """
    Create advanced features for the models
    """
    features = df.copy()
    
    # Basic features
    features['Returns'] = features['Close'].pct_change()
    features['Log_Returns'] = np.log(features['Close'] / features['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        features[f'SMA_{window}'] = features['Close'].rolling(window).mean()
        features[f'EMA_{window}'] = features['Close'].ewm(span=window).mean()
    
    # MACD
    features['MACD'] = features['EMA_12'] if 'EMA_12' in features else \
                       features['Close'].ewm(span=12).mean() - features['Close'].ewm(span=26).mean()
    features['MACD_Signal'] = features['MACD'].ewm(span=9).mean()
    
    # RSI
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Bollinger Bands
    features['BB_Middle'] = features['Close'].rolling(20).mean()
    bb_std = features['Close'].rolling(20).std()
    features['BB_Upper'] = features['BB_Middle'] + 2 * bb_std
    features['BB_Lower'] = features['BB_Middle'] - 2 * bb_std
    features['BB_Width'] = (features['BB_Upper'] - features['BB_Lower']) / features['BB_Middle']
    features['BB_Position'] = (features['Close'] - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'] + 1e-10)
    
    # Volatility
    features['Volatility_5'] = features['Returns'].rolling(5).std()
    features['Volatility_20'] = features['Returns'].rolling(20).std()
    
    # ATR (Average True Range)
    high_low = features['High'] - features['Low']
    high_close = abs(features['High'] - features['Close'].shift())
    low_close = abs(features['Low'] - features['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['ATR'] = true_range.rolling(14).mean()
    
    # Momentum
    features['Momentum_5'] = features['Close'] / features['Close'].shift(5) - 1
    features['Momentum_10'] = features['Close'] / features['Close'].shift(10) - 1
    features['Momentum_20'] = features['Close'] / features['Close'].shift(20) - 1
    
    # Price position
    features['Price_Position_52w'] = (features['Close'] - features['Close'].rolling(252).min()) / \
                                      (features['Close'].rolling(252).max() - features['Close'].rolling(252).min() + 1e-10)
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        features[f'Close_Lag_{lag}'] = features['Close'].shift(lag)
        features[f'Returns_Lag_{lag}'] = features['Returns'].shift(lag)
    
    return features.dropna()


# Feature columns for models
ADVANCED_FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'BB_Position',
    'Volatility_5', 'Volatility_20', 'ATR',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3'
]
