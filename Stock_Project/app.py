"""
📈 NSE Stock Forecasting & Prediction Dashboard
Enterprise-Grade Streamlit Application with Advanced ML Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow import
TF_AVAILABLE = False
tf = None
load_model = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="NSE Stock Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enterprise Look
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .dashboard-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #0d9488;
        height: 100%;
    }
    
    .metric-card-advanced {
        background: linear-gradient(135deg, #f0fdfa 0%, #ffffff 100%);
        border-left: 4px solid #8b5cf6;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.3rem 0;
    }
    
    .metric-delta-positive { color: #10b981; font-size: 0.9rem; font-weight: 600; }
    .metric-delta-negative { color: #ef4444; font-size: 0.9rem; font-weight: 600; }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .prediction-box-bearish {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .prediction-box-neutral {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-classic { background: #dbeafe; color: #1e40af; }
    .badge-advanced { background: #f3e8ff; color: #7c3aed; }
    .badge-ensemble { background: #d1fae5; color: #065f46; }
    
    .regime-indicator {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .regime-bull { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); color: #065f46; }
    .regime-bear { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b; }
    .regime-sideways { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #92400e; }
    
    .signal-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .signal-buy { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
    .signal-sell { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
    .signal-hold { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background-color: #0f172a; color: white; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING AND CACHING
# =============================================================================


@st.cache_data(show_spinner="Loading NSE market data...")
def load_data():
    """Load NSE stock data from bundled sample CSVs."""
    try:
        # --- Stocks ---
        stocks_df = pd.read_csv(
            'archive/sample/stocks_sample.csv',
            parse_dates=['Date'],
            dtype={
                'Open': 'float32', 'High': 'float32',
                'Low': 'float32', 'Close': 'float32',
                'Volume': 'int64', 'Change Pct': 'float32'
            }
        )
        stocks_df['Stock'] = stocks_df['Stock'].astype('category')

        # --- Indexes ---
        try:
            nse_indexes = pd.read_csv(
                'archive/sample/indexes_sample.csv',
                parse_dates=['Date']
            )
        except Exception:
            # Build a synthetic index from RELIANCE as fallback
            rel = stocks_df[stocks_df['Stock'] == 'RELIANCE'].copy()
            rel['Index'] = 'NIFTY 50'
            nse_indexes = rel.rename(columns={'Stock': '_drop'}).drop(columns=['_drop'], errors='ignore')

        return nse_indexes, stocks_df

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None



@st.cache_resource
def load_models():
    """Load all trained models - Classic and Advanced"""
    models = {}
    errors = []
    
    # === CLASSIC MODELS ===
    # Linear Regression
    try:
        with open('linear_regression_model.pkl', 'rb') as f:
            models['linear_regression'] = pickle.load(f)
    except Exception as e:
        errors.append(f"Linear Regression: {e}")
    
    # Classic scalers
    try:
        with open('scaler_X.pkl', 'rb') as f:
            models['scaler_X'] = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            models['scaler_y'] = pickle.load(f)
    except Exception as e:
        errors.append(f"Classic Scalers: {e}")
    
    # LSTM
    if TF_AVAILABLE:
        try:
            models['lstm'] = load_model('nifty50_lstm_model.h5', compile=False)
            models['lstm'].compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception as e:
            errors.append(f"LSTM: {e}")
    
    # === ADVANCED MODELS ===
    # GRU
    if TF_AVAILABLE:
        try:
            models['gru'] = load_model('gru_model.h5', compile=False)
            models['gru'].compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        except Exception as e:
            errors.append(f"GRU: {e}")
    
    # Temporal Fusion Transformer
    if TF_AVAILABLE:
        try:
            # Rebuild TFT architecture and load weights
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (Input, Dense, Dropout, GRU, 
                MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D)
            
            def build_tft_model(seq_length=60, n_features=32, d_model=64, n_heads=4):
                inputs = Input(shape=(seq_length, n_features))
                x = Dense(d_model)(inputs)
                x = LayerNormalization()(x)
                attention_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
                x = LayerNormalization()(x + attention_output)
                x = GRU(d_model, return_sequences=True)(x)
                x = Dropout(0.2)(x)
                attention_output2 = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
                x = LayerNormalization()(x + attention_output2)
                x = GlobalAveragePooling1D()(x)
                x = Dense(32, activation='relu')(x)
                x = Dropout(0.2)(x)
                outputs = Dense(1)(x)
                model = Model(inputs, outputs)
                return model
            
            models['tft'] = build_tft_model()
            models['tft'].load_weights('tft_model.h5')
            models['tft'].compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
        except Exception as e:
            errors.append(f"TFT: {e}")
    
    # Advanced scalers
    try:
        with open('advanced_scaler_X.pkl', 'rb') as f:
            models['advanced_scaler_X'] = pickle.load(f)
        with open('advanced_scaler_y.pkl', 'rb') as f:
            models['advanced_scaler_y'] = pickle.load(f)
    except Exception as e:
        errors.append(f"Advanced Scalers: {e}")
    
    # Model config
    try:
        with open('model_config.pkl', 'rb') as f:
            models['config'] = pickle.load(f)
    except Exception as e:
        errors.append(f"Model Config: {e}")
    
    if errors:
        models['_errors'] = errors
    
    return models


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Classic feature columns (27 features)
CLASSIC_FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'Volatility_5', 'Volatility_20',
    'Momentum_5', 'Momentum_10', 'HL_Range_Pct',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3'
]

# Advanced feature columns (32 features)
ADVANCED_FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI',
    'BB_Width', 'BB_Position', 'Volatility_5', 'Volatility_20',
    'ATR_Pct', 'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Stoch_K', 'Stoch_D', 'Williams_R',
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
    'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3'
]

def create_features(data):
    """Create comprehensive technical indicators"""
    df = data.copy()
    
    # Basic returns
    df['Returns'] = df['Close'].pct_change() * 100
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
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
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close']
    
    # Momentum
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14 + 1e-10) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Williams %R
    df['Williams_R'] = (high_14 - df['Close']) / (high_14 - low_14 + 1e-10) * -100
    
    # High-Low range
    df['HL_Range'] = df['High'] - df['Low']
    df['HL_Range_Pct'] = df['HL_Range'] / df['Close'] * 100
    
    # Lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
    
    return df


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

class MarketRegimeDetector:
    """Detects market regimes: Bull, Bear, Sideways"""
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def detect_regime(self, df):
        """Detect current market regime"""
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
        
        if sma_ratio > 1.02:
            bull_score += 2
        elif sma_ratio < 0.98:
            bear_score += 2
        
        if price_momentum > 0.05:
            bull_score += 1.5
        elif price_momentum < -0.05:
            bear_score += 1.5
        
        if rsi > 60:
            bull_score += 1
        elif rsi < 40:
            bear_score += 1
        
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


# =============================================================================
# TRADING SIGNAL GENERATOR
# =============================================================================

class TradingSignalGenerator:
    """Generate trading signals with confidence levels"""
    
    def __init__(self, threshold_buy=0.02, threshold_sell=-0.02):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
    
    def generate_signal(self, current_price, predicted_price, uncertainty, 
                       regime, regime_confidence, rsi=50, macd_hist=0):
        """Generate trading signal with confidence"""
        expected_return = (predicted_price - current_price) / current_price
        risk_adjusted_return = expected_return - (uncertainty / current_price)
        
        # Base signal
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
            rsi_multiplier = 0.7
        elif rsi < 30 and base_signal == 'SELL':
            rsi_multiplier = 0.7
        elif rsi < 30 and base_signal == 'BUY':
            rsi_multiplier = 1.2
        elif rsi > 70 and base_signal == 'SELL':
            rsi_multiplier = 1.2
        
        # MACD confirmation
        macd_multiplier = 1.0
        if macd_hist > 0 and base_signal == 'BUY':
            macd_multiplier = 1.1
        elif macd_hist < 0 and base_signal == 'SELL':
            macd_multiplier = 1.1
        
        final_conf = min(base_conf * regime_multiplier * rsi_multiplier * macd_multiplier * regime_confidence, 0.95)
        
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


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_price_chart(df, title="Price Chart"):
    """Create interactive candlestick chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Volume')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='OHLC',
            increasing_line_color='#10b981', decreasing_line_color='#ef4444'
        ), row=1, col=1
    )
    
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20',
                      line=dict(color='#3b82f6', width=1.5)), row=1, col=1
        )
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50',
                      line=dict(color='#f59e0b', width=1.5)), row=1, col=1
        )
    
    colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.7), row=2, col=1
    )
    
    fig.update_layout(
        height=600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False, template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_technical_indicators_chart(df):
    """Create technical indicators subplot"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD')
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Close',
                  line=dict(color='#1e3c72', width=2)), row=1, col=1
    )
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='#94a3b8', width=1, dash='dash')), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='#94a3b8', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(148, 163, 184, 0.1)'), row=1, col=1
        )
    
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                      line=dict(color='#8b5cf6', width=2)), row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
    
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
                      line=dict(color='#3b82f6', width=2)), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
                      line=dict(color='#f59e0b', width=2)), row=3, col=1
        )
        fig.add_hline(y=0, line_dash="solid", line_color="#94a3b8", row=3, col=1)
    
    fig.update_layout(
        height=700, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white', margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_prediction_chart(actual_dates, actual_prices, pred_dates, pred_prices, 
                           model_name, ci_lower=None, ci_upper=None):
    """Create prediction visualization with optional confidence intervals"""
    fig = go.Figure()
    
    actual_dates = pd.to_datetime(actual_dates)
    pred_dates = pd.to_datetime(pred_dates)
    
    fig.add_trace(
        go.Scatter(x=actual_dates, y=actual_prices, name='Actual',
                  line=dict(color='#1e3c72', width=2), mode='lines')
    )
    
    # Confidence interval
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(
            go.Scatter(x=pred_dates, y=ci_upper, fill=None, mode='lines',
                      line=dict(color='rgba(239, 68, 68, 0.2)'), name='Upper CI', showlegend=False)
        )
        fig.add_trace(
            go.Scatter(x=pred_dates, y=ci_lower, fill='tonexty', mode='lines',
                      line=dict(color='rgba(239, 68, 68, 0.2)'), name='95% CI',
                      fillcolor='rgba(239, 68, 68, 0.1)')
        )
    
    fig.add_trace(
        go.Scatter(x=pred_dates, y=pred_prices, name=f'{model_name} Prediction',
                  line=dict(color='#ef4444', width=2, dash='dash'),
                  mode='lines+markers', marker=dict(size=8))
    )
    
    if len(actual_dates) > 0:
        last_actual_date = actual_dates.iloc[-1] if hasattr(actual_dates, 'iloc') else actual_dates[-1]
        fig.add_shape(
            type="line", x0=last_actual_date, x1=last_actual_date, y0=0, y1=1,
            yref="paper", line=dict(color="#94a3b8", width=2, dash="dot")
        )
        fig.add_annotation(
            x=last_actual_date, y=1.05, yref="paper", text="Forecast Start",
            showarrow=False, font=dict(size=10, color="#64748b")
        )
    
    fig.update_layout(
        title=f'{model_name} Price Forecast', xaxis_title='Date', yaxis_title='Price (INR)',
        height=450, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    return fig

def create_ensemble_comparison_chart(predictions_dict, current_price):
    """Create comparison chart for all models"""
    fig = go.Figure()
    
    colors = {
        'Linear Regression': '#3b82f6',
        'LSTM': '#10b981',
        'GRU': '#8b5cf6',
        'TFT': '#f59e0b',
        'Ensemble': '#ef4444'
    }
    
    for model_name, pred_data in predictions_dict.items():
        forecast = pred_data.get('forecast', [])
        if forecast:
            days = list(range(1, len(forecast) + 1))
            fig.add_trace(
                go.Scatter(x=days, y=forecast, name=model_name,
                          line=dict(color=colors.get(model_name, '#64748b'), width=2),
                          mode='lines+markers', marker=dict(size=6))
            )
    
    fig.add_hline(y=current_price, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Current: ₹{current_price:,.2f}")
    
    fig.update_layout(
        title='Model Comparison - Forecast Trajectories',
        xaxis_title='Days Ahead', yaxis_title='Predicted Price (INR)',
        height=400, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def make_classic_predictions(df, models, forecast_days=5):
    """Generate predictions using classic models (LR, LSTM)"""
    predictions = {}
    
    df_features = create_features(df)
    df_clean = df_features.dropna()
    
    if len(df_clean) < 60:
        return None, "Insufficient data (need 60+ days)"
    
    try:
        X = df_clean[CLASSIC_FEATURE_COLUMNS].values
    except KeyError as e:
        return None, f"Missing features: {e}"
    
    scaler_X = models.get('scaler_X')
    scaler_y = models.get('scaler_y')
    
    if scaler_X is None or scaler_y is None:
        return None, "Classic scalers not loaded"
    
    try:
        X_scaled = scaler_X.transform(X)
    except Exception as e:
        return None, f"Scaling error: {e}"
    
    current_price = df_clean['Close'].iloc[-1]
    
    # Linear Regression
    lr_model = models.get('linear_regression')
    if lr_model:
        try:
            lr_forecast = []
            last_pred = lr_model.predict(X_scaled[-1].reshape(1, -1))[0]
            lr_forecast.append(last_pred)
            avg_daily_change = df_clean['Close'].pct_change().mean()
            for i in range(1, forecast_days):
                next_pred = lr_forecast[-1] * (1 + avg_daily_change * 0.5)
                lr_forecast.append(next_pred)
            predictions['Linear Regression'] = {
                'next_day': lr_forecast[0],
                'forecast': lr_forecast,
                'current_price': current_price,
                'model_type': 'classic'
            }
        except Exception as e:
            st.warning(f"LR error: {e}")
    
    # LSTM
    lstm_model = models.get('lstm')
    if lstm_model is not None and TF_AVAILABLE:
        seq_length = 60
        if len(X_scaled) >= seq_length:
            try:
                lstm_forecast = []
                current_seq = X_scaled[-seq_length:].copy()
                for _ in range(forecast_days):
                    pred_scaled = lstm_model.predict(current_seq.reshape(1, seq_length, -1), verbose=0)
                    pred = scaler_y.inverse_transform(pred_scaled)[0, 0]
                    lstm_forecast.append(float(pred))
                    current_seq = np.roll(current_seq, -1, axis=0)
                    current_seq[-1] = current_seq[-2].copy()
                predictions['LSTM'] = {
                    'next_day': lstm_forecast[0],
                    'forecast': lstm_forecast,
                    'current_price': current_price,
                    'model_type': 'classic'
                }
            except Exception as e:
                st.warning(f"LSTM error: {e}")
    
    return predictions, None

def make_advanced_predictions(df, models, forecast_days=5, n_samples=30):
    """Generate predictions using advanced models (GRU, TFT) with uncertainty"""
    predictions = {}
    
    df_features = create_features(df)
    df_clean = df_features.dropna()
    
    if len(df_clean) < 60:
        return None, "Insufficient data (need 60+ days)"
    
    scaler_X = models.get('advanced_scaler_X')
    scaler_y = models.get('advanced_scaler_y')
    
    if scaler_X is None or scaler_y is None:
        return None, "Advanced scalers not loaded"
    
    try:
        X = df_clean[ADVANCED_FEATURE_COLUMNS].values
        X_scaled = scaler_X.transform(X)
    except Exception as e:
        return None, f"Feature error: {e}"
    
    current_price = df_clean['Close'].iloc[-1]
    seq_length = 60
    
    if len(X_scaled) < seq_length:
        return None, "Insufficient sequence length"
    
    # GRU with MC Dropout
    gru_model = models.get('gru')
    if gru_model is not None and TF_AVAILABLE:
        try:
            gru_forecast = []
            gru_std = []
            current_seq = X_scaled[-seq_length:].copy()
            
            for _ in range(forecast_days):
                preds = []
                for _ in range(n_samples):
                    pred = gru_model(current_seq.reshape(1, seq_length, -1), training=True)
                    preds.append(pred.numpy()[0, 0])
                
                mean_pred = np.mean(preds)
                std_pred = np.std(preds)
                
                pred_price = scaler_y.inverse_transform([[mean_pred]])[0, 0]
                std_price = std_pred * (scaler_y.data_max_ - scaler_y.data_min_)[0]
                
                gru_forecast.append(float(pred_price))
                gru_std.append(float(std_price))
                
                current_seq = np.roll(current_seq, -1, axis=0)
                new_features = current_seq[-2].copy()
                new_features[3] = mean_pred
                current_seq[-1] = new_features
            
            predictions['GRU'] = {
                'next_day': gru_forecast[0],
                'forecast': gru_forecast,
                'std': gru_std,
                'ci_lower': [f - 1.96*s for f, s in zip(gru_forecast, gru_std)],
                'ci_upper': [f + 1.96*s for f, s in zip(gru_forecast, gru_std)],
                'current_price': current_price,
                'model_type': 'advanced'
            }
        except Exception as e:
            st.warning(f"GRU error: {e}")
    
    # TFT with MC Dropout
    tft_model = models.get('tft')
    if tft_model is not None and TF_AVAILABLE:
        try:
            tft_forecast = []
            tft_std = []
            current_seq = X_scaled[-seq_length:].copy()
            
            for _ in range(forecast_days):
                preds = []
                for _ in range(n_samples):
                    pred = tft_model(current_seq.reshape(1, seq_length, -1), training=True)
                    preds.append(pred.numpy()[0, 0])
                
                mean_pred = np.mean(preds)
                std_pred = np.std(preds)
                
                pred_price = scaler_y.inverse_transform([[mean_pred]])[0, 0]
                std_price = std_pred * (scaler_y.data_max_ - scaler_y.data_min_)[0]
                
                tft_forecast.append(float(pred_price))
                tft_std.append(float(std_price))
                
                current_seq = np.roll(current_seq, -1, axis=0)
                new_features = current_seq[-2].copy()
                new_features[3] = mean_pred
                current_seq[-1] = new_features
            
            predictions['TFT'] = {
                'next_day': tft_forecast[0],
                'forecast': tft_forecast,
                'std': tft_std,
                'ci_lower': [f - 1.96*s for f, s in zip(tft_forecast, tft_std)],
                'ci_upper': [f + 1.96*s for f, s in zip(tft_forecast, tft_std)],
                'current_price': current_price,
                'model_type': 'advanced'
            }
        except Exception as e:
            st.warning(f"TFT error: {e}")
    
    return predictions, None

def make_ensemble_prediction(classic_preds, advanced_preds, regime, regime_confidence):
    """Create regime-aware ensemble prediction"""
    all_preds = {}
    if classic_preds:
        all_preds.update(classic_preds)
    if advanced_preds:
        all_preds.update(advanced_preds)
    
    if not all_preds:
        return None
    
    # Regime-based weights
    weights = {
        'bull': {'Linear Regression': 0.8, 'LSTM': 1.0, 'GRU': 1.2, 'TFT': 1.1},
        'bear': {'Linear Regression': 0.9, 'LSTM': 1.1, 'GRU': 1.0, 'TFT': 1.3},
        'sideways': {'Linear Regression': 1.0, 'LSTM': 1.0, 'GRU': 1.0, 'TFT': 1.1}
    }
    
    regime_weights = weights.get(regime, weights['sideways'])
    
    # Calculate ensemble forecast
    forecast_days = max(len(p.get('forecast', [])) for p in all_preds.values())
    ensemble_forecast = []
    ensemble_std = []
    
    for day in range(forecast_days):
        weighted_sum = 0
        total_weight = 0
        day_preds = []
        
        for model_name, pred_data in all_preds.items():
            forecast = pred_data.get('forecast', [])
            if day < len(forecast):
                weight = regime_weights.get(model_name, 1.0)
                weighted_sum += forecast[day] * weight
                total_weight += weight
                day_preds.append(forecast[day])
        
        if total_weight > 0:
            ensemble_forecast.append(weighted_sum / total_weight)
            ensemble_std.append(np.std(day_preds) if len(day_preds) > 1 else 0)
    
    current_price = list(all_preds.values())[0].get('current_price', 0)
    
    return {
        'next_day': ensemble_forecast[0] if ensemble_forecast else 0,
        'forecast': ensemble_forecast,
        'std': ensemble_std,
        'ci_lower': [f - 1.96*s for f, s in zip(ensemble_forecast, ensemble_std)],
        'ci_upper': [f + 1.96*s for f, s in zip(ensemble_forecast, ensemble_std)],
        'current_price': current_price,
        'model_type': 'ensemble',
        'regime': regime,
        'regime_confidence': regime_confidence
    }


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Load data and models
    nse_indexes, stocks_df = load_data()
    models = load_models()
    
    if nse_indexes is None:
        st.error("Failed to load data. Please check if data files exist.")
        return
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">📈 NSE Stock Forecasting Dashboard</h1>
        <p class="dashboard-subtitle">Enterprise-Grade Stock Analysis & Prediction Platform | Classic + Advanced ML Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        st.markdown("---")
        
        # Data Selection
        st.markdown("#### 📊 Data Selection")
        data_type = st.radio("Select Data Type", ["Index", "Stock"], horizontal=True)
        
        if data_type == "Index":
            available_options = nse_indexes['Index'].unique().tolist()
            selected = st.selectbox("Select Index", available_options, 
                                   index=available_options.index('NIFTY 50') if 'NIFTY 50' in available_options else 0)
            df = nse_indexes[nse_indexes['Index'] == selected].copy()
        else:
            available_options = stocks_df['Stock'].unique().tolist()
            selected = st.selectbox("Select Stock", available_options)
            df = stocks_df[stocks_df['Stock'] == selected].copy()
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        st.markdown("---")
        
        # Date Range
        st.markdown("#### 📅 Date Range")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=365), max_date),
            min_value=min_date, max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        
        st.markdown("---")
        
        # Forecast Settings
        st.markdown("#### 🔮 Forecast Settings")
        forecast_days = st.slider("Forecast Days", 1, 14, 7)
        
        st.markdown("##### Model Selection")
        use_classic = st.checkbox("Classic Models (LR, LSTM)", value=True)
        use_advanced = st.checkbox("Advanced Models (GRU, TFT)", value=True)
        use_ensemble = st.checkbox("Ensemble Prediction", value=True)
        
        st.markdown("---")
        
        # Model Status
        st.markdown("#### 🤖 Model Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classic:**")
            st.markdown(f"• LR: {'✅' if models.get('linear_regression') else '❌'}")
            st.markdown(f"• LSTM: {'✅' if models.get('lstm') else '❌'}")
        with col2:
            st.markdown("**Advanced:**")
            st.markdown(f"• GRU: {'✅' if models.get('gru') else '❌'}")
            st.markdown(f"• TFT: {'✅' if models.get('tft') else '❌'}")
        
        if models.get('_errors'):
            with st.expander("⚠️ Loading Issues"):
                for err in models['_errors']:
                    st.warning(err)
    
    # Main Content
    if len(df) == 0:
        st.warning("No data available for the selected criteria.")
        return
    
    # Add technical indicators
    df = create_features(df)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    price_change = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{latest['Close']:,.2f}</div>
            <div class="{'metric-delta-positive' if price_change >= 0 else 'metric-delta-negative'}">
                {'▲' if price_change >= 0 else '▼'} {abs(price_change):,.2f} ({price_change_pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day High</div>
            <div class="metric-value">₹{latest['High']:,.2f}</div>
            <div class="metric-delta-positive">Range: ₹{latest['High'] - latest['Low']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day Low</div>
            <div class="metric-value">₹{latest['Low']:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volume_change = ((latest['Volume'] - prev['Volume']) / prev['Volume'] * 100) if prev['Volume'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{latest['Volume']/1e6:,.2f}M</div>
            <div class="{'metric-delta-positive' if volume_change >= 0 else 'metric-delta-negative'}">
                {volume_change:+.1f}% vs prev
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        rsi_value = latest.get('RSI', 50)
        rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{rsi_value:.1f}</div>
            <div class="metric-label">{rsi_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price Analysis", "🔮 Classic Forecast", "🚀 Advanced Forecast", 
        "📊 Technical Analysis", "📋 Data Explorer"
    ])

    
    # TAB 1: Price Analysis
    with tab1:
        st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)
        fig = create_price_chart(df, f"{selected} Price Chart")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 📊 Period Performance")
            period_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            st.metric("Total Return", f"{period_return:.2f}%")
        with col2:
            st.markdown("##### 📈 Volatility")
            volatility = df['Returns'].std() if 'Returns' in df.columns else 0
            st.metric("Daily Volatility", f"{volatility:.2f}%")
        with col3:
            st.markdown("##### 📉 Drawdown")
            rolling_max = df['Close'].cummax()
            drawdown = ((df['Close'] - rolling_max) / rolling_max * 100).min()
            st.metric("Max Drawdown", f"{drawdown:.2f}%")
    
    # TAB 2: Classic Forecast
    with tab2:
        st.markdown('<div class="section-header">Classic Models Forecast</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <span class="model-badge badge-classic">Linear Regression</span>
            <span class="model-badge badge-classic">LSTM (Deep Learning)</span>
        </div>
        """, unsafe_allow_html=True)
        
        if selected == 'NIFTY 50' and use_classic:
            with st.spinner("Generating classic predictions..."):
                classic_preds, error = make_classic_predictions(df, models, forecast_days)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif classic_preds:
                st.markdown("### 📊 Next Day Predictions")
                cols = st.columns(len(classic_preds))
                
                for i, (model_name, pred_data) in enumerate(classic_preds.items()):
                    with cols[i]:
                        next_day_pred = pred_data['next_day']
                        current_price = pred_data['current_price']
                        change = ((next_day_pred - current_price) / current_price) * 100
                        
                        box_class = "prediction-box" if change >= 0 else "prediction-box prediction-box-bearish"
                        direction = "BULLISH 📈" if change >= 0 else "BEARISH 📉"
                        
                        st.markdown(f"""
                        <div class="{box_class}">
                            <h4 style="margin:0; color:white;">{model_name}</h4>
                            <h2 style="margin:10px 0; color:white;">₹{next_day_pred:,.2f}</h2>
                            <p style="margin:5px 0;">{direction}</p>
                            <p style="margin:0; font-size:0.9rem;">Change: {change:+.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### 📈 Forecast Charts")
                for model_name, pred_data in classic_preds.items():
                    forecast = pred_data['forecast']
                    last_date = df['Date'].iloc[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast), freq='B')
                    hist_df = df.tail(60)
                    
                    fig = create_prediction_chart(
                        hist_df['Date'].values, hist_df['Close'].values,
                        forecast_dates, forecast, model_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No classic predictions available.")
        else:
            st.info("🔮 Classic forecast available for NIFTY 50. Select NIFTY 50 and enable Classic Models.")

    
    # TAB 3: Advanced Forecast
    with tab3:
        st.markdown('<div class="section-header">Advanced Models Forecast</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <span class="model-badge badge-advanced">GRU (Bidirectional)</span>
            <span class="model-badge badge-advanced">Temporal Fusion Transformer</span>
            <span class="model-badge badge-ensemble">Regime-Aware Ensemble</span>
        </div>
        """, unsafe_allow_html=True)
        
        if selected == 'NIFTY 50' and use_advanced:
            # Detect market regime
            regime_detector = MarketRegimeDetector()
            regime, regime_confidence = regime_detector.detect_regime(df)
            
            # Display regime
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                regime_class = f"regime-{regime}"
                regime_icon = "🐂" if regime == 'bull' else "🐻" if regime == 'bear' else "➡️"
                st.markdown(f"""
                <div class="regime-indicator {regime_class}">
                    <h3 style="margin:0;">{regime_icon} Market Regime: {regime.upper()}</h3>
                    <p style="margin:5px 0;">Confidence: {regime_confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            with st.spinner("Generating advanced predictions with uncertainty estimation..."):
                advanced_preds, error = make_advanced_predictions(df, models, forecast_days)
                classic_preds_for_ensemble, _ = make_classic_predictions(df, models, forecast_days) if use_classic else (None, None)
            
            if error:
                st.error(f"❌ Error: {error}")
            elif advanced_preds:
                # Ensemble prediction
                ensemble_pred = None
                if use_ensemble:
                    ensemble_pred = make_ensemble_prediction(classic_preds_for_ensemble, advanced_preds, regime, regime_confidence)
                    if ensemble_pred:
                        advanced_preds['Ensemble'] = ensemble_pred
                
                # Next day predictions
                st.markdown("### 📊 Next Day Predictions with Uncertainty")
                cols = st.columns(len(advanced_preds))
                
                for i, (model_name, pred_data) in enumerate(advanced_preds.items()):
                    with cols[i]:
                        next_day_pred = pred_data['next_day']
                        current_price = pred_data['current_price']
                        change = ((next_day_pred - current_price) / current_price) * 100
                        std = pred_data.get('std', [0])[0] if pred_data.get('std') else 0
                        
                        if model_name == 'Ensemble':
                            box_class = "prediction-box prediction-box-neutral" if abs(change) < 1 else ("prediction-box" if change >= 0 else "prediction-box prediction-box-bearish")
                        else:
                            box_class = "prediction-box" if change >= 0 else "prediction-box prediction-box-bearish"
                        
                        direction = "BULLISH 📈" if change >= 0 else "BEARISH 📉"
                        
                        st.markdown(f"""
                        <div class="{box_class}">
                            <h4 style="margin:0; color:white;">{model_name}</h4>
                            <h2 style="margin:10px 0; color:white;">₹{next_day_pred:,.2f}</h2>
                            <p style="margin:5px 0;">{direction}</p>
                            <p style="margin:0; font-size:0.85rem;">Change: {change:+.2f}%</p>
                            {f'<p style="margin:0; font-size:0.8rem;">±₹{std:,.2f}</p>' if std > 0 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Trading Signal
                if ensemble_pred:
                    st.markdown("### 🎯 Trading Signal")
                    signal_gen = TradingSignalGenerator()
                    signal_result = signal_gen.generate_signal(
                        current_price=latest['Close'],
                        predicted_price=ensemble_pred['next_day'],
                        uncertainty=ensemble_pred.get('std', [0])[0] if ensemble_pred.get('std') else 0,
                        regime=regime,
                        regime_confidence=regime_confidence,
                        rsi=latest.get('RSI', 50),
                        macd_hist=latest.get('MACD_Hist', 0)
                    )
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        signal_class = f"signal-{signal_result['signal'].lower()}"
                        st.markdown(f"""
                        <div class="signal-box {signal_class}">
                            {signal_result['signal']} ({signal_result['confidence']:.0%} confidence)
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Reasoning:**")
                        for reason in signal_result['reasoning']:
                            st.markdown(f"• {reason}")
                
                # Forecast charts with confidence intervals
                st.markdown("### 📈 Forecast Charts with Uncertainty Bands")
                for model_name, pred_data in advanced_preds.items():
                    forecast = pred_data['forecast']
                    ci_lower = pred_data.get('ci_lower')
                    ci_upper = pred_data.get('ci_upper')
                    
                    last_date = df['Date'].iloc[-1]
                    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast), freq='B')
                    hist_df = df.tail(60)
                    
                    fig = create_prediction_chart(
                        hist_df['Date'].values, hist_df['Close'].values,
                        forecast_dates, forecast, model_name,
                        ci_lower=ci_lower, ci_upper=ci_upper
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison
                st.markdown("### 📊 Model Comparison")
                all_preds = {}
                if classic_preds_for_ensemble:
                    all_preds.update(classic_preds_for_ensemble)
                all_preds.update(advanced_preds)
                
                fig = create_ensemble_comparison_chart(all_preds, latest['Close'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.markdown("### 📋 Detailed Forecast Table")
                last_date = df['Date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='B')
                
                table_data = {'Date': forecast_dates.strftime('%Y-%m-%d')}
                for model_name, pred_data in advanced_preds.items():
                    forecast = pred_data.get('forecast', [])
                    table_data[f'{model_name} (₹)'] = [f"{p:,.2f}" for p in forecast[:forecast_days]]
                    if pred_data.get('std'):
                        table_data[f'{model_name} ±'] = [f"₹{s:,.2f}" for s in pred_data['std'][:forecast_days]]
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            else:
                st.warning("No advanced predictions available. Check if GRU/TFT models are loaded.")
        else:
            st.info("🚀 Advanced forecast available for NIFTY 50. Select NIFTY 50 and enable Advanced Models.")

    
    # TAB 4: Technical Analysis
    with tab4:
        st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)
        
        fig = create_technical_indicators_chart(df)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📊 Moving Averages")
            ma_data = {
                'Indicator': ['SMA 5', 'SMA 10', 'SMA 20', 'SMA 50'],
                'Value': [
                    f"₹{latest.get('SMA_5', 0):,.2f}",
                    f"₹{latest.get('SMA_10', 0):,.2f}",
                    f"₹{latest.get('SMA_20', 0):,.2f}",
                    f"₹{latest.get('SMA_50', 0):,.2f}"
                ],
                'Signal': [
                    '🟢 Above' if latest['Close'] > latest.get('SMA_5', 0) else '🔴 Below',
                    '🟢 Above' if latest['Close'] > latest.get('SMA_10', 0) else '🔴 Below',
                    '🟢 Above' if latest['Close'] > latest.get('SMA_20', 0) else '🔴 Below',
                    '🟢 Above' if latest['Close'] > latest.get('SMA_50', 0) else '🔴 Below'
                ]
            }
            st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("##### 📈 Momentum Indicators")
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            
            momentum_data = {
                'Indicator': ['RSI (14)', 'MACD', 'MACD Signal', 'Stoch K', 'Williams %R'],
                'Value': [
                    f"{rsi:.2f}", f"{macd:.2f}", f"{macd_signal:.2f}",
                    f"{latest.get('Stoch_K', 50):.2f}", f"{latest.get('Williams_R', -50):.2f}"
                ],
                'Signal': [
                    '🔴 Overbought' if rsi > 70 else '🟢 Oversold' if rsi < 30 else '⚪ Neutral',
                    '🟢 Bullish' if macd > macd_signal else '🔴 Bearish',
                    '🟢 Bullish' if macd > 0 else '🔴 Bearish',
                    '🔴 Overbought' if latest.get('Stoch_K', 50) > 80 else '🟢 Oversold' if latest.get('Stoch_K', 50) < 20 else '⚪ Neutral',
                    '🟢 Oversold' if latest.get('Williams_R', -50) < -80 else '🔴 Overbought' if latest.get('Williams_R', -50) > -20 else '⚪ Neutral'
                ]
            }
            st.dataframe(pd.DataFrame(momentum_data), use_container_width=True, hide_index=True)
        
        with col3:
            st.markdown("##### 📉 Volatility & Risk")
            vol_5 = latest.get('Volatility_5', 0)
            vol_20 = latest.get('Volatility_20', 0)
            atr_pct = latest.get('ATR_Pct', 0) * 100
            
            vol_data = {
                'Indicator': ['5-Day Vol', '20-Day Vol', 'BB Width', 'ATR %'],
                'Value': [f"{vol_5:.2f}%", f"{vol_20:.2f}%", f"{latest.get('BB_Width', 0):.4f}", f"{atr_pct:.2f}%"],
                'Status': [
                    '🔴 High' if vol_5 > 2 else '🟢 Low',
                    '🔴 High' if vol_20 > 2 else '🟢 Low',
                    '🔴 Wide' if latest.get('BB_Width', 0) > 0.1 else '🟢 Narrow',
                    '🔴 High' if atr_pct > 2 else '🟢 Low'
                ]
            }
            st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)
    
    # TAB 5: Data Explorer
    with tab5:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### 📋 Historical Data")
            display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'Returns' in df.columns:
                display_cols.append('Returns')
            
            st.dataframe(
                df[display_cols].tail(100).sort_values('Date', ascending=False),
                use_container_width=True, hide_index=True
            )
        
        with col2:
            st.markdown("##### 📊 Statistics")
            stats = df['Close'].describe()
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [f"{stats['count']:,.0f}", f"₹{stats['mean']:,.2f}", 
                         f"₹{stats['std']:,.2f}", f"₹{stats['min']:,.2f}",
                         f"₹{stats['25%']:,.2f}", f"₹{stats['50%']:,.2f}",
                         f"₹{stats['75%']:,.2f}", f"₹{stats['max']:,.2f}"]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            st.markdown("##### 📥 Export Data")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV", data=csv,
                file_name=f"{selected}_data.csv", mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem;">
        <p>📈 NSE Stock Forecasting Dashboard | Classic + Advanced ML Models</p>
        <p>Models: Linear Regression • LSTM • GRU • Temporal Fusion Transformer • Regime-Aware Ensemble</p>
        <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
