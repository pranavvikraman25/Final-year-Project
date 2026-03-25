# 📈 NSE Stock Forecasting Platform

**Enterprise-Grade Stock Price Prediction System with Advanced Machine Learning**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-Proprietary-green)

---

## 🎯 Overview

A sophisticated stock market forecasting platform that combines classical machine learning and deep learning models to predict NSE (National Stock Exchange) stock prices. Features an interactive web dashboard with real-time predictions, technical analysis, and trading signal generation.

### Key Features

✨ **Multi-Model Predictions**
- Linear Regression (baseline)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Temporal Fusion Transformer (TFT)
- Ensemble forecasting with regime detection

📊 **Interactive Dashboard**
- Real-time price analysis with candlestick charts
- 1-14 day ahead forecasts
- Technical indicator visualization (15+ indicators)
- Trading signal generation (BUY/SELL/HOLD)
- Market regime detection (Bull/Bear/Sideways)

🎯 **Advanced Analytics**
- Uncertainty quantification with confidence intervals
- Monte Carlo Dropout for prediction uncertainty
- Regime-aware ensemble weighting
- Risk management recommendations

📈 **Technical Indicators**
- Moving Averages (SMA, EMA)
- MACD, RSI, Stochastic Oscillator
- Bollinger Bands, ATR
- Williams %R, Momentum indicators
- Volume analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Installation

1. **Clone or extract the project:**
   ```bash
   cd NSE-Stock-Forecasting
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models (if not provided):**
   ```bash
   python train_advanced_models.py
   ```

5. **Launch dashboard:**
   ```bash
   streamlit run app.py
   ```

6. **Open browser:**
   Navigate to `http://localhost:8501`

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup and configuration guide
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide for non-technical users

---

## 🏗️ Architecture

### Project Structure

```
NSE-Stock-Forecasting/
├── app.py                          # Streamlit dashboard (main application)
├── train_advanced_models.py        # Advanced model training pipeline
├── stock_prediction_colab.py       # Classic model training
├── advanced_models.py              # Model class definitions
├── eda_stock_analysis.py           # Exploratory data analysis
├── requirements.txt                # Python dependencies
│
├── archive/                        # Data directory
│   ├── nse_indexes.csv            # NSE index historical data
│   ├── stocks_df.csv              # Individual stock data
│   └── indexes_df.csv             # Index list
│
└── Model Files (generated):
    ├── nifty50_lstm_model.h5      # LSTM model weights
    ├── gru_model.h5               # GRU model weights
    ├── tft_model.h5               # TFT model weights
    ├── linear_regression_model.pkl # Linear model
    └── *.pkl                       # Scalers and configuration
```

### Technology Stack

**Core ML/AI:**
- TensorFlow 2.13+ (Deep Learning)
- scikit-learn 1.3+ (Classical ML)
- NumPy, pandas (Data Processing)

**Visualization:**
- Streamlit 1.28+ (Web Dashboard)
- Plotly 5.18+ (Interactive Charts)
- Matplotlib, Seaborn (Static Plots)

**Models:**
- LSTM: 3-layer bidirectional architecture
- GRU: Faster alternative with similar performance
- TFT: Attention-based temporal model
- Ensemble: Regime-aware weighted combination

---

## 💡 Usage

### Dashboard Tabs

**1. 📊 Price Analysis**
- Current price and daily metrics
- Candlestick charts with volume
- Period performance analysis
- Volatility and drawdown metrics

**2. 🔮 Classic Forecast**
- Linear Regression predictions
- LSTM forecasts
- 7-day forecast visualization
- Model performance metrics

**3. 🚀 Advanced Forecast**
- GRU and TFT predictions
- Uncertainty bands (confidence intervals)
- Market regime indicator
- Trading signals with confidence levels
- Risk management recommendations

**4. 📈 Technical Analysis**
- 15+ technical indicators
- Overbought/oversold signals
- Momentum and volatility metrics
- Trend analysis

**5. 🔍 Data Explorer**
- Historical data table
- Statistical summaries
- CSV export functionality

### Understanding Predictions

**Signal Types:**
- 🟢 **BUY**: Model predicts price increase (confidence > 60%)
- 🔴 **SELL**: Model predicts price decrease (confidence > 60%)
- 🟡 **HOLD**: Uncertain conditions (confidence < 60%)

**Market Regimes:**
- 🐂 **Bull**: Uptrend, positive momentum
- 🐻 **Bear**: Downtrend, negative momentum
- ➡️ **Sideways**: Range-bound, low momentum

**Confidence Levels:**
- 80-100%: Very High (strong signal)
- 60-80%: High (reliable signal)
- 40-60%: Medium (uncertain)
- 0-40%: Low (wait for clarity)

---

## 🔧 Configuration

### Model Training Parameters

**Classic Models:**
- Sequence Length: 60 days
- Features: 27 technical indicators
- Train/Test Split: 80/20
- LSTM Architecture: 128 → 64 → 32 units

**Advanced Models:**
- Sequence Length: 60 days
- Features: 32 technical indicators
- GRU: Bidirectional with MC Dropout
- TFT: Multi-head attention + GRU
- Ensemble: Regime-aware weighting

### Customization

Edit configuration in respective training scripts:
- `stock_prediction_colab.py` - Classic models
- `train_advanced_models.py` - Advanced models
- `advanced_models.py` - Model architectures

---

## 📊 Model Performance

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

### Typical Performance (NIFTY 50)

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | ~150 | ~120 | 0.85 |
| LSTM | ~100 | ~80 | 0.92 |
| GRU | ~95 | ~75 | 0.93 |
| TFT | ~90 | ~70 | 0.94 |
| Ensemble | ~85 | ~65 | 0.95 |

*Performance varies based on market conditions and data quality*

---

## 🔄 Data Updates

### Updating Market Data

1. Obtain latest CSV files with same structure
2. Replace files in `archive/` folder
3. Restart dashboard
4. (Optional) Retrain models for improved accuracy

### Data Requirements

**Required Columns:**
- Date (datetime format)
- Open, High, Low, Close (float)
- Volume (integer)
- Index/Stock (string identifier)

**Minimum Data:**
- 60 trading days for sequence-based models
- No missing values in OHLC columns

---

## 🛠️ Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Retrain models
python train_advanced_models.py
```

**Port already in use:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**Memory errors:**
- Close other applications
- Reduce batch size in training scripts
- Train models individually

**TensorFlow installation issues:**
```bash
# Use CPU version
pip install tensorflow-cpu

# For Apple Silicon
pip install tensorflow-macos tensorflow-metal
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

---

## 📈 Best Practices

### For Accurate Predictions

✅ Update data regularly (weekly recommended)
✅ Retrain models quarterly or after major market events
✅ Use ensemble predictions for important decisions
✅ Check confidence levels before acting on signals
✅ Monitor multiple timeframes
✅ Combine with fundamental analysis

### Risk Management

⚠️ **Important Disclaimers:**
- This is a prediction tool, not financial advice
- Past performance doesn't guarantee future results
- Always use stop-losses and position sizing
- Consult financial advisors for investment decisions
- Test strategies with paper trading first

---

## 🔐 Security & Privacy

- All processing done locally on your machine
- No data sent to external servers
- Models trained on your own data
- Full control over data and predictions

---

## 🚧 Maintenance

### Regular Tasks

**Weekly:**
- Update market data
- Review prediction accuracy

**Monthly:**
- Check for package updates
- Backup model files

**Quarterly:**
- Retrain models with new data
- Evaluate model performance
- Update feature engineering

### Backup Command

```bash
# Create backup
tar -czf backup_$(date +%Y%m%d).tar.gz *.h5 *.pkl archive/
```

---

## 📝 System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB
- OS: Windows 10, macOS 10.15, Ubuntu 20.04

### Recommended Requirements
- CPU: 8+ cores
- RAM: 16GB
- Storage: 10GB
- GPU: NVIDIA GPU with CUDA (optional, speeds up training)

---

## 🤝 Support

### Getting Help

1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions
2. Review [QUICK_START.md](QUICK_START.md) for basic usage
3. Check terminal output for error messages
4. Verify all dependencies are installed

### Reporting Issues

When reporting issues, provide:
- Operating system and version
- Python version (`python --version`)
- Error message (full text)
- Steps to reproduce

---

## 📄 License

Proprietary - All rights reserved

---

## 🎓 Credits

**Technologies Used:**
- TensorFlow/Keras - Deep Learning Framework
- Streamlit - Web Application Framework
- Plotly - Interactive Visualizations
- scikit-learn - Machine Learning Library
- pandas/NumPy - Data Processing

---

## 📞 Contact

For technical support or inquiries, contact your system administrator.

---

**Version:** 1.0  
**Last Updated:** March 2026  
**Status:** Production Ready

---

## 🎯 Quick Links

- [Setup Guide](SETUP_GUIDE.md) - Comprehensive installation and configuration
- [Quick Start](QUICK_START.md) - Fast setup for non-technical users
- [Requirements](requirements.txt) - Python dependencies

---

**Happy Forecasting! 📈**
