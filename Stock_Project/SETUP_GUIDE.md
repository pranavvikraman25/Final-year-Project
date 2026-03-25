# 📈 NSE Stock Forecasting Platform - Setup Guide

**Version:** 1.0  
**Last Updated:** March 2026  
**Platform:** Windows, macOS, Linux

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Data Setup](#data-setup)
4. [Model Training](#model-training)
5. [Running the Dashboard](#running-the-dashboard)
6. [Troubleshooting](#troubleshooting)
7. [Usage Guide](#usage-guide)
8. [Maintenance](#maintenance)

---

## System Requirements

### Hardware Requirements
- **CPU:** Minimum 4 cores (8+ cores recommended for faster training)
- **RAM:** Minimum 8GB (16GB+ recommended)
- **Storage:** 5GB free space
- **GPU:** Optional (NVIDIA GPU with CUDA support speeds up training significantly)

### Software Requirements
- **Python:** Version 3.8 - 3.11 (3.10 recommended)
- **Operating System:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Internet Connection:** Required for initial package installation

---

## Installation Steps

### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python@3.10
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Step 2: Clone/Extract Project Files

Extract the project files to your desired location:
```
NSE-Stock-Forecasting/
├── app.py
├── train_advanced_models.py
├── stock_prediction_colab.py
├── advanced_models.py
├── eda_stock_analysis.py
├── requirements.txt
├── archive/
│   ├── nse_indexes.csv
│   ├── stocks_df.csv
│   └── indexes_df.csv
└── (model files will be generated)
```

### Step 3: Create Virtual Environment

**Navigate to project directory:**
```bash
cd path/to/NSE-Stock-Forecasting
```

**Create virtual environment:**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

### Step 4: Install Dependencies

**Install all required packages:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will install:**
- Streamlit (web dashboard)
- TensorFlow (deep learning models)
- scikit-learn (machine learning)
- pandas, numpy (data processing)
- plotly (interactive charts)
- matplotlib, seaborn (visualizations)

**Installation time:** 5-15 minutes depending on internet speed

**Verify installation:**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
```

---

## Data Setup

### Step 1: Verify Data Files

Ensure the following files exist in the `archive/` folder:

1. **nse_indexes.csv** - Historical NSE index data (NIFTY 50, NIFTY Bank, etc.)
2. **stocks_df.csv** - Individual stock data
3. **indexes_df.csv** - List of available indexes

**Required columns in nse_indexes.csv:**
- Date, Index, Open, High, Low, Close, Volume

### Step 2: Data Validation

Run the exploratory data analysis script to validate your data:

```bash
python eda_stock_analysis.py
```

**Expected output:**
- Data overview statistics
- Missing value analysis
- Date range information
- Sample visualizations saved as PNG files

**Generated files:**
- `eda_market_overview.png`
- `eda_nifty50_analysis.png`
- `eda_correlation_volatility.png`
- `eda_timeseries_analysis.png`
- `eda_index_comparison.png`

---

## Model Training

### Option 1: Train Classic Models (Linear Regression + LSTM)

**Run training script:**
```bash
python stock_prediction_colab.py
```

**Training time:** 10-30 minutes (depends on hardware)

**Generated files:**
- `nifty50_lstm_model.h5` - LSTM model weights
- `linear_regression_model.pkl` - Linear Regression model
- `scaler_X.pkl` - Feature scaler
- `scaler_y.pkl` - Target scaler

**Expected output:**
```
Training LSTM Model...
Epoch 1/50 - loss: 0.0234 - val_loss: 0.0189
...
Model saved successfully!
```

### Option 2: Train Advanced Models (GRU + TFT)

**Run advanced training script:**
```bash
python train_advanced_models.py
```

**Training time:** 20-60 minutes (depends on hardware)

**Generated files:**
- `gru_model.h5` - GRU model weights
- `tft_model.h5` - Temporal Fusion Transformer weights
- `advanced_scaler_X.pkl` - Advanced feature scaler
- `advanced_scaler_y.pkl` - Advanced target scaler
- `model_config.pkl` - Model configuration

**Expected output:**
```
[1] Loading data...
[2] Creating advanced features...
[3] Training GRU Model...
[4] Training TFT Model...
Models saved successfully!
```

### Option 3: Use Pre-trained Models

If pre-trained models are provided, skip training and verify files exist:
- ✓ nifty50_lstm_model.h5
- ✓ gru_model.h5
- ✓ tft_model.h5
- ✓ linear_regression_model.pkl
- ✓ All scaler files (.pkl)

---

## Running the Dashboard

### Step 1: Start the Application

**Ensure virtual environment is activated:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Launch Streamlit dashboard:**
```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

### Step 2: Access the Dashboard

1. **Automatic:** Browser should open automatically
2. **Manual:** Open browser and navigate to `http://localhost:8501`

### Step 3: Dashboard Overview

**Sidebar Controls:**
- Select Index/Stock (default: NIFTY 50)
- Choose date range
- Set forecast horizon (1-14 days)
- View model status

**Main Tabs:**
1. **📊 Price Analysis** - Candlestick charts, volume, performance metrics
2. **🔮 Classic Forecast** - Linear Regression and LSTM predictions
3. **🚀 Advanced Forecast** - GRU/TFT with uncertainty bands and trading signals
4. **📈 Technical Analysis** - 15+ technical indicators
5. **🔍 Data Explorer** - Historical data table with export

---

## Troubleshooting

### Issue 1: TensorFlow Installation Fails

**Error:** `Could not find a version that satisfies the requirement tensorflow`

**Solution:**
```bash
# For Windows/Linux
pip install tensorflow-cpu

# For macOS (Apple Silicon)
pip install tensorflow-macos tensorflow-metal
```

### Issue 2: Models Not Loading

**Error:** `Model file not found`

**Solution:**
1. Verify model files exist in project root
2. Re-run training scripts
3. Check file permissions

### Issue 3: Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8501 | xargs kill -9
```

### Issue 4: Memory Error During Training

**Error:** `MemoryError` or system freezes

**Solution:**
1. Close other applications
2. Reduce batch size in training scripts
3. Use smaller sequence length
4. Train models one at a time

### Issue 5: Data Loading Errors

**Error:** `FileNotFoundError: archive/nse_indexes.csv`

**Solution:**
1. Verify `archive/` folder exists
2. Check CSV file names match exactly
3. Ensure files are not corrupted

### Issue 6: Streamlit Shows Blank Page

**Solution:**
1. Clear browser cache
2. Try incognito/private mode
3. Check browser console for errors (F12)
4. Restart Streamlit server

---

## Usage Guide

### Making Predictions

1. **Select Data Source:**
   - Choose NIFTY 50 or other index from sidebar
   - Set date range (minimum 60 days required)

2. **View Current Analysis:**
   - Check "Price Analysis" tab for current market status
   - Review technical indicators in "Technical Analysis" tab

3. **Get Predictions:**
   - Navigate to "Classic Forecast" for baseline predictions
   - Check "Advanced Forecast" for sophisticated models with uncertainty

4. **Interpret Trading Signals:**
   - **BUY Signal:** Model predicts price increase
   - **SELL Signal:** Model predicts price decrease
   - **HOLD Signal:** Uncertain or sideways movement
   - **Confidence:** Higher percentage = stronger signal

5. **Understand Market Regime:**
   - **🐂 Bull Market:** Uptrend, momentum positive
   - **🐻 Bear Market:** Downtrend, momentum negative
   - **➡️ Sideways:** Range-bound, low momentum

### Exporting Data

1. Go to "Data Explorer" tab
2. Review historical data table
3. Click "Download Data as CSV" button
4. File saves to your Downloads folder

### Updating Data

**To use latest market data:**

1. Replace CSV files in `archive/` folder with updated data
2. Ensure same column structure
3. Restart Streamlit dashboard
4. Optionally retrain models for better accuracy

---

## Maintenance

### Regular Updates

**Weekly:**
- Update data files with latest market data
- Review prediction accuracy

**Monthly:**
- Retrain models with new data
- Check for package updates: `pip list --outdated`

**Quarterly:**
- Full model retraining
- Performance evaluation
- Feature engineering review

### Model Retraining

**When to retrain:**
- Prediction accuracy drops significantly
- Major market regime changes
- After 3-6 months of new data accumulation

**How to retrain:**
```bash
# Backup old models
mkdir models_backup
cp *.h5 *.pkl models_backup/

# Retrain
python train_advanced_models.py
python stock_prediction_colab.py

# Restart dashboard
streamlit run app.py
```

### Performance Monitoring

**Key metrics to track:**
- Prediction RMSE and MAE
- Signal accuracy (% correct predictions)
- Model confidence levels
- Execution time

### Backup Strategy

**Important files to backup:**
- All model files (*.h5, *.pkl)
- Data files (archive/*.csv)
- Configuration files
- Custom modifications to code

**Backup command:**
```bash
# Create backup
tar -czf backup_$(date +%Y%m%d).tar.gz *.h5 *.pkl archive/

# Or use zip
zip -r backup_$(date +%Y%m%d).zip *.h5 *.pkl archive/
```

---

## Advanced Configuration

### GPU Acceleration (Optional)

**For NVIDIA GPUs:**

1. Install CUDA Toolkit (11.8 or 12.x)
2. Install cuDNN
3. Install GPU-enabled TensorFlow:
   ```bash
   pip install tensorflow[and-cuda]
   ```

**Verify GPU:**
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

### Custom Port Configuration

**Edit Streamlit config:**

Create `.streamlit/config.toml`:
```toml
[server]
port = 8502
address = "0.0.0.0"

[theme]
primaryColor = "#0d9488"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

### Network Access

**To access from other devices on network:**

1. Find your IP address:
   ```bash
   # Windows
   ipconfig
   
   # macOS/Linux
   ifconfig
   ```

2. Run with network access:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

3. Access from other devices:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```

---

## Support & Contact

### Getting Help

1. **Check logs:** Terminal output shows detailed error messages
2. **Review documentation:** This guide covers common issues
3. **Check requirements:** Ensure all dependencies installed correctly

### System Information

**To report issues, provide:**
```bash
python --version
pip list
```

---

## Quick Start Checklist

- [ ] Python 3.8-3.11 installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data files in `archive/` folder
- [ ] Models trained or pre-trained models available
- [ ] Dashboard launches successfully (`streamlit run app.py`)
- [ ] Can access dashboard in browser
- [ ] Predictions working correctly

---

## Appendix

### File Structure Reference

```
NSE-Stock-Forecasting/
├── app.py                              # Main Streamlit dashboard
├── train_advanced_models.py            # Advanced model training
├── stock_prediction_colab.py           # Classic model training
├── advanced_models.py                  # Model class definitions
├── eda_stock_analysis.py               # Data analysis script
├── eda_visualizations.py               # Visualization utilities
├── fix_tft_model.py                    # TFT model utilities
├── requirements.txt                    # Python dependencies
├── SETUP_GUIDE.md                      # This file
│
├── archive/                            # Data directory
│   ├── nse_indexes.csv                # Index historical data
│   ├── stocks_df.csv                  # Stock historical data
│   └── indexes_df.csv                 # Index list
│
├── venv/                               # Virtual environment (created)
│
└── Model Files (generated after training):
    ├── nifty50_lstm_model.h5          # LSTM weights
    ├── gru_model.h5                   # GRU weights
    ├── tft_model.h5                   # TFT weights
    ├── linear_regression_model.pkl    # Linear model
    ├── scaler_X.pkl                   # Classic feature scaler
    ├── scaler_y.pkl                   # Classic target scaler
    ├── advanced_scaler_X.pkl          # Advanced feature scaler
    ├── advanced_scaler_y.pkl          # Advanced target scaler
    └── model_config.pkl               # Configuration
```

### Command Reference

```bash
# Environment Management
python -m venv venv                    # Create virtual environment
source venv/bin/activate               # Activate (macOS/Linux)
venv\Scripts\activate                  # Activate (Windows)
deactivate                             # Deactivate environment

# Installation
pip install -r requirements.txt        # Install dependencies
pip list                               # List installed packages
pip freeze > requirements.txt          # Save current packages

# Training
python eda_stock_analysis.py           # Run data analysis
python stock_prediction_colab.py       # Train classic models
python train_advanced_models.py        # Train advanced models

# Running
streamlit run app.py                   # Start dashboard
streamlit run app.py --server.port 8502  # Custom port

# Maintenance
pip install --upgrade streamlit        # Update package
pip list --outdated                    # Check for updates
```

---

**End of Setup Guide**

For questions or issues, refer to the troubleshooting section or contact technical support.
