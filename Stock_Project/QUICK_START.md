# 🚀 Quick Start Guide - NSE Stock Forecasting Platform

**For Non-Technical Users**

---

## What You Need

- A computer (Windows, Mac, or Linux)
- Internet connection
- 30 minutes for setup

---

## Step-by-Step Setup

### 1️⃣ Install Python (5 minutes)

**Windows Users:**
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.10"
3. Run the installer
4. ⚠️ **IMPORTANT:** Check the box "Add Python to PATH"
5. Click "Install Now"

**Mac Users:**
1. Open Terminal (search for "Terminal" in Spotlight)
2. Copy and paste this command:
   ```bash
   brew install python@3.10
   ```
3. Press Enter

### 2️⃣ Extract Project Files (1 minute)

1. Extract the ZIP file you received
2. Remember the folder location (e.g., Desktop/NSE-Stock-Forecasting)

### 3️⃣ Open Terminal/Command Prompt (1 minute)

**Windows:**
- Press `Windows + R`
- Type `cmd` and press Enter

**Mac:**
- Press `Command + Space`
- Type "Terminal" and press Enter

### 4️⃣ Navigate to Project Folder (1 minute)

Type this command (replace with your actual path):

**Windows:**
```bash
cd C:\Users\YourName\Desktop\NSE-Stock-Forecasting
```

**Mac:**
```bash
cd ~/Desktop/NSE-Stock-Forecasting
```

### 5️⃣ Create Virtual Environment (2 minutes)

Copy and paste these commands one by one:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal.

### 6️⃣ Install Required Software (10 minutes)

Copy and paste this command:

```bash
pip install -r requirements.txt
```

Wait for it to finish (you'll see lots of text scrolling).

### 7️⃣ Train Models (Optional - 30 minutes)

If models are not included, run:

```bash
python train_advanced_models.py
```

Wait for "Training complete!" message.

### 8️⃣ Start the Dashboard (1 minute)

Copy and paste this command:

```bash
streamlit run app.py
```

Your browser should open automatically!

---

## Using the Dashboard

### Main Features

**📊 Price Analysis Tab**
- View current stock prices
- See candlestick charts
- Check daily performance

**🔮 Classic Forecast Tab**
- Get basic predictions
- View 7-day forecast
- See trend direction

**🚀 Advanced Forecast Tab**
- Get sophisticated predictions
- See confidence levels
- Get BUY/SELL/HOLD signals

**📈 Technical Analysis Tab**
- View RSI, MACD, and other indicators
- Check overbought/oversold conditions

**🔍 Data Explorer Tab**
- Browse historical data
- Download data as CSV

### Understanding Signals

**🟢 BUY Signal**
- Model predicts price will go UP
- Higher confidence % = stronger signal
- Check "Confidence" percentage

**🔴 SELL Signal**
- Model predicts price will go DOWN
- Higher confidence % = stronger signal
- Consider stop-loss recommendations

**🟡 HOLD Signal**
- Uncertain market conditions
- Wait for clearer signals
- Monitor technical indicators

### Market Regime

**🐂 Bull Market**
- Prices trending upward
- Positive momentum
- Good time for buying

**🐻 Bear Market**
- Prices trending downward
- Negative momentum
- Consider selling or waiting

**➡️ Sideways Market**
- Prices moving in range
- Low momentum
- Wait for breakout

---

## Daily Usage

### Every Morning Routine

1. Open Terminal/Command Prompt
2. Navigate to project folder:
   ```bash
   cd path/to/NSE-Stock-Forecasting
   ```
3. Activate environment:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Mac
   source venv/bin/activate
   ```
4. Start dashboard:
   ```bash
   streamlit run app.py
   ```
5. Check predictions and signals

### Stopping the Dashboard

- Press `Ctrl + C` in the terminal
- Close the terminal window

---

## Common Issues & Solutions

### ❌ "Python not found"
**Solution:** Reinstall Python and check "Add to PATH"

### ❌ "Module not found"
**Solution:** Run `pip install -r requirements.txt` again

### ❌ "Port already in use"
**Solution:** Close other terminal windows or use:
```bash
streamlit run app.py --server.port 8502
```

### ❌ Dashboard shows blank page
**Solution:** 
1. Press `Ctrl + C` to stop
2. Run `streamlit run app.py` again
3. Try a different browser

### ❌ Models not loading
**Solution:** Run training script:
```bash
python train_advanced_models.py
```

---

## Tips for Best Results

✅ **Update data weekly** - More recent data = better predictions

✅ **Check multiple indicators** - Don't rely on one signal alone

✅ **Use confidence levels** - Higher confidence = more reliable

✅ **Monitor market regime** - Adjust strategy based on Bull/Bear/Sideways

✅ **Set stop-losses** - Follow risk management recommendations

✅ **Compare models** - Check both Classic and Advanced forecasts

⚠️ **Remember:** This is a prediction tool, not financial advice. Always do your own research and consult financial advisors.

---

## Getting Help

### If something doesn't work:

1. **Check the error message** in terminal
2. **Read the SETUP_GUIDE.md** for detailed troubleshooting
3. **Take a screenshot** of the error
4. **Note what you were doing** when error occurred

### Information to provide when asking for help:

- Operating system (Windows/Mac/Linux)
- Python version: Run `python --version`
- Error message (copy from terminal)
- What step you're on

---

## Updating Data

### To use latest market data:

1. Get updated CSV files
2. Replace files in `archive/` folder:
   - nse_indexes.csv
   - stocks_df.csv
3. Restart the dashboard
4. (Optional) Retrain models for better accuracy

---

## Keyboard Shortcuts

**In Dashboard:**
- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show keyboard shortcuts

**In Terminal:**
- `Ctrl + C` - Stop the dashboard
- `Up Arrow` - Previous command
- `Tab` - Auto-complete

---

## Video Tutorial Links

(Add links to video tutorials if available)

- [ ] Installation walkthrough
- [ ] First-time setup
- [ ] Using the dashboard
- [ ] Interpreting signals
- [ ] Updating data

---

## Checklist for First-Time Setup

- [ ] Python installed
- [ ] Project files extracted
- [ ] Terminal opened
- [ ] Navigated to project folder
- [ ] Virtual environment created
- [ ] Virtual environment activated (see `(venv)` in terminal)
- [ ] Dependencies installed
- [ ] Models trained or available
- [ ] Dashboard started successfully
- [ ] Can see dashboard in browser
- [ ] Can select NIFTY 50 and see data
- [ ] Can view predictions

---

## Next Steps After Setup

1. **Explore the dashboard** - Click through all tabs
2. **Try different date ranges** - See how predictions change
3. **Compare models** - Check Classic vs Advanced forecasts
4. **Export data** - Download CSV from Data Explorer
5. **Set up daily routine** - Make it part of your morning analysis

---

**Congratulations! You're ready to use the NSE Stock Forecasting Platform! 🎉**

For detailed technical information, see SETUP_GUIDE.md
