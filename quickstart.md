# 🚀 Quick Start Guide

Get your Alpaca Trading Bot running in 5 minutes!

## 📋 Prerequisites

- **Python 3.8+** 
- **Alpaca Markets Account** (free paper trading)
- **Groq API Key** (optional, free)

## ⚡ Lightning Setup

### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
git clone <repository-url>
cd trading_bot
chmod +x start.sh
./start.sh --paper
```

**Windows:**
```cmd
git clone <repository-url>
cd trading_bot
start.bat --paper
```

### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd trading_bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup configuration
cp .env.example .env
# Edit .env with your API keys

# 5. Start in paper trading mode
python main.py --paper
```

## 🔑 Required API Keys

### 1. Alpaca Markets (Required)
1. Visit [alpaca.markets](https://alpaca.markets/)
2. Create free account
3. Get API Key & Secret from dashboard
4. Add to `.env` file:
```bash
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 2. Groq AI (Optional)
1. Visit [console.groq.com](https://console.groq.com/)
2. Create free account (generous free tier)
3. Generate API key
4. Add to `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key
```

## 🎯 First Run

```bash
# Validate configuration
python main.py --validate-only

# Start paper trading
python main.py --paper --debug

# Trade specific symbols
python main.py --paper --symbols TQQQ SQQQ

# Check what it's doing
tail -f logs/trading_bot.log
```

## 📊 What You'll See

```
🤖 Alpaca Trading Bot with Groq AI Integration
==================================================
📊 Trading symbols: TQQQ, SQQQ, SPXL, SPXS
🔗 Alpaca URL: https://paper-api.alpaca.markets
🧠 Groq AI: Enabled (mixtral-8x7b-32768)
🎯 Max Positions: 3
💰 Position Size: 10.0%

================================================================================
TRADING BOT STATUS - 2024-01-01 10:30:00
================================================================================
Analysis Cycles: 10
Active Strategies: 2
Account Equity: $100,000.00
Active Positions: 1
Pending Orders: 0
```

## 🛠️ Common Commands

```bash
# Development
make setup              # Complete dev setup
make test              # Run tests
make lint              # Check code quality
make run-paper         # Start paper trading

# Docker
docker-compose up -d   # Run in Docker
docker-compose logs -f # View logs

# Configuration
python main.py --help  # See all options
```

## 🚨 Safety First

- **Always start with `--paper`** (paper trading)
- **Never risk more than you can afford to lose**
- **Test thoroughly before live trading**
- **Monitor your bot regularly**

## 📈 Next Steps

1. **Monitor Performance**: Check logs and account
2. **Adjust Settings**: Edit configuration in `.env`
3. **Add Strategies**: Extend `strategies/` folder
4. **Live Trading**: Remove `--paper` when ready

## 🆘 Need Help?

- **Check logs**: `tail -f logs/trading_bot.log`
- **Validate config**: `python main.py --validate-only`
- **Run tests**: `make test`
- **Read README**: Full documentation available

## 🎉 You're Ready!

Your bot is now analyzing markets and making paper trades. Monitor the logs to see it in action!

**Happy Trading!** 📈🤖
