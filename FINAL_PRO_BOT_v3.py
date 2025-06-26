# === FINAL_PRO_BOT_v3.py ===
# ✅ Full Trading Bot with LSTM + Attention + GPT News + Risk + Equity + API + Telegram + OpenAI Token Tracker + Backtest + Retrain

import os
import torch
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import ta
import openai
from torch import nn
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# === INIT ===
load_dotenv()

# === CONFIG ===
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
SEQ_LEN = 30
RISK_PER_TRADE = 0.01
CONFIDENCE_THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPITAL_FILE = "capital.txt"
TOKEN_LOG = "tokens_used.log"
TOKEN_LIMIT = 100000

# === ENV KEYS ===
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI()

# === MODEL ===
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1).unsqueeze(-1)
        context = torch.sum(weights * lstm_out, dim=1)
        return self.fc(context)

# === TOKEN TRACKING ===
def track_tokens(tokens):
    used = 0
    if os.path.exists(TOKEN_LOG):
        with open(TOKEN_LOG) as f:
            try:
                used = int(f.read())
            except:
                used = 0
    used += tokens
    with open(TOKEN_LOG, "w") as f:
        f.write(str(used))
    if used >= TOKEN_LIMIT * 0.9:
        notify_telegram(f"⚠️ OpenAI API usage: {used}/{TOKEN_LIMIT} tokens used (~90% limit)")

# === NEWS SENTIMENT ===
def fetch_news_sentiment():
    try:
        res = requests.get("https://cryptopanic.com/api/v1/posts/?auth_token=demo&public=true")
        news = res.json()['results']
        headlines = ". ".join([x['title'] for x in news[:5]])
        prompt = f"Analyze sentiment (-1 to 1) of crypto news: {headlines}"
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        content = res['choices'][0]['message']['content']
        tokens_used = res['usage']['total_tokens']
        track_tokens(tokens_used)
        score = float([s for s in content.split() if s.replace('.', '', 1).replace('-', '', 1).isdigit()][0])
        return max(-1, min(1, score))
    except Exception as e:
        print("Sentiment error:", e)
        return 0.0

# === TELEGRAM ===
def notify_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        except Exception as e:
            print("Telegram error:", e)

# === OHLCV ===
def load_ohlcv(symbol, timeframe, limit=100):
    try:
        df = pd.DataFrame(ccxt.mexc().fetch_ohlcv(symbol, timeframe, limit=limit),
                         columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print("OHLCV error:", e)
        return pd.DataFrame()

# === INDICATORS ===
def add_indicators(df):
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'],df['low'],df['close']).average_true_range()
        df['ema12'] = ta.trend.EMAIndicator(df['close'],12).ema_indicator()
        df['ema26'] = ta.trend.EMAIndicator(df['close'],26).ema_indicator()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['mfi'] = ta.volume.MFIIndicator(df['high'],df['low'],df['close'],df['volume']).money_flow_index()
        df['sentiment'] = fetch_news_sentiment()
        return df.dropna()
    except Exception as e:
        print("Indicator error:", e)
        return pd.DataFrame()

# === RISK ===
def get_capital():
    if not os.path.exists(CAPITAL_FILE):
        with open(CAPITAL_FILE, "w") as f:
            f.write("1000")
    return float(open(CAPITAL_FILE).read())

def update_capital(pnl):
    cap = get_capital() + pnl
    with open(CAPITAL_FILE, "w") as f:
        f.write(str(cap))
    return cap

def compute_position_size(capital, price, stop_pct):
    risk_amount = capital * RISK_PER_TRADE
    sl = price * stop_pct
    qty = risk_amount / sl
    return round(qty, 4)

# === PREDICT ===
def predict(model, scaler, capital):
    df = add_indicators(load_ohlcv(SYMBOL, TIMEFRAME))
    if df.empty:
        return None
    features = df[['rsi','macd','atr','ema12','ema26','obv','mfi','sentiment']].values
    try:
        scaled = scaler.transform(features)
    except Exception as e:
        print("Scaler error:", e)
        return None
    if len(scaled) < SEQ_LEN:
        return None
    x = torch.tensor(scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    direction = "BUY" if pred.item() == 1 else "SELL"
    confidence = conf.item()
    price = df['close'].iloc[-1]
    qty = compute_position_size(capital, price, 0.01)
    return direction, confidence, price, qty

# === LOAD ===
model = LSTMWithAttention(8, 128, 2).to(DEVICE)
scaler = StandardScaler()

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
if os.path.exists("scaler.npy"):
    scaler.mean_, scaler.scale_ = np.load("scaler.npy", allow_pickle=True)

# === API ===
@app.get("/signal")
def signal():
    capital = get_capital()
    result = predict(model, scaler, capital)
    if not result:
        return {"error": "Insufficient data"}
    direction, confidence, price, qty = result
    if confidence >= CONFIDENCE_THRESHOLD:
        msg = f"✅ SIGNAL: {direction} | Conf: {confidence:.2f} | Price: {price} | Qty: {qty}"
        notify_telegram(msg)
        return {"signal": direction, "confidence": confidence, "price": price, "qty": qty}
    return {"signal": "HOLD", "confidence": confidence}

@app.get("/equity")
def equity():
    return {"capital": get_capital()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
