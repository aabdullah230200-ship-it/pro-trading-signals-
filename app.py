from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# ============================================================
# FREE PRICE DATA SOURCES (NO API KEY NEEDED)
# ============================================================

def get_binance_klines(symbol, interval='1h', limit=500):
    """Fetch OHLCV from Binance - completely free, no key needed"""
    try:
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_vol','trades','taker_buy_base',
            'taker_buy_quote','ignore'
        ])
        df['open']   = df['open'].astype(float)
        df['high']   = df['high'].astype(float)
        df['low']    = df['low'].astype(float)
        df['close']  = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Binance error: {e}")
        return None

def get_yahoo_klines(symbol, interval='1h', period='60d'):
    """Fetch OHLCV from Yahoo Finance - free"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'}, inplace=True)
        return df
    except Exception as e:
        print(f"Yahoo error: {e}")
        return None

def get_price_data(pair, interval='1h'):
    """Route to correct data source based on pair"""
    binance_map = {
        'BTCUSD': 'BTCUSDT',
        'XAUUSD': 'XAUUSDT',
        'ETHUSD': 'ETHUSDT',
    }
    yahoo_map = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'NZDUSD': 'NZDUSD=X',
        'XAUUSD': 'GC=F',
        'BTCUSD': 'BTC-USD',
    }
    interval_yahoo = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '1h', '1d': '1d'
    }
    # Try Binance first for crypto/gold
    if pair in binance_map:
        df = get_binance_klines(binance_map[pair], interval)
        if df is not None and len(df) > 50:
            return df
    # Fallback to Yahoo Finance
    if pair in yahoo_map:
        yf_interval = interval_yahoo.get(interval, '1h')
        period = '7d' if interval in ['1m', '5m', '15m'] else '60d'
        df = get_yahoo_klines(yahoo_map[pair], yf_interval, period)
        if df is not None and len(df) > 50:
            return df
    return None

# ============================================================
# TECHNICAL INDICATORS ENGINE
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_vwap(df):
    typical = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calc_bollinger(series, period=20, std_dev=2):
    sma = calc_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

def calc_pivot_points(df):
    """Classic Pivot Points from previous candle"""
    prev = df.iloc[-2]
    pivot = (prev['high'] + prev['low'] + prev['close']) / 3
    r1 = 2 * pivot - prev['low']
    r2 = pivot + (prev['high'] - prev['low'])
    r3 = prev['high'] + 2 * (pivot - prev['low'])
    s1 = 2 * pivot - prev['high']
    s2 = pivot - (prev['high'] - prev['low'])
    s3 = prev['low'] - 2 * (prev['high'] - pivot)
    return {
        'pivot': round(pivot, 5),
        'r1': round(r1, 5), 'r2': round(r2, 5), 'r3': round(r3, 5),
        's1': round(s1, 5), 's2': round(s2, 5), 's3': round(s3, 5)
    }

def detect_order_blocks(df, lookback=20):
    """Smart Money Concept - Order Block Detection"""
    blocks = []
    for i in range(lookback, len(df) - 1):
        candle = df.iloc[i]
        next_c = df.iloc[i + 1]
        body = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        if candle_range == 0:
            continue
        body_ratio = body / candle_range
        # Bullish OB: bearish candle followed by strong bullish impulse
        if (candle['close'] < candle['open'] and
                next_c['close'] > next_c['open'] and
                body_ratio > 0.5 and
                (next_c['close'] - next_c['open']) > body * 1.5):
            blocks.append({
                'type': 'bullish_ob',
                'high': round(float(candle['high']), 5),
                'low': round(float(candle['low']), 5),
                'index': i
            })
        # Bearish OB: bullish candle followed by strong bearish impulse
        if (candle['close'] > candle['open'] and
                next_c['close'] < next_c['open'] and
                body_ratio > 0.5 and
                (next_c['open'] - next_c['close']) > body * 1.5):
            blocks.append({
                'type': 'bearish_ob',
                'high': round(float(candle['high']), 5),
                'low': round(float(candle['low']), 5),
                'index': i
            })
    return blocks[-5:] if blocks else []

def snake_strategy(df):
    """
    SNAKE STRATEGY - استراتيجية الثعبان
    EMA9 x EMA21 crossover + RSI confirmation + EMA50 trend filter
    """
    ema9  = calc_ema(df['close'], 9)
    ema21 = calc_ema(df['close'], 21)
    ema50 = calc_ema(df['close'], 50)
    rsi   = calc_rsi(df['close'], 14)
    signals = []
    for i in range(22, len(df)):
        prev_above = ema9.iloc[i - 1] > ema21.iloc[i - 1]
        curr_above = ema9.iloc[i] > ema21.iloc[i]
        if not prev_above and curr_above:
            strength = 'strong' if (rsi.iloc[i] > 45 and df['close'].iloc[i] > ema50.iloc[i]) else 'weak'
            signals.append({'index': i, 'signal': 'SNAKE_BUY', 'strength': strength})
        elif prev_above and not curr_above:
            strength = 'strong' if (rsi.iloc[i] < 55 and df['close'].iloc[i] < ema50.iloc[i]) else 'weak'
            signals.append({'index': i, 'signal': 'SNAKE_SELL', 'strength': strength})
    return signals

def calc_supertrend(df, period=10, multiplier=3):
    atr = calc_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1] if not pd.isna(direction.iloc[i - 1]) else 0
        supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return supertrend, direction

def detect_market_structure(df):
    """Detect Higher Highs / Lower Lows for trend structure"""
    highs = df['high'].values
    lows  = df['low'].values
    n = len(highs)
    if n < 10:
        return 'UNKNOWN'
    recent_highs = highs[-10:]
    recent_lows  = lows[-10:]
    hh = recent_highs[-1] > recent_highs[-5]
    hl = recent_lows[-1]  > recent_lows[-5]
    lh = recent_highs[-1] < recent_highs[-5]
    ll = recent_lows[-1]  < recent_lows[-5]
    if hh and hl:
        return 'UPTREND (HH+HL)'
    elif lh and ll:
        return 'DOWNTREND (LH+LL)'
    else:
        return 'RANGING'

def calc_stochastic(df, k_period=14, d_period=3):
    low_min  = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return float(k.iloc[-1]), float(d.iloc[-1])

# ============================================================
# MAIN SIGNAL ENGINE
# ============================================================

def generate_signal(pair, interval='1h'):
    df = get_price_data(pair, interval)
    if df is None or len(df) < 50:
        return {'error': f'Could not fetch data for {pair}. Please try again.'}

    # Ensure enough data
    df = df.tail(500).reset_index(drop=True)
    close = df['close']
    current_price = float(close.iloc[-1])

    # ===== INDICATORS =====
    ema9   = calc_ema(close, 9)
    ema21  = calc_ema(close, 21)
    ema50  = calc_ema(close, 50)
    ema200 = calc_ema(close, min(200, len(df)-1))
    rsi    = calc_rsi(close, 14)
    macd_line, macd_signal_line, macd_hist = calc_macd(close)
    atr    = calc_atr(df, 14)
    vwap   = calc_vwap(df)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    supertrend, st_direction   = calc_supertrend(df)

    cur_rsi    = float(rsi.iloc[-1])
    cur_macd   = float(macd_line.iloc[-1])
    cur_macd_s = float(macd_signal_line.iloc[-1])
    cur_macd_h = float(macd_hist.iloc[-1])
    cur_atr    = float(atr.iloc[-1])
    cur_vwap   = float(vwap.iloc[-1])
    cur_ema200 = float(ema200.iloc[-1])
    cur_ema50  = float(ema50.iloc[-1])
    cur_ema21  = float(ema21.iloc[-1])
    cur_ema9   = float(ema9.iloc[-1])
    cur_bb_up  = float(bb_upper.iloc[-1])
    cur_bb_low = float(bb_lower.iloc[-1])
    cur_bb_mid = float(bb_mid.iloc[-1])
    cur_st_dir = int(st_direction.iloc[-1]) if not pd.isna(st_direction.iloc[-1]) else 0

    # Stochastic
    stoch_k, stoch_d = calc_stochastic(df)

    # Market structure
    market_structure = detect_market_structure(df)

    # ===== CURRENT CANDLE ANALYSIS =====
    cur_candle  = df.iloc[-1]
    prev_candle = df.iloc[-2]
    candle_body  = float(cur_candle['close']) - float(cur_candle['open'])
    candle_range = float(cur_candle['high'])  - float(cur_candle['low'])
    body_ratio   = abs(candle_body) / candle_range if candle_range > 0 else 0
    is_bullish_candle = candle_body > 0
    candle_strength = 'قوية' if body_ratio > 0.6 else ('متوسطة' if body_ratio > 0.35 else 'ضعيفة')

    upper_wick = float(cur_candle['high']) - max(float(cur_candle['open']), float(cur_candle['close']))
    lower_wick = min(float(cur_candle['open']), float(cur_candle['close'])) - float(cur_candle['low'])
    wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

    # ===== SCORING SYSTEM =====
    bull_score = 0
    bear_score = 0
    signals_list = []

    # 1. EMA200 Trend Filter (20 pts)
    if current_price > cur_ema200:
        bull_score += 20
        signals_list.append({'indicator': 'EMA200', 'signal': 'BULL', 'value': f'السعر فوق EMA200 ({round(cur_ema200, 4)})', 'icon': '📈'})
    else:
        bear_score += 20
        signals_list.append({'indicator': 'EMA200', 'signal': 'BEAR', 'value': f'السعر تحت EMA200 ({round(cur_ema200, 4)})', 'icon': '📉'})

    # 2. RSI (20 pts)
    if cur_rsi < 30:
        bull_score += 20
        signals_list.append({'indicator': 'RSI', 'signal': 'STRONG BULL', 'value': f'ذروة البيع: {round(cur_rsi, 1)}', 'icon': '🔥'})
    elif cur_rsi < 45:
        bull_score += 10
        signals_list.append({'indicator': 'RSI', 'signal': 'BULL', 'value': f'منطقة صعود: {round(cur_rsi, 1)}', 'icon': '📈'})
    elif cur_rsi > 70:
        bear_score += 20
        signals_list.append({'indicator': 'RSI', 'signal': 'STRONG BEAR', 'value': f'ذروة الشراء: {round(cur_rsi, 1)}', 'icon': '⚠️'})
    elif cur_rsi > 55:
        bear_score += 10
        signals_list.append({'indicator': 'RSI', 'signal': 'BEAR', 'value': f'منطقة هبوط: {round(cur_rsi, 1)}', 'icon': '📉'})
    else:
        signals_list.append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'value': f'محايد: {round(cur_rsi, 1)}', 'icon': '➖'})

    # 3. MACD (15 pts)
    if cur_macd > cur_macd_s and cur_macd_h > 0:
        bull_score += 15
        signals_list.append({'indicator': 'MACD', 'signal': 'BULL', 'value': f'تقاطع صعودي، hist: {round(cur_macd_h, 5)}', 'icon': '📈'})
    elif cur_macd < cur_macd_s and cur_macd_h < 0:
        bear_score += 15
        signals_list.append({'indicator': 'MACD', 'signal': 'BEAR', 'value': f'تقاطع هبوطي، hist: {round(cur_macd_h, 5)}', 'icon': '📉'})
    else:
        signals_list.append({'indicator': 'MACD', 'signal': 'NEUTRAL', 'value': f'في التقاطع، hist: {round(cur_macd_h, 5)}', 'icon': '➖'})

    # 4. VWAP (10 pts)
    if current_price > cur_vwap:
        bull_score += 10
        signals_list.append({'indicator': 'VWAP', 'signal': 'BULL', 'value': f'فوق VWAP ({round(cur_vwap, 4)})', 'icon': '📈'})
    else:
        bear_score += 10
        signals_list.append({'indicator': 'VWAP', 'signal': 'BEAR', 'value': f'تحت VWAP ({round(cur_vwap, 4)})', 'icon': '📉'})

    # 5. EMA Stack (15 pts)
    if cur_ema9 > cur_ema21 > cur_ema50:
        bull_score += 15
        signals_list.append({'indicator': 'EMA Stack', 'signal': 'STRONG BULL', 'value': 'EMA9 > EMA21 > EMA50', 'icon': '🚀'})
    elif cur_ema9 < cur_ema21 < cur_ema50:
        bear_score += 15
        signals_list.append({'indicator': 'EMA Stack', 'signal': 'STRONG BEAR', 'value': 'EMA9 < EMA21 < EMA50', 'icon': '💥'})
    else:
        signals_list.append({'indicator': 'EMA Stack', 'signal': 'NEUTRAL', 'value': 'تباين في المتوسطات', 'icon': '➖'})

    # 6. Bollinger Bands (10 pts)
    bb_range = cur_bb_up - cur_bb_low
    bb_pos = (current_price - cur_bb_low) / bb_range if bb_range > 0 else 0.5
    if current_price < cur_bb_low:
        bull_score += 10
        signals_list.append({'indicator': 'Bollinger', 'signal': 'BULL', 'value': 'تحت الشريط السفلي - ارتداد محتمل', 'icon': '🔄'})
    elif current_price > cur_bb_up:
        bear_score += 10
        signals_list.append({'indicator': 'Bollinger', 'signal': 'BEAR', 'value': 'فوق الشريط العلوي - تصحيح محتمل', 'icon': '🔄'})
    else:
        signals_list.append({'indicator': 'Bollinger', 'signal': 'NEUTRAL', 'value': f'موضع BB: {round(bb_pos*100, 1)}%', 'icon': '➖'})

    # 7. Supertrend (10 pts)
    if cur_st_dir == 1:
        bull_score += 10
        signals_list.append({'indicator': 'Supertrend', 'signal': 'BULL', 'value': 'اتجاه صعودي مؤكد', 'icon': '✅'})
    elif cur_st_dir == -1:
        bear_score += 10
        signals_list.append({'indicator': 'Supertrend', 'signal': 'BEAR', 'value': 'اتجاه هبوطي مؤكد', 'icon': '❌'})

    # 8. Stochastic (bonus)
    if stoch_k < 20 and stoch_k > stoch_d:
        bull_score += 5
        signals_list.append({'indicator': 'Stochastic', 'signal': 'BULL', 'value': f'K={round(stoch_k,1)} تشبع بيع+تقاطع', 'icon': '📈'})
    elif stoch_k > 80 and stoch_k < stoch_d:
        bear_score += 5
        signals_list.append({'indicator': 'Stochastic', 'signal': 'BEAR', 'value': f'K={round(stoch_k,1)} تشبع شراء+تقاطع', 'icon': '📉'})
    else:
        signals_list.append({'indicator': 'Stochastic', 'signal': 'NEUTRAL', 'value': f'K={round(stoch_k,1)} D={round(stoch_d,1)}', 'icon': '➖'})

    # ===== FINAL SIGNAL =====
    total = bull_score + bear_score
    bull_pct = (bull_score / total * 100) if total > 0 else 50
    bear_pct = (bear_score / total * 100) if total > 0 else 50

    if bull_score >= bear_score + 20:
        main_signal = 'BUY'
        confidence = min(95, 50 + (bull_score - bear_score))
    elif bear_score >= bull_score + 20:
        main_signal = 'SELL'
        confidence = min(95, 50 + (bear_score - bull_score))
    else:
        main_signal = 'HOLD'
        confidence = 50

    # ===== ENTRY / SL / TP (ATR-based) =====
    atr_val = cur_atr
    if main_signal == 'BUY':
        entry = current_price
        sl    = round(entry - 1.5 * atr_val, 5)
        tp1   = round(entry + 1.5 * atr_val, 5)
        tp2   = round(entry + 3.0 * atr_val, 5)
        tp3   = round(entry + 4.5 * atr_val, 5)
        rr    = round((tp1 - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0
    elif main_signal == 'SELL':
        entry = current_price
        sl    = round(entry + 1.5 * atr_val, 5)
        tp1   = round(entry - 1.5 * atr_val, 5)
        tp2   = round(entry - 3.0 * atr_val, 5)
        tp3   = round(entry - 4.5 * atr_val, 5)
        rr    = round((entry - tp1) / (sl - entry), 2) if (sl - entry) != 0 else 0
    else:
        entry = current_price
        sl    = round(current_price - atr_val, 5)
        tp1   = round(current_price + atr_val, 5)
        tp2   = round(current_price + 2 * atr_val, 5)
        tp3   = round(current_price + 3 * atr_val, 5)
        rr    = 1.0

    # ===== CANDLE PATTERN =====
    if body_ratio < 0.1:
        candle_pattern = 'Doji (تردد)'
    elif body_ratio > 0.7:
        candle_pattern = 'Marubozu - قوي جداً'
    elif wick_ratio > 0.6 and is_bullish_candle:
        candle_pattern = 'Hammer (ارتداد صعودي)'
    elif upper_wick > lower_wick * 2 and not is_bullish_candle:
        candle_pattern = 'Shooting Star (ارتداد هبوطي)'
    elif body_ratio > 0.4:
        candle_pattern = 'Standard ' + ('Bullish' if is_bullish_candle else 'Bearish')
    else:
        candle_pattern = 'Spinning Top (تردد)'

    # Next candle prediction
    if main_signal == 'BUY':
        next_candle = '🟢 الشمعة القادمة صعودية (استمرار)'
        next_prob = confidence
    elif main_signal == 'SELL':
        next_candle = '🔴 الشمعة القادمة هبوطية (استمرار)'
        next_prob = confidence
    else:
        next_candle = '⚪ غير محدد - انتظر تأكيداً'
        next_prob = 50

    # Pivots & Order Blocks & Snake
    try:
        pivots = calc_pivot_points(df)
    except:
        pivots = {}
    try:
        obs = detect_order_blocks(df)
    except:
        obs = []
    try:
        snake_sigs = snake_strategy(df)
        last_snake = snake_sigs[-1] if snake_sigs else None
    except:
        last_snake = None

    return {
        'pair':      pair,
        'interval':  interval,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'price':     round(current_price, 5),
        'signal':    main_signal,
        'confidence': round(confidence, 1),
        'bull_score': bull_score,
        'bear_score': bear_score,
        'bull_pct':  round(bull_pct, 1),
        'bear_pct':  round(bear_pct, 1),
        'market_structure': market_structure,
        'candle': {
            'direction': '🟢 صاعدة' if is_bullish_candle else '🔴 هابطة',
            'pattern':   candle_pattern,
            'strength':  candle_strength,
            'body_ratio': round(body_ratio * 100, 1),
            'open':  round(float(cur_candle['open']), 5),
            'high':  round(float(cur_candle['high']), 5),
            'low':   round(float(cur_candle['low']), 5),
            'close': round(float(cur_candle['close']), 5),
            'upper_wick': round(upper_wick, 5),
            'lower_wick': round(lower_wick, 5),
        },
        'next_candle':      next_candle,
        'next_candle_prob': next_prob,
        'trade': {
            'entry': round(entry, 5),
            'sl':    round(sl, 5),
            'tp1':   round(tp1, 5),
            'tp2':   round(tp2, 5),
            'tp3':   round(tp3, 5),
            'rr':    rr,
        },
        'indicators': {
            'ema9':         round(cur_ema9, 5),
            'ema21':        round(cur_ema21, 5),
            'ema50':        round(cur_ema50, 5),
            'ema200':       round(cur_ema200, 5),
            'rsi':          round(cur_rsi, 2),
            'stoch_k':      round(stoch_k, 2),
            'stoch_d':      round(stoch_d, 2),
            'macd':         round(cur_macd, 5),
            'macd_signal':  round(cur_macd_s, 5),
            'macd_hist':    round(cur_macd_h, 5),
            'atr':          round(cur_atr, 5),
            'vwap':         round(cur_vwap, 5),
            'bb_upper':     round(cur_bb_up, 5),
            'bb_mid':       round(cur_bb_mid, 5),
            'bb_lower':     round(cur_bb_low, 5),
            'supertrend_dir': cur_st_dir,
        },
        'signals_breakdown': signals_list,
        'pivot_points':  pivots,
        'order_blocks':  obs,
        'snake_signal':  last_snake,
        'mt5_symbol':    pair,
    }

# ============================================================
# FLASK ROUTES
# ============================================================

PAIRS     = ['XAUUSD', 'BTCUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
INTERVALS = ['5m', '15m', '30m', '1h', '4h', '1d']

@app.route('/')
def index():
    return render_template('index.html', pairs=PAIRS, intervals=INTERVALS)

@app.route('/api/signal')
def api_signal():
    pair     = request.args.get('pair', 'XAUUSD')
    interval = request.args.get('interval', '1h')
    result   = generate_signal(pair, interval)
    return jsonify(result)

@app.route('/api/multi_signal')
def api_multi_signal():
    interval = request.args.get('interval', '1h')
    results  = []
    for pair in ['XAUUSD', 'BTCUSD', 'EURUSD', 'GBPUSD', 'USDJPY']:
        try:
            r = generate_signal(pair, interval)
            if 'error' not in r:
                results.append(r)
        except Exception as e:
            results.append({'pair': pair, 'error': str(e)})
    return jsonify(results)

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
