"""
╔══════════════════════════════════════════════════════════╗
║   PRO TRADING SIGNALS - محرك الإشارات الاحترافي          ║
║   Multi-Timeframe | Smart Money | Snake Strategy         ║
║   Real prices: Binance (free) + Yahoo Finance (free)     ║
╚══════════════════════════════════════════════════════════╝
"""
from flask import Flask, render_template, jsonify, request
import requests, pandas as pd, numpy as np
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1.  مصادر الأسعار المجانية
# ─────────────────────────────────────────────
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

BINANCE_MAP = {"BTCUSD":"BTCUSDT","XAUUSD":"XAUUSDT","ETHUSD":"ETHUSDT"}
YAHOO_MAP   = {
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
    "USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X",
    "NZDUSD":"NZDUSD=X","XAUUSD":"GC=F","BTCUSD":"BTC-USD",
}
YAHOO_IV = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
            "1h":"1h","4h":"1h","1d":"1d"}
HIGHER_TF = {"1m":"5m","5m":"15m","15m":"1h","30m":"1h","1h":"4h","4h":"1d","1d":"1d"}

def _binance(symbol, interval, limit=600):
    try:
        r = requests.get(BINANCE_URL,
            params={"symbol":symbol,"interval":interval,"limit":limit},timeout=12)
        raw = r.json()
        if not isinstance(raw, list): return None
        df = pd.DataFrame(raw, columns=[
            "timestamp","open","high","low","close","volume",
            "ct","qv","n","tbb","tbq","ig"])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["timestamp","open","high","low","close","volume"]]
    except Exception as e:
        print(f"[Binance] {e}"); return None

def _yahoo(symbol, interval, period):
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={"datetime":"timestamp","date":"timestamp"}, inplace=True)
        df = df[["timestamp","open","high","low","close","volume"]]
        return df
    except Exception as e:
        print(f"[Yahoo] {e}"); return None

def fetch(pair, interval):
    period = "7d" if interval in ["1m","5m","15m"] else "90d"
    if pair in BINANCE_MAP:
        df = _binance(BINANCE_MAP[pair], interval)
        if df is not None and len(df) >= 60: return df
    if pair in YAHOO_MAP:
        df = _yahoo(YAHOO_MAP[pair], YAHOO_IV.get(interval,"1h"), period)
        if df is not None and len(df) >= 60: return df
    return None

# ─────────────────────────────────────────────
# 2.  حساب المؤشرات
# ─────────────────────────────────────────────
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l = (-d).clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100/(1 + g/(l+1e-10))

def macd(s, f=12, sl=26, sig=9):
    m = ema(s,f) - ema(s,sl)
    sg = ema(m, sig)
    return m, sg, m-sg

def atr(df, n=14):
    h,l,c = df.high, df.low, df.close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(span=n, adjust=False).mean()

def vwap(df):
    tp = (df.high + df.low + df.close)/3
    return (tp * df.volume).cumsum() / (df.volume.cumsum() + 1e-10)

def bollinger(s, n=20, k=2):
    m = sma(s,n); sd = s.rolling(n).std()
    return m+k*sd, m, m-k*sd

def stochastic(df, k=14, d=3):
    lo = df.low.rolling(k).min(); hi = df.high.rolling(k).max()
    K = 100*(df.close-lo)/(hi-lo+1e-10)
    return K, K.rolling(d).mean()

def supertrend(df, n=10, m=3):
    a = atr(df, n)
    hl2 = (df.high+df.low)/2
    ub = hl2 + m*a; lb = hl2 - m*a
    st  = pd.Series(np.nan, index=df.index)
    dir = pd.Series(0,     index=df.index)
    for i in range(1, len(df)):
        prev = st.iloc[i-1] if not np.isnan(st.iloc[i-1]) else lb.iloc[i]
        if df.close.iloc[i] > ub.iloc[i-1]: dir.iloc[i]=1
        elif df.close.iloc[i] < lb.iloc[i-1]: dir.iloc[i]=-1
        else: dir.iloc[i]=dir.iloc[i-1]
        st.iloc[i] = lb.iloc[i] if dir.iloc[i]==1 else ub.iloc[i]
    return st, dir

def pivot_points(df):
    p = df.iloc[-2]
    pp = (p.high+p.low+p.close)/3
    return dict(
        pivot=round(pp,5),
        r1=round(2*pp-p.low,5), r2=round(pp+(p.high-p.low),5),
        r3=round(p.high+2*(pp-p.low),5),
        s1=round(2*pp-p.high,5), s2=round(pp-(p.high-p.low),5),
        s3=round(p.low-2*(p.high-pp),5))

def order_blocks(df, lb=30):
    blocks=[]
    for i in range(lb, len(df)-1):
        c,nx = df.iloc[i], df.iloc[i+1]
        body = abs(c.close-c.open)
        rng  = c.high-c.low
        if rng<1e-10: continue
        br = body/rng
        if c.close<c.open and nx.close>nx.open and br>0.5 and (nx.close-nx.open)>body*1.5:
            blocks.append({"type":"bullish_ob","high":round(float(c.high),5),"low":round(float(c.low),5),"idx":i})
        if c.close>c.open and nx.close<nx.open and br>0.5 and (nx.open-nx.close)>body*1.5:
            blocks.append({"type":"bearish_ob","high":round(float(c.high),5),"low":round(float(c.low),5),"idx":i})
    return blocks[-6:] if blocks else []

def market_structure(df):
    h = df.high.values[-20:]; l = df.low.values[-20:]
    hh = h[-1]>h[-10]; hl = l[-1]>l[-10]
    lh = h[-1]<h[-10]; ll = l[-1]<l[-10]
    if hh and hl: return "UPTREND 📈 (HH+HL)"
    if lh and ll: return "DOWNTREND 📉 (LH+LL)"
    return "RANGING ↔️"

def candle_pattern(c, prev):
    body  = abs(c.close-c.open)
    rng   = c.high-c.low
    if rng<1e-10: return "Doji"
    br = body/rng
    bull = c.close>c.open
    uw = c.high - max(c.open,c.close)
    lw = min(c.open,c.close) - c.low
    # engulfing
    if bull and c.open<prev.close and c.close>prev.open and prev.close<prev.open:
        return "🔥 Bullish Engulfing"
    if not bull and c.open>prev.close and c.close<prev.open and prev.close>prev.open:
        return "💀 Bearish Engulfing"
    if br<0.08: return "⚪ Doji (تردد)"
    if br>0.75: return ("🟢 Marubozu صعودي" if bull else "🔴 Marubozu هبوطي")
    if lw>body*2 and br<0.4: return ("🔨 Hammer ارتداد" if bull else "🔨 Hammer")
    if uw>body*2 and br<0.4: return "🌠 Shooting Star هبوط"
    if br>0.4: return ("🟢 Candle صعودية" if bull else "🔴 Candle هبوطية")
    return "↔️ Spinning Top"

# ─────────────────────────────────────────────
# 3.  استراتيجية الثعبان
# ─────────────────────────────────────────────
def snake_strategy(df):
    e9  = ema(df.close,9)
    e21 = ema(df.close,21)
    e50 = ema(df.close,50)
    rs  = rsi(df.close,14)
    sigs=[]
    for i in range(22,len(df)):
        pa = e9.iloc[i-1]>e21.iloc[i-1]
        ca = e9.iloc[i]  >e21.iloc[i]
        if not pa and ca:
            st = "strong" if rs.iloc[i]>45 and df.close.iloc[i]>e50.iloc[i] else "weak"
            sigs.append({"i":i,"dir":"BUY","strength":st})
        elif pa and not ca:
            st = "strong" if rs.iloc[i]<55 and df.close.iloc[i]<e50.iloc[i] else "weak"
            sigs.append({"i":i,"dir":"SELL","strength":st})
    return sigs[-1] if sigs else None

# ─────────────────────────────────────────────
# 4.  محرك الإشارة الرئيسي — متعدد الإطارات
# ─────────────────────────────────────────────
def analyse_tf(df):
    """إرجاع ملخص مؤشرات لإطار زمني واحد"""
    if df is None or len(df)<50: return None
    df = df.tail(500).reset_index(drop=True)
    c = df.close
    price = float(c.iloc[-1])

    e9  = ema(c,9);  e21 = ema(c,21)
    e50 = ema(c,50); e200= ema(c,min(200,len(df)-1))
    rs  = rsi(c,14)
    ml,ms,mh = macd(c)
    at  = atr(df,14)
    vw  = vwap(df)
    bbu,bbm,bbl = bollinger(c)
    _,stdir = supertrend(df)
    sk,sd = stochastic(df)

    cur_rsi  = float(rs.iloc[-1])
    cur_mh   = float(mh.iloc[-1])
    cur_ml   = float(ml.iloc[-1])
    cur_ms   = float(ms.iloc[-1])
    cur_atr  = float(at.iloc[-1])
    cur_vwap = float(vw.iloc[-1])
    e200v    = float(e200.iloc[-1])
    e50v     = float(e50.iloc[-1])
    e21v     = float(e21.iloc[-1])
    e9v      = float(e9.iloc[-1])
    bbuv     = float(bbu.iloc[-1])
    bblv     = float(bbl.iloc[-1])
    bbmv     = float(bbm.iloc[-1])
    stdir_v  = int(stdir.iloc[-1]) if not np.isnan(stdir.iloc[-1]) else 0
    sk_v     = float(sk.iloc[-1])
    sd_v     = float(sd.iloc[-1])

    bull=0; bear=0; reasons=[]

    # EMA200 trend filter  — 25 pts
    if price>e200v:
        bull+=25; reasons.append({"ind":"EMA200","sig":"BULL","val":f"السعر فوق EMA200 ({e200v:.4f})","icon":"📈"})
    else:
        bear+=25; reasons.append({"ind":"EMA200","sig":"BEAR","val":f"السعر تحت EMA200 ({e200v:.4f})","icon":"📉"})

    # EMA stack 9>21>50  — 20 pts
    if e9v>e21v>e50v:
        bull+=20; reasons.append({"ind":"EMA Stack","sig":"STRONG BULL","val":"EMA9>EMA21>EMA50 تراص صعودي","icon":"🚀"})
    elif e9v<e21v<e50v:
        bear+=20; reasons.append({"ind":"EMA Stack","sig":"STRONG BEAR","val":"EMA9<EMA21<EMA50 تراص هبوطي","icon":"💥"})
    else:
        reasons.append({"ind":"EMA Stack","sig":"NEUTRAL","val":"المتوسطات متباينة","icon":"➖"})

    # RSI  — 20 pts
    if cur_rsi<28:
        bull+=20; reasons.append({"ind":"RSI","sig":"STRONG BULL","val":f"تشبع بيع حاد {cur_rsi:.1f}","icon":"🔥"})
    elif cur_rsi<42:
        bull+=12; reasons.append({"ind":"RSI","sig":"BULL","val":f"منطقة صعود {cur_rsi:.1f}","icon":"📈"})
    elif cur_rsi>72:
        bear+=20; reasons.append({"ind":"RSI","sig":"STRONG BEAR","val":f"تشبع شراء حاد {cur_rsi:.1f}","icon":"⚠️"})
    elif cur_rsi>58:
        bear+=12; reasons.append({"ind":"RSI","sig":"BEAR","val":f"منطقة هبوط {cur_rsi:.1f}","icon":"📉"})
    else:
        reasons.append({"ind":"RSI","sig":"NEUTRAL","val":f"محايد {cur_rsi:.1f}","icon":"➖"})

    # MACD  — 15 pts
    macd_prev_h = float(mh.iloc[-2])
    if cur_ml>cur_ms and cur_mh>0 and cur_mh>macd_prev_h:
        bull+=15; reasons.append({"ind":"MACD","sig":"BULL","val":f"تقاطع صعودي متسارع {cur_mh:.5f}","icon":"📈"})
    elif cur_ml<cur_ms and cur_mh<0 and cur_mh<macd_prev_h:
        bear+=15; reasons.append({"ind":"MACD","sig":"BEAR","val":f"تقاطع هبوطي متسارع {cur_mh:.5f}","icon":"📉"})
    elif cur_ml>cur_ms:
        bull+=7; reasons.append({"ind":"MACD","sig":"BULL","val":f"فوق الإشارة {cur_mh:.5f}","icon":"📈"})
    elif cur_ml<cur_ms:
        bear+=7; reasons.append({"ind":"MACD","sig":"BEAR","val":f"تحت الإشارة {cur_mh:.5f}","icon":"📉"})
    else:
        reasons.append({"ind":"MACD","sig":"NEUTRAL","val":f"في التقاطع {cur_mh:.5f}","icon":"➖"})

    # VWAP  — 10 pts
    if price>cur_vwap:
        bull+=10; reasons.append({"ind":"VWAP","sig":"BULL","val":f"فوق VWAP {cur_vwap:.4f}","icon":"📈"})
    else:
        bear+=10; reasons.append({"ind":"VWAP","sig":"BEAR","val":f"تحت VWAP {cur_vwap:.4f}","icon":"📉"})

    # Supertrend  — 15 pts
    if stdir_v==1:
        bull+=15; reasons.append({"ind":"Supertrend","sig":"BULL","val":"اتجاه صعودي مؤكد ✅","icon":"✅"})
    elif stdir_v==-1:
        bear+=15; reasons.append({"ind":"Supertrend","sig":"BEAR","val":"اتجاه هبوطي مؤكد ❌","icon":"❌"})

    # Bollinger  — 10 pts
    bb_rng = bbuv-bblv
    bb_pct = (price-bblv)/bb_rng if bb_rng>0 else 0.5
    if price<bblv:
        bull+=10; reasons.append({"ind":"Bollinger","sig":"BULL","val":"تحت الشريط السفلي — ارتداد محتمل","icon":"🔄"})
    elif price>bbuv:
        bear+=10; reasons.append({"ind":"Bollinger","sig":"BEAR","val":"فوق الشريط العلوي — تصحيح محتمل","icon":"🔄"})
    else:
        reasons.append({"ind":"Bollinger","sig":"NEUTRAL","val":f"موضع {bb_pct*100:.0f}%","icon":"➖"})

    # Stochastic  — 10 pts
    if sk_v<20 and sk_v>sd_v:
        bull+=10; reasons.append({"ind":"Stoch","sig":"BULL","val":f"K={sk_v:.1f} تشبع بيع + تقاطع صعودي","icon":"📈"})
    elif sk_v>80 and sk_v<sd_v:
        bear+=10; reasons.append({"ind":"Stoch","sig":"BEAR","val":f"K={sk_v:.1f} تشبع شراء + تقاطع هبوطي","icon":"📉"})
    else:
        reasons.append({"ind":"Stoch","sig":"NEUTRAL","val":f"K={sk_v:.1f} D={sd_v:.1f}","icon":"➖"})

    return {"bull":bull,"bear":bear,"reasons":reasons,
            "price":price,"atr":cur_atr,"rsi":cur_rsi,
            "e9":e9v,"e21":e21v,"e50":e50v,"e200":e200v,
            "macd":cur_ml,"macd_sig":cur_ms,"macd_hist":cur_mh,
            "vwap":cur_vwap,"bb_u":bbuv,"bb_m":bbmv,"bb_l":bblv,
            "st_dir":stdir_v,"stoch_k":sk_v,"stoch_d":sd_v,
            "df":df}

def generate_signal(pair, interval="1h"):
    # ── جلب الإطار الرئيسي ──
    df_main = fetch(pair, interval)
    if df_main is None or len(df_main)<60:
        return {"error":f"لا توجد بيانات كافية لـ {pair}. حاول مرة أخرى."}

    main = analyse_tf(df_main)
    if main is None:
        return {"error":"فشل تحليل البيانات"}

    # ── جلب الإطار الأعلى (تأكيد الاتجاه) ──
    htf = HIGHER_TF.get(interval, "1d")
    df_high = fetch(pair, htf)
    higher = analyse_tf(df_high) if df_high is not None else None

    price   = main["price"]
    cur_atr = main["atr"]
    reasons = main["reasons"]
    df      = main["df"]

    bull = main["bull"]
    bear = main["bear"]

    # ── تعزيز بالإطار الأعلى (25 pts إضافية) ──
    htf_confirm = "غير متاح"
    if higher:
        htf_total = higher["bull"]+higher["bear"]
        htf_bull_pct = higher["bull"]/htf_total*100 if htf_total>0 else 50
        if htf_bull_pct>=60:
            bull+=25; htf_confirm="صعود ✅"
            reasons.append({"ind":f"HTF {htf.upper()}","sig":"BULL","val":f"الإطار الأعلى صعودي {htf_bull_pct:.0f}%","icon":"🔝"})
        elif htf_bull_pct<=40:
            bear+=25; htf_confirm="هبوط ✅"
            reasons.append({"ind":f"HTF {htf.upper()}","sig":"BEAR","val":f"الإطار الأعلى هبوطي {100-htf_bull_pct:.0f}%","icon":"🔝"})
        else:
            htf_confirm="محايد ➖"
            reasons.append({"ind":f"HTF {htf.upper()}","sig":"NEUTRAL","val":"الإطار الأعلى محايد","icon":"➖"})

    # ── إشارة الثعبان (10 pts) ──
    snake = snake_strategy(df)
    snake_info = None
    if snake:
        if snake["dir"]=="BUY":
            if snake["strength"]=="strong": bull+=10
            else: bull+=5
            snake_info={"dir":"BUY","strength":snake["strength"],"label":"🐍 ثعبان شراء"}
        else:
            if snake["strength"]=="strong": bear+=10
            else: bear+=5
            snake_info={"dir":"SELL","strength":snake["strength"],"label":"🐍 ثعبان بيع"}

    # ── مؤشر الزخم المتفق عليه (حارس الجودة) ──
    #   لا إشارة إلا إذا اتفق RSI + MACD + Supertrend
    rsi_bull  = main["rsi"]<50
    macd_bull = main["macd_hist"]>0
    st_bull   = main["st_dir"]==1
    rsi_bear  = main["rsi"]>50
    macd_bear = main["macd_hist"]<0
    st_bear   = main["st_dir"]==-1

    agreement_bull = sum([rsi_bull, macd_bull, st_bull])   # 0-3
    agreement_bear = sum([rsi_bear, macd_bear, st_bear])   # 0-3

    # ── قرار الإشارة النهائي ──
    total = bull+bear if bull+bear>0 else 1
    bull_pct = bull/total*100
    bear_pct = bear/total*100

    # شرط الإشارة: فارق ≥ 20 نقطة + اتفاق ≥ 2 مؤشرات
    if bull-bear>=20 and agreement_bull>=2:
        signal="BUY"
        confidence=min(95, 50 + (bull-bear)//2 + agreement_bull*3)
        quality = "⭐⭐⭐ ممتازة" if bull-bear>=50 and agreement_bull==3 else \
                  "⭐⭐ جيدة"  if bull-bear>=30 else "⭐ مقبولة"
    elif bear-bull>=20 and agreement_bear>=2:
        signal="SELL"
        confidence=min(95, 50 + (bear-bull)//2 + agreement_bear*3)
        quality = "⭐⭐⭐ ممتازة" if bear-bull>=50 and agreement_bear==3 else \
                  "⭐⭐ جيدة"  if bear-bull>=30 else "⭐ مقبولة"
    else:
        signal="WAIT"
        confidence=50
        quality="⏳ انتظر تأكيداً"

    # ── نقاط الدخول والخروج (بسعر السوق الحقيقي) ──
    spread = cur_atr*0.05      # تقدير السبريد
    if signal=="BUY":
        entry = round(price + spread, 5)          # سعر الشراء الفعلي
        sl    = round(entry - 2.0*cur_atr, 5)
        tp1   = round(entry + 1.5*cur_atr, 5)
        tp2   = round(entry + 3.0*cur_atr, 5)
        tp3   = round(entry + 5.0*cur_atr, 5)
    elif signal=="SELL":
        entry = round(price - spread, 5)          # سعر البيع الفعلي
        sl    = round(entry + 2.0*cur_atr, 5)
        tp1   = round(entry - 1.5*cur_atr, 5)
        tp2   = round(entry - 3.0*cur_atr, 5)
        tp3   = round(entry - 5.0*cur_atr, 5)
    else:
        entry=price; sl=price-cur_atr; tp1=price+cur_atr
        tp2=price+2*cur_atr; tp3=price+3*cur_atr
    sl_pips = round(abs(entry-sl)/cur_atr*100,1)
    rr = round(abs(tp1-entry)/abs(entry-sl),2) if abs(entry-sl)>0 else 1.0

    # ── تحليل الشمعة ──
    cur_c  = df.iloc[-1]; prev_c = df.iloc[-2]
    c_body = abs(float(cur_c.close)-float(cur_c.open))
    c_rng  = float(cur_c.high)-float(cur_c.low)
    c_br   = c_body/c_rng if c_rng>0 else 0
    is_bull_c = float(cur_c.close)>float(cur_c.open)
    c_uw = float(cur_c.high)-max(float(cur_c.open),float(cur_c.close))
    c_lw = min(float(cur_c.open),float(cur_c.close))-float(cur_c.low)
    c_pat = candle_pattern(cur_c, prev_c)
    c_str = "قوية 💪" if c_br>0.65 else ("متوسطة" if c_br>0.38 else "ضعيفة")

    # توقع الشمعة القادمة
    if signal=="BUY":
        next_c="🟢 صاعدة (استمرار صعود)"; next_p=confidence
    elif signal=="SELL":
        next_c="🔴 هابطة (استمرار هبوط)"; next_p=confidence
    else:
        next_c="⚪ غير محدد — انتظر إغلاق الشمعة"; next_p=50

    # ── pivot + OB ──
    try: pvt=pivot_points(df)
    except: pvt={}
    obs = order_blocks(df)
    ms  = market_structure(df)

    return {
        "pair":      pair,
        "interval":  interval,
        "htf":       htf,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "price":     round(price,5),
        "signal":    signal,          # BUY / SELL / WAIT
        "confidence":confidence,
        "quality":   quality,
        "bull_score":bull,"bear_score":bear,
        "bull_pct":  round(bull_pct,1),"bear_pct":round(bear_pct,1),
        "htf_confirm":htf_confirm,
        "agreement_bull":agreement_bull,"agreement_bear":agreement_bear,
        "market_structure":ms,
        "trade":{
            "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "rr":rr,"sl_pips":sl_pips,
            "direction":"LONG 📈" if signal=="BUY" else ("SHORT 📉" if signal=="SELL" else "لا صفقة"),
            "size_note":"حجم اللوت حسب إدارة رأس المال (1-2% خطر)"
        },
        "candle":{
            "direction":"🟢 صاعدة" if is_bull_c else "🔴 هابطة",
            "pattern":c_pat,"strength":c_str,
            "body_pct":round(c_br*100,1),
            "open":round(float(cur_c.open),5),"high":round(float(cur_c.high),5),
            "low":round(float(cur_c.low),5),"close":round(float(cur_c.close),5),
            "upper_wick":round(c_uw,5),"lower_wick":round(c_lw,5),
        },
        "next_candle":next_c,"next_candle_prob":next_p,
        "indicators":{
            "ema9":round(main["e9"],5),"ema21":round(main["e21"],5),
            "ema50":round(main["e50"],5),"ema200":round(main["e200"],5),
            "rsi":round(main["rsi"],2),
            "stoch_k":round(main["stoch_k"],2),"stoch_d":round(main["stoch_d"],2),
            "macd":round(main["macd"],5),"macd_signal":round(main["macd_sig"],5),
            "macd_hist":round(main["macd_hist"],5),
            "atr":round(cur_atr,5),"vwap":round(main["vwap"],5),
            "bb_upper":round(main["bb_u"],5),"bb_mid":round(main["bb_m"],5),
            "bb_lower":round(main["bb_l"],5),
            "supertrend_dir":main["st_dir"],
        },
        "signals_breakdown":reasons,
        "pivot_points":pvt,"order_blocks":obs,
        "snake_signal":snake_info,
    }

# ─────────────────────────────────────────────
# 5.  Flask Routes
# ─────────────────────────────────────────────
PAIRS     = ["XAUUSD","BTCUSD","EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","NZDUSD","USDCAD"]
INTERVALS = ["5m","15m","30m","1h","4h","1d"]

@app.route("/")
def index():
    return render_template("index.html", pairs=PAIRS, intervals=INTERVALS)

@app.route("/api/signal")
def api_signal():
    pair     = request.args.get("pair","XAUUSD")
    interval = request.args.get("interval","1h")
    return jsonify(generate_signal(pair, interval))

@app.route("/api/multi_signal")
def api_multi():
    iv = request.args.get("interval","1h")
    out=[]
    for p in ["XAUUSD","BTCUSD","EURUSD","GBPUSD","USDJPY"]:
        try:
            r=generate_signal(p,iv)
            if "error" not in r: out.append(r)
        except Exception as e:
            out.append({"pair":p,"error":str(e)})
    return jsonify(out)

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","time":datetime.utcnow().isoformat()})

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
