import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import re
from openai import OpenAI
from datetime import datetime

# ==================== 配置区 ====================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
SYMBOL = "BTC-USDT-SWAP"
REFRESH_INTERVAL = 10

BASE_URL = "https://api.deepseek.com"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
# ==============================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = "点击下方按钮开始分析"
if "indicators" not in st.session_state:
    st.session_state.indicators = None

# ---------- 全局紧凑样式 + 顶部留空 ----------
st.markdown("""
<style>
    /* 顶部留出三行空白 */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0.2rem !important;
    }
    .stButton button { width: 100%; font-weight: bold; }
    h3 { padding-top: 0.3rem; margin-bottom: 0.2rem; }
    hr { margin: 0.2rem 0; }
    /* 统一所有 metric 数值大小 */
    .stMetric {
        font-size: 1.3rem !important;
    }
    .stMetric label {
        font-size: 0.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 数据获取函数 ----------
def get_ticker():
    try:
        resp = requests.get(f"{OKX_TICKER_URL}?instId={SYMBOL}", timeout=10)
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            return data["data"][0]
    except:
        pass
    return None

def get_funding_rate():
    try:
        resp = requests.get(f"{OKX_FUNDING_URL}?instId={SYMBOL}", timeout=10)
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            return data["data"][0]
    except:
        pass
    return None

def get_candles(bar="15m", limit=100):
    try:
        resp = requests.get(f"{OKX_CANDLES_URL}?instId={SYMBOL}&bar={bar}&limit={limit}", timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return data["data"]
    except:
        pass
    return None

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_kdj(high, low, close, n=9, m1=3, m2=3):
    low_min = low.rolling(window=n, min_periods=1).min()
    high_max = high.rolling(window=n, min_periods=1).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    k = np.zeros(len(rsv)); d = np.zeros(len(rsv))
    for i in range(len(rsv)):
        if i == 0:
            k[i] = 50; d[i] = 50
        else:
            k[i] = (m1-1)/m1 * k[i-1] + 1/m1 * rsv.iloc[i]
            d[i] = (m2-1)/m2 * d[i-1] + 1/m2 * k[i]
    j = 3 * k - 2 * d
    return k, d, j

def compute_indicators(candles, label=""):
    if not candles or len(candles) < 30:
        return None
    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(candles, columns=cols)
    df = df.iloc[::-1].reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol", "volCcy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["close"])
    df["K"], df["D"], df["J"] = compute_kdj(df["high"], df["low"], df["close"])

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOL_MA5"] = df["vol"].rolling(5).mean()
    df["VOL_MA20"] = df["vol"].rolling(20).mean()

    recent_20 = df.tail(20); recent_50 = df.tail(50)
    resistance_20, support_20 = recent_20["high"].max(), recent_20["low"].min()
    resistance_50, support_50 = recent_50["high"].max(), recent_50["low"].min()

    latest = df.iloc[-1]
    return {
        "label": label,
        "latest_close": latest["close"],
        "MA5": latest["MA5"], "MA10": latest["MA10"], "MA20": latest["MA20"], "MA50": latest["MA50"],
        "MACD": latest["MACD"], "MACD_signal": latest["MACD_signal"], "MACD_hist": latest["MACD_hist"],
        "K": latest["K"], "D": latest["D"], "J": latest["J"], "RSI": latest["RSI"],
        "vol_latest": latest["vol"], "VOL_MA5": latest["VOL_MA5"], "VOL_MA20": latest["VOL_MA20"],
        "support_20": support_20, "resistance_20": resistance_20,
        "support_50": support_50, "resistance_50": resistance_50,
        "trend": "上升" if latest["MA5"] > latest["MA20"] else "下降",
    }

def run_analysis():
    ticker = get_ticker()
    funding = get_funding_rate()
    if not ticker:
        st.error("获取行情数据失败")
        return

    periods = {"5m": ("5m", 100), "15m": ("15m", 100), "1h": ("1H", 100), "1d": ("1D", 60)}
    indicators = {}
    for name, (bar, limit) in periods.items():
        data = get_candles(bar, limit)
        if data:
            ind = compute_indicators(data, f"{name}线")
            if ind:
                indicators[name] = ind

    st.session_state.indicators = indicators

    def fmt(ind, title):
        if ind is None: return f"{title}：数据不足"
        return f"""{title}：价格 {ind['latest_close']:.2f} 趋势 {ind['trend']}
MA5/10/20/50：{ind['MA5']:.2f}/{ind['MA10']:.2f}/{ind['MA20']:.2f}/{ind['MA50']:.2f}
MACD(DIF/DEA/柱)：{ind['MACD']:.2f}/{ind['MACD_signal']:.2f}/{ind['MACD_hist']:.2f}
KDJ(K/D/J)：{ind['K']:.2f}/{ind['D']:.2f}/{ind['J']:.2f} RSI：{ind['RSI']:.2f}
撑压(20/50)：{ind['support_20']:.2f}/{ind['resistance_20']:.2f} | {ind['support_50']:.2f}/{ind['resistance_50']:.2f}"""

    prompt = f"""你是BTC永续合约短线分析师。当前时间：{datetime.now().strftime("%H:%M:%S")}
价格：{ticker['last']} USDT
资金费率：{funding.get('fundingRate', 'N/A') if funding else 'N/A'}
多周期技术指标：
{fmt(indicators.get('5m'), '5分钟线')}
{fmt(indicators.get('15m'), '15分钟线')}
{fmt(indicators.get('1h'), '1小时线')}
{fmt(indicators.get('1d'), '日线')}
请结合多周期指标和资金费率，进行简短分析（不超过250字），必须包含：
1. 当前多空力量对比与市场情绪
2. 关键支撑位与压力位
3. 短线操作思路（做多/做空/观望）及风险提示
请将第三点操作思路部分用【】包裹。"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=350
        )
        analysis = resp.choices[0].message.content
    except Exception as e:
        analysis = f"AI分析出错: {e}"

    st.session_state.latest_analysis = analysis

def main():
    st.set_page_config(page_title="BTC盯盘", layout="wide")
    
    # ---------- 小标题 + 温馨提示 ----------
    st.markdown("<h4 style='margin-top:0; margin-bottom:0.1rem;'>📈 BTC永续合约</h4>", unsafe_allow_html=True)
    st.caption("温馨提示：本页面仅用于AI交流学习，不构成任何投资建议。")

    ticker = get_ticker()
    funding = get_funding_rate()
    if not ticker:
        st.warning("行情获取失败，请稍候...")
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

    current_price = float(ticker["last"])
    change_24h = (current_price - float(ticker["open24h"])) / float(ticker["open24h"]) * 100

    sod_utc0 = float(ticker.get("sodUtc0", 0))
    change_today = (current_price - sod_utc0) / sod_utc0 * 100 if sod_utc0 != 0 else 0.0
    vol_btc = float(ticker.get("volCcy24h", 0))

    # ---------- 第一行：当前价格、24h最高、24h最低（统一字体大小） ----------
    st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:0.8rem;">
    <div style="line-height:1.2">
        <span style="font-size:0.7rem; color:gray;">当前价格</span><br>
        <span style="font-size:1.3rem; font-weight:bold;">{current_price:.2f}</span>
    </div>
    <div style="line-height:1.2; text-align:center">
        <span style="font-size:0.7rem; color:gray;">24h最高</span><br>
        <span style="font-size:1.3rem; font-weight:bold;">{ticker['high24h']}</span>
    </div>
    <div style="line-height:1.2; text-align:right">
        <span style="font-size:0.7rem; color:gray;">24h最低</span><br>
        <span style="font-size:1.3rem; font-weight:bold;">{ticker['low24h']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # ---------- 第二行：24h涨跌、当日涨跌 ----------
    col4, col5 = st.columns(2)
    col4.metric("24h涨跌", f"{change_24h:+.2f}%")
    col5.metric("当日涨跌", f"{change_today:+.2f}%")

    # ---------- 第三行：成交量 + 资金费率 ----------
    col6, col7 = st.columns([1, 2])
    col6.metric("24h量(BTC)", f"{vol_btc:.2f}")
    if funding:
        col7.caption(f"💰 资金费率: {funding.get('fundingRate','N/A')} | 下次结算: {funding.get('nextFundingTime','N/A')}")

    # ---------- 技术指标（默认展示） ----------
    inds = st.session_state.indicators
    if inds:
        st.markdown("##### 📊 技术指标")
        col_a, col_b = st.columns(2)
        if "15m" in inds:
            col_a.markdown("<small><b>15分钟线</b>：MA5={MA5:.2f} MA20={MA20:.2f} MACD={MACD:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
            col_a.markdown("<small>KDJ: K={K:.2f} D={D:.2f} J={J:.2f} RSI={RSI:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
            col_a.markdown("<small>撑压(20/50)：{support_20:.2f}/{resistance_20:.2f} | {support_50:.2f}/{resistance_50:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
        if "1h" in inds:
            col_b.markdown("<small><b>1小时线</b>：MA5={MA5:.2f} MA20={MA20:.2f} 趋势={trend}</small>".format(**inds['1h']), unsafe_allow_html=True)
            col_b.markdown("<small>撑压(50)：{support_50:.2f}/{resistance_50:.2f}</small>".format(**inds['1h']), unsafe_allow_html=True)
        if "5m" in inds:
            col_b.markdown("<small><b>5分钟线</b>：趋势={trend} 成交量MA5={VOL_MA5:.2f} RSI={RSI:.2f}</small>".format(**inds['5m']), unsafe_allow_html=True)
        if "1d" in inds:
            col_a.markdown("<small><b>日线</b>：MA5={MA5:.2f} MA20={MA20:.2f} 趋势={trend}</small>".format(**inds['1d']), unsafe_allow_html=True)
    else:
        st.caption("技术指标将在首次分析后显示")

    # ---------- AI 分析按钮与结果 ----------
    st.subheader("🤖 AI 短线分析")
    if st.button("🔍 手动触发 AI 分析", use_container_width=True):
        with st.spinner("分析中..."):
            run_analysis()
        st.rerun()

    analysis = st.session_state.latest_analysis
    if "【" in analysis and "】" in analysis:
        parts = re.split(r'(【.*?】)', analysis)
        html_parts = []
        for part in parts:
            if part.startswith('【') and part.endswith('】'):
                html_parts.append(f"<span style='color:red;font-weight:bold'>{part[1:-1]}</span>")
            else:
                html_parts.append(part.replace('\n', '<br>'))
        st.markdown("".join(html_parts), unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:red'>{analysis.replace(chr(10), '<br>')}</span>", unsafe_allow_html=True)

    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
