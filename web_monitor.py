import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import re
from openai import OpenAI
from datetime import datetime, timedelta

# ==================== 配置区 ====================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
SYMBOL = "BTC-USDT-SWAP"
REFRESH_INTERVAL = 1                 # 数据刷新间隔：1秒

BASE_URL = "https://api.deepseek.com"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
# ==============================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = "点击“deepseek分析”按钮查看结果"
if "latest_suggestion" not in st.session_state:
    st.session_state.latest_suggestion = ""
if "indicators" not in st.session_state:
    st.session_state.indicators = None

# ---------- 紧凑样式 + 顶部留空 ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0.2rem !important;
    }
    .stButton button {
        width: auto !important;
        min-width: 100px;
        padding: 0.2rem 0.8rem;
        font-weight: bold;
    }
    .small-title {
        font-size: 0.9rem;
        font-weight: bold;
        margin-top: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .price-large {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .price-label {
        font-size: 0.7rem;
        color: white;
        margin-bottom: 0.1rem;
    }
    .data-row {
        display: flex;
        justify-content: space-around;
        margin-bottom: 0.8rem;
    }
    .data-item {
        line-height: 1.2;
        text-align: center;
        flex: 1;
    }
    .data-label {
        font-size: 0.7rem;
        color: gray;
    }
    .data-value {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .time-text {
        font-size: 0.7rem;
        color: gray;
        margin-top: 0.1rem;
    }
    .btn-right {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.5rem;
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
    """手动触发时的完整AI分析"""
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
    suggestion = ""
    match = re.search(r'【(.*?)】', analysis)
    if match:
        suggestion = match.group(1)
    st.session_state.latest_suggestion = suggestion

def main():
    st.set_page_config(page_title="BTC盯盘", layout="wide")
    
    # 标题
    st.markdown("<h4 style='margin:0; white-space:nowrap;'>📈 BTC永续合约</h4>", unsafe_allow_html=True)
    st.caption("温馨提示：本页面仅用于AI交流学习，不构成任何投资建议。")

    # 手动分析按钮
    st.markdown('<div class="btn-right">', unsafe_allow_html=True)
    if st.button("deepseek分析", key="ai_btn"):
        with st.spinner("分析中..."):
            run_analysis()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # 置顶AI建议
    suggestion = st.session_state.latest_suggestion
    if suggestion:
        st.markdown(f"<div style='color:#DB4437; font-weight:bold; margin-bottom:0.3rem;'>{suggestion}</div>", unsafe_allow_html=True)

    # 获取数据
    ticker = get_ticker()
    funding = get_funding_rate()
    if not ticker:
        st.warning("行情获取失败，请稍候...")
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

    current_price = float(ticker["last"])

    # 涨跌计算（保持逻辑不变，但只用于显示）
    change_daily = 0.0
    daily_kline = get_candles("1D", 1)
    if daily_kline and len(daily_kline) > 0:
        try:
            open_beijing = float(daily_kline[0][1])
            if open_beijing > 0:
                change_daily = (current_price - open_beijing) / open_beijing * 100
        except:
            change_daily = (current_price - float(ticker["open24h"])) / float(ticker["open24h"]) * 100
    else:
        change_daily = (current_price - float(ticker["open24h"])) / float(ticker["open24h"]) * 100

    change_yesterday = 0.0
    daily_candles = get_candles("1D", 2)
    if daily_candles and len(daily_candles) >= 2:
        cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
        df_daily = pd.DataFrame(daily_candles, columns=cols)
        for col in ["open", "high", "low", "close", "vol", "volCcy"]:
            df_daily[col] = pd.to_numeric(df_daily[col], errors="coerce")
        df_daily = df_daily.sort_values("ts", ascending=True)
        if len(df_daily) >= 2:
            yesterday = df_daily.iloc[-2]
            if yesterday["open"] > 0:
                change_yesterday = (yesterday["close"] - yesterday["open"]) / yesterday["open"] * 100

    vol_btc = float(ticker.get("volCcy24h", 0))
    volume_usdt = vol_btc * current_price

    # 时间显示
    ticker_ts = ticker.get("ts", "")
    if ticker_ts:
        try:
            utc_time = datetime.utcfromtimestamp(int(ticker_ts) / 1000)
            beijing_time = utc_time + timedelta(hours=8)
            update_time = beijing_time.strftime("%Y-%m-%d %H:%M:%S")
        except:
            update_time = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        update_time = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

    # 当前价格
    st.markdown(f"""
    <div style="margin-bottom:0.5rem; text-align: center;">
        <div class="price-label">当前价格</div>
        <div class="price-large">{current_price:.2f} USDT</div>
        <div class="time-text">数据获取时间：{update_time}</div>
    </div>
    """, unsafe_allow_html=True)

    # 24h最高/最低
    st.markdown(f"""
    <div class="data-row">
        <div class="data-item">
            <div class="data-label">24h最高</div>
            <div class="data-value">{ticker['high24h']}</div>
        </div>
        <div class="data-item">
            <div class="data-label">24h最低</div>
            <div class="data-value">{ticker['low24h']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 涨跌行
    def color_str(val):
        if val > 0: return f'<span style="color:#DB4437;">+{val:.2f}%</span>'
        elif val < 0: return f'<span style="color:#0F9D58;">{val:.2f}%</span>'
        else: return '<span style="color:gray;">0.00%</span>'

    st.markdown(f"""
    <div class="data-row">
        <div class="data-item">
            <div class="data-label">当日涨跌</div>
            <div class="data-value">{color_str(change_daily)}</div>
        </div>
        <div class="data-item">
            <div class="data-label">昨日涨跌</div>
            <div class="data-value">{color_str(change_yesterday)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 量额行
    vol_btc_wan = vol_btc / 10000
    volume_usdt_yi = volume_usdt / 100000000
    st.markdown(f"""
    <div class="data-row">
        <div class="data-item">
            <div class="data-label">24h量(万)</div>
            <div class="data-value">{vol_btc_wan:.2f}</div>
        </div>
        <div class="data-item">
            <div class="data-label">24h额(亿)</div>
            <div class="data-value">{volume_usdt_yi:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 资金费率
    if funding:
        funding_rate = funding.get('fundingRate', 'N/A')
        next_time_str = ""
        next_funding_ts = funding.get('nextFundingTime', '')
        if next_funding_ts:
            try:
                utc_time = datetime.utcfromtimestamp(int(next_funding_ts) / 1000)
                beijing_next = utc_time + timedelta(hours=8)
                next_time_str = beijing_next.strftime("%Y-%m-%d %H:%M:%S")
            except:
                next_time_str = "转换失败"
        if next_time_str:
            st.caption(f"💰 资金费率: {funding_rate} | 下次结算: {next_time_str}")
        else:
            st.caption(f"💰 资金费率: {funding_rate}")

    # 技术指标
    inds = st.session_state.indicators
    if inds:
        st.markdown("<div class='small-title'>📊 技术指标</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        if "15m" in inds:
            col_a.markdown("<small>15分线：MA5={MA5:.2f} MA20={MA20:.2f} MACD={MACD:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
            col_a.markdown("<small>KDJ: K={K:.2f} D={D:.2f} J={J:.2f} RSI={RSI:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
            col_a.markdown("<small>撑压(20/50)：{support_20:.2f}/{resistance_20:.2f} | {support_50:.2f}/{resistance_50:.2f}</small>".format(**inds['15m']), unsafe_allow_html=True)
        if "1h" in inds:
            col_b.markdown("<small>1小时线：MA5={MA5:.2f} MA20={MA20:.2f} 趋势={trend}</small>".format(**inds['1h']), unsafe_allow_html=True)
            col_b.markdown("<small>撑压(50)：{support_50:.2f}/{resistance_50:.2f}</small>".format(**inds['1h']), unsafe_allow_html=True)
        if "5m" in inds:
            col_b.markdown("<small>5分钟线：趋势={trend} 量MA5={VOL_MA5:.2f} RSI={RSI:.2f}</small>".format(**inds['5m']), unsafe_allow_html=True)
        if "1d" in inds:
            col_a.markdown("<small>日线：MA5={MA5:.2f} MA20={MA20:.2f} 趋势={trend}</small>".format(**inds['1d']), unsafe_allow_html=True)
    else:
        st.caption("技术指标将在首次分析后显示")

    # AI 分析详情
    st.markdown("<div class='small-title'>🤖 AI 分析详情</div>", unsafe_allow_html=True)
    analysis = st.session_state.latest_analysis
    st.markdown(f"<small>{analysis.replace(chr(10), '<br>')}</small>", unsafe_allow_html=True)

    time.sleep(REFRESH_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
