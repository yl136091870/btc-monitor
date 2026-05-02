import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import re
from openai import OpenAI
from datetime import datetime

# ==================== 配置区 ====================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]  # Streamlit Cloud 读取 Secret
SYMBOL = "BTC-USDT-SWAP"
CHECK_INTERVAL = 10
PRICE_CHANGE_THRESHOLD = 0.01
ANALYSIS_COOLDOWN = 30

BASE_URL = "https://api.deepseek.com"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
# ==============================================

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

# 初始化 session_state
if "prev_price" not in st.session_state:
    st.session_state.prev_price = None
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0
if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = "等待首次分析..."
if "manual_trigger" not in st.session_state:
    st.session_state.manual_trigger = False

def get_ticker():
    try:
        resp = requests.get(f"{OKX_TICKER_URL}?instId={SYMBOL}", timeout=10)
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            return data["data"][0]
    except Exception as e:
        st.error(f"ticker 获取失败: {e}")
    return None

def get_funding_rate():
    try:
        resp = requests.get(f"{OKX_FUNDING_URL}?instId={SYMBOL}", timeout=10)
        data = resp.json()
        if data.get("code") == "0" and data.get("data"):
            return data["data"][0]
    except Exception as e:
        st.error(f"资金费率获取失败: {e}")
    return None

def get_candles(bar="15m", limit=200):
    try:
        resp = requests.get(f"{OKX_CANDLES_URL}?instId={SYMBOL}&bar={bar}&limit={limit}", timeout=10)
        data = resp.json()
        if data.get("code") == "0":
            return data["data"]
    except Exception as e:
        st.error(f"K线获取失败 ({bar}): {e}")
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
        "resistance_20": resistance_20, "support_20": support_20,
        "resistance_50": resistance_50, "support_50": support_50,
        "trend": "上升" if latest["MA5"] > latest["MA20"] else "下降",
        "ma_cross": "金叉" if latest["MA5"] > latest["MA10"] else "死叉",
    }

def run_analysis():
    """执行完整的 AI 分析流程，并更新 session_state"""
    ticker = get_ticker()
    funding = get_funding_rate()
    if not ticker:
        st.error("获取行情数据失败，无法分析")
        return

    # 获取多周期K线并计算指标
    ind_5m = compute_indicators(get_candles("5m", 200), "5分钟线")
    ind_15m = compute_indicators(get_candles("15m", 200), "15分钟线")
    ind_1h = compute_indicators(get_candles("1H", 100), "1小时线")
    ind_1d = compute_indicators(get_candles("1D", 60), "日线")

    # 构造 prompt
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
{fmt(ind_5m, '5分钟线')}
{fmt(ind_15m, '15分钟线')}
{fmt(ind_1h, '1小时线')}
{fmt(ind_1d, '日线')}
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

    # 更新全局分析结果和时间
    st.session_state.latest_analysis = analysis
    st.session_state.last_analysis_time = time.time()
    # 也可存储指标到 session_state 以便显示
    st.session_state.indicators = {
        "5m": ind_5m, "15m": ind_15m, "1h": ind_1h, "1d": ind_1d
    }

def main():
    st.set_page_config(page_title="BTC永续合约盯盘", layout="wide")
    st.title("📈 BTC 永续合约实时监控")

    # ---- 手动分析按钮 ----
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 手动触发 AI 分析", use_container_width=True):
            st.session_state.manual_trigger = True
            st.rerun()

    # 处理手动触发
    if st.session_state.manual_trigger:
        with st.spinner("正在获取数据并分析..."):
            run_analysis()
        st.session_state.manual_trigger = False
        st.rerun()

    # ---- 自动触发逻辑（保留）----
    ticker = get_ticker()
    funding = get_funding_rate()
    if not ticker:
        st.warning("无法获取行情数据")
        time.sleep(CHECK_INTERVAL)
        st.rerun()

    current_price = float(ticker["last"])
    change_24h = (current_price - float(ticker["open24h"])) / float(ticker["open24h"]) * 100

    # 显示价格卡片
    col1, col2, col3 = st.columns(3)
    col1.metric("最新价 (USDT)", f"{current_price:.2f}")
    col1.metric("24h涨跌", f"{change_24h:.2f}%")
    col2.metric("24h最高", ticker["high24h"])
    col2.metric("24h最低", ticker["low24h"])
    col3.metric("卖一价", f"{ticker['askPx']} (量:{ticker['askSz']})")
    col3.metric("买一价", f"{ticker['bidPx']} (量:{ticker['bidSz']})")

    if funding:
        st.caption(f"💰 资金费率: {funding.get('fundingRate', 'N/A')} | 下一次结算: {funding.get('nextFundingTime', 'N/A')}")

    # 自动触发判断
    trigger = False
    change_pct = 0.0
    if st.session_state.prev_price is not None:
        change_pct = (current_price - st.session_state.prev_price) / st.session_state.prev_price * 100
        if abs(change_pct) >= PRICE_CHANGE_THRESHOLD:
            if time.time() - st.session_state.last_analysis_time > ANALYSIS_COOLDOWN:
                trigger = True

    if trigger:
        with st.spinner("波动触发分析中..."):
            run_analysis()

    # ---- 显示指标（如果有存储）----
    if "indicators" in st.session_state:
        inds = st.session_state.indicators
        st.subheader("📊 最新多周期技术指标")
        col1, col2 = st.columns(2)
        if inds["15m"]:
            col1.markdown(f"**15分钟线**：MA5={inds['15m']['MA5']:.2f} MA20={inds['15m']['MA20']:.2f} MACD={inds['15m']['MACD']:.2f}")
            col1.markdown(f"KDJ: K={inds['15m']['K']:.2f} D={inds['15m']['D']:.2f} J={inds['15m']['J']:.2f} RSI={inds['15m']['RSI']:.2f}")
        if inds["1h"]:
            col2.markdown(f"**1小时线**：MA5={inds['1h']['MA5']:.2f} MA20={inds['1h']['MA20']:.2f} 趋势={inds['1h']['trend']}")
        if inds["5m"]:
            col2.markdown(f"**5分钟线**：趋势={inds['5m']['trend']} 成交量MA5={inds['5m']['VOL_MA5']:.2f} RSI={inds['5m']['RSI']:.2f}")
        if inds["1d"]:
            col1.markdown(f"**日线**：MA5={inds['1d']['MA5']:.2f} MA20={inds['1d']['MA20']:.2f} 趋势={inds['1d']['trend']}")

    # ---- 显示 AI 分析（红色高亮操作建议）----
    st.subheader("🤖 AI 短线分析")
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

    # 更新前价并自动刷新
    st.session_state.prev_price = current_price
    time.sleep(CHECK_INTERVAL)
    st.rerun()

if __name__ == "__main__":
    main()
