import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="BTC 实时趋势监控",
    page_icon="₿",
    layout="wide" # 改为宽屏布局，方便展示图表
)

# 自定义 CSS 提升美观度
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)  # 缓存 60 秒，防止频繁请求 API
def get_bitcoin_detailed_data():
    """
    获取比特币实时数据及历史价格（过去7天）
    """
    try:
        # 1. 获取实时价格和24h变化
        price_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        # 2. 获取历史数据（用于绘图）
        history_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7&interval=daily"

        p_res = requests.get(price_url, timeout=10)
        h_res = requests.get(history_url, timeout=10)

        p_res.raise_for_status()
        h_res.raise_for_status()

        p_data = p_res.json()
        h_data = h_res.json()

        # 解析实时数据
        price = p_data['bitcoin']['usd']
        change_24h_pct = p_data['bitcoin']['usd_24h_change']
        old_price = price / (1 + (change_24h_pct / 100))
        change_24h_amount = price - old_price

        # 解析历史数据用于折线图
        prices = h_data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)

        return {
            "price": price,
            "change_pct": change_24h_pct,
            "change_amount": change_24h_amount,
            "history_df": df['price'],
            "time": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        st.error(f"数据抓取失败: {e}")
        return None

# --- 侧边栏 ---
with st.sidebar:
    st.title("⚙️ 配置中心")
    st.info("数据每 60 秒自动更新，也可点击下方手动刷新。")
    if st.button("🔄 立即强制刷新", type="primary"):
        st.cache_data.clear()
        st.rerun()

# --- 主界面 ---
st.title("₿ Bitcoin 实时价格与趋势看板")

data = get_bitcoin_detailed_data()

if data:
    # 第一行：核心指标
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("当前价格 (USD)", f"${data['price']:,.2f}")
    with m2:
        st.metric("24h 涨跌额", f"${abs(data['change_amount']):,.2f}",
                  delta=f"{data['change_amount']:,.2f}")
    with m3:
        st.metric("24h 涨跌幅", f"{data['change_pct']:.2f}%",
                  delta=f"{data['change_pct']:.2f}%")

    # 第二行：历史趋势图
    st.subheader("📈 过去 7 天价格走势")
    st.area_chart(data['history_df'], use_container_width=True)

    # 页脚状态
    st.caption(f"🏁 数据最后同步时间: {data['time']} (UTC) | 数据源: CoinGecko")
else:
    st.warning("⚠️ 暂时无法获取实时行情，请稍后刷新。")

st.divider()