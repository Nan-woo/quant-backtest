"""
Phase 4.3 — 鲁棒性测试

双向均值回归策略搬到 4 只不同行业股票上验证。
用 Plotly 画多策略对比图。
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# 板块 1：数据获取（一次性，跑完可以注释掉）
# ============================================================
def download_data():
    import baostock as bs
    bs.login()
    stocks = {
        "sh.600519": "茅台",
        "sh.600036": "招行",
        "sz.000725": "京东方",
        "sh.600585": "海螺",
    }
    for code, name in stocks.items():
        rs = bs.query_history_k_data_plus(code,
            "date,open,close,high,low,volume",
            start_date="2020-01-01", end_date="2026-07-31",
            frequency="d", adjustflag="2")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
        df.to_csv(f"data/{code.split('.')[1]}.csv", index=False)
        print(f"  {name} ({code}) → {len(df)} 行")
    bs.logout()
    print("  下载完成\n")

# ============================================================
# 板块 2：回测函数（双向均值回归）
# ============================================================
MA = 20
THRESHOLD = 0.05

def backtest(filename, label, ma=MA, threshold=THRESHOLD):
    """读 CSV → 预处理 → 跑策略 → 返回净值序列"""
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["收盘"] = df["close"]
    df["ma"] = df["收盘"].rolling(ma).mean()
    df["偏离"] = df["收盘"] / df["ma"] - 1

    df["持仓"] = 0
    in_pos = False

    # ---- 逐天模拟 ----
    for i in range(1, len(df)):
        dev = df["偏离"].iloc[i]
        # =============================================================
        # 【你填】买卖信号
        # =============================================================
        if not in_pos and dev < -threshold:
            in_pos = True
        elif in_pos and dev > threshold:
            in_pos = False

        # =============================================================
        df.iloc[i, df.columns.get_loc("持仓")] = int(in_pos)

    # ---- 算收益 ----
    pos = df["持仓"].shift(1).fillna(0)
    df["日收"] = pos * df["收盘"].pct_change()
    df["净值"] = (1 + df["日收"]).cumprod() * 100_000

    # ---- 统计 ----
    net = df["净值"]
    ret = (net.iloc[-1] / 100_000 - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    buy_hold = (df["收盘"].iloc[-1] / df["收盘"].iloc[0] - 1) * 100
    trades = ((df["持仓"].diff() == 1).sum())
    df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * 100_000

    print(f"  {label:<6}  策略{ret:>+7.2f}%  买持{buy_hold:>+7.2f}%  回撤{dd:>+7.2f}%  {int(trades)}次交易")

    return df["净值"], df["买持净值"], label, ret, dd

# ============================================================
# 板块 3：主程序
# ============================================================
if __name__ == "__main__":
    # ---- 先下载数据（已下载过就注释掉下面这行） ----
    # download_data()  # 已下载，注释掉

    print("=" * 72)
    print("  Phase 4.3 — 鲁棒性测试")
    print("=" * 72)
    print()

    # ---- 跑 4 只股票 ----
    # =============================================================
    # 【你填】如果你换了股票，也要改这里
    # =============================================================
    tasks = [
        ("data/600519.csv", "茅台"),
        ("data/600036.csv", "招行"),
        ("data/000725.csv", "京东方"),
        ("data/600585.csv", "海螺"),
    ]
    # =============================================================

    results = []
    for filename, label in tasks:
        net, bh_net, lbl, ret, dd = backtest(filename, label)
        results.append((net, bh_net, lbl, ret, dd))

    # ---- Plotly 对比图 ----
    fig = go.Figure()

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for (net, bh_net, label, ret, dd), color in zip(results, colors):
        # 策略线（实线）
        fig.add_trace(go.Scatter(
            x=net.index, y=net, name=f"{label}策略 {ret:+.1f}%",
            line=dict(color=color, width=2.0)
        ))
        # 买入持有线（同色虚线）
        bh_ret = (bh_net.iloc[-1] / 100_000 - 1) * 100
        fig.add_trace(go.Scatter(
            x=bh_net.index, y=bh_net,
            name=f"{label}买持 {bh_ret:+.1f}%",
            line=dict(color=color, width=1.0, dash="dash"),
            opacity=0.5
        ))

    fig.add_hline(y=100_000, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="双向均值回归 — 4只股票鲁棒性对比",
        xaxis_title="日期", yaxis_title="净值",
        hovermode="x unified",
        font=dict(size=13),
    )
    fig.show()
    print()

    # ---- 你的判断 ----
    print("=" * 72)
    print("  现在看浏览器里的图，回答三个问题：")
    print("  1. 哪只股票效果最好？为什么？")
    print("  2. 哪只效果最差？差在哪？")
    print("  3. 这个策略适合什么类型的股票？")
    print("=" * 72)

    # ============================================================
    # 板块 4：p 值检验 — 每只票的策略收益是不是运气
    # ============================================================
    print("\n" + "=" * 72)
    print("  p 值检验：1000 次随机策略 vs 实际策略")
    print("=" * 72)
    print()

    # 对每只票分别做
    for filename, label in tasks:
        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        df["收盘"] = pd.to_numeric(df["close"])
        df["ma"] = df["收盘"].rolling(MA).mean()
        df["偏离"] = df["收盘"] / df["ma"] - 1

        # ---- 实际策略 ----
        df["持仓"] = 0
        in_pos = False
        for i in range(1, len(df)):
            dev = df["偏离"].iloc[i]
            if not in_pos and dev < -THRESHOLD:
                in_pos = True
            elif in_pos and dev > THRESHOLD:
                in_pos = False
            df.iloc[i, df.columns.get_loc("持仓")] = int(in_pos)
        pos = df["持仓"].shift(1).fillna(0)
        df["日收"] = pos * df["收盘"].pct_change()
        actual_ret = (1 + df["日收"]).cumprod().iloc[-1]

        # ---- 1000 次随机策略 ----
        n_days = len(df)
        # 随机策略：每天随机决定持仓，但保持跟实际策略同样的平均持仓比例
        actual_pos_rate = df["持仓"].mean()
        np.random.seed(42)
        rand_rets = []
        for _ in range(1000):
            rand_pos = np.random.choice([0, 1], size=n_days, p=[1-actual_pos_rate, actual_pos_rate])
            rand_pos = pd.Series(rand_pos, index=df.index).shift(1).fillna(0)
            rand_日收 = rand_pos * df["收盘"].pct_change()
            rand_ret = (1 + rand_日收).cumprod().iloc[-1]
            rand_rets.append(rand_ret)

        # ---- p 值 ----
        rand_rets = np.array(rand_rets)
        p_value = (rand_rets >= actual_ret).mean()
        avg_rand = rand_rets.mean()

        print(f"  {label}:")
        print(f"    实际策略: {actual_ret*100:.1f}%   随机均值: {avg_rand*100:.1f}%")
        print(f"    跑赢 {p_value*100:.0f}% 随机 → p={p_value:.3f}  {'✅显著' if p_value<0.05 else '❌不显著'}  {'' if p_value>=0.05 else '(p<0.05)'}")
        print()

    print("=" * 72)
