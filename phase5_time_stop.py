"""
Phase 5.1 — 时间止损（封装因子1最后一个改进）

逻辑：偏离度买入 + 信号卖出 + 持仓45天仍不盈利 → 强制割掉
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MA, THRESHOLD, TIME_STOP = 20, 0.05, 45
SPLIT = "2024-01-01"

tasks = [
    ("data/600519.csv", "茅台"),
    ("data/600036.csv", "招行"),
    ("data/000725.csv", "京东方"),
    ("data/600585.csv", "海螺"),
]

def backtest(data, ma, th, time_stop_days=None):
    """time_stop_days=None 表示不加时间止损"""
    d = data.copy()
    d["收盘"] = pd.to_numeric(d["close"])
    d["ma"] = d["收盘"].rolling(ma).mean()
    d["偏离"] = d["收盘"] / d["ma"] - 1
    d["持仓"] = 0
    in_pos = False
    entry_price = 0
    days_held = 0   # 持仓天数计数器

    for i in range(1, len(d)):
        dev = d["偏离"].iloc[i]
        price = d["收盘"].iloc[i]

        # ---- 入场 ----
        if not in_pos and dev < -th:
            in_pos = True
            entry_price = price
            days_held = 0

        elif in_pos:
            days_held += 1

            # 卖出条件（优先级：信号 > 时间止损）
            signal_sell = dev > th
            # ===================================================
            # 【你填】时间止损：持仓超过 N 天，还在亏 → 割
            # ===================================================
            time_stop = (
                time_stop_days is not None
                and days_held >= time_stop_days
                and price < entry_price          # 不盈利
            )
            # ===================================================

            if signal_sell or time_stop:
                in_pos = False
                days_held = 0

        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)

    pos = d["持仓"].shift(1).fillna(0)
    d["日收"] = pos * d["收盘"].pct_change()
    d["净值"] = (1 + d["日收"]).cumprod()
    return d

# ============================================================
# 跑
# ============================================================
print("=" * 72)
print(f"  Phase 5 — 时间止损（{TIME_STOP}天不盈利强制退出）")
print("=" * 72)
print()

for file, name in tasks:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    train = df[df.index < SPLIT].copy()
    test = df[df.index >= SPLIT].copy()

    # 训练期
    t_orig = backtest(train, MA, THRESHOLD)
    t_stop = backtest(train, MA, THRESHOLD, TIME_STOP)

    # 测试期（用训练期定好的参数）
    s_orig = backtest(test, MA, THRESHOLD)
    s_stop = backtest(test, MA, THRESHOLD, TIME_STOP)

    # 评估函数
    def eval_backtest(d, prefix):
        net = d["净值"]
        total_ret = (net.iloc[-1] - 1) * 100
        daily = d["日收"].dropna()
        ann_ret = daily.mean() * 252
        ann_vol = daily.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        nv = (1 + d["日收"]).cumprod()
        mdd = ((nv - nv.cummax()) / nv.cummax() * 100).min()
        trades = (d["持仓"].diff() == 1).sum()
        return total_ret, sharpe, mdd, trades

    tr, sh, dd, ts = eval_backtest(t_orig, "train_orig")
    tr2, sh2, dd2, ts2 = eval_backtest(t_stop, "train_stop")
    sr, sh_s, dd_s, ts_s = eval_backtest(s_orig, "test_orig")
    sr2, sh2_s, dd2_s, ts2_s = eval_backtest(s_stop, "test_stop")

    print(f"  【{name}】")
    print(f"    训练期: 原版{tr:+.1f}% S{sh:.2f}  时间止损{tr2:+.1f}% S{sh2:.2f}  交易{ts:.0f}→{ts2:.0f}次")
    print(f"    测试期: 原版{sr:+.1f}% S{sh_s:.2f}  时间止损{sr2:+.1f}% S{sh2_s:.2f}  交易{ts_s:.0f}→{ts2_s:.0f}次")
    arrow = "↑" if sr2 > sr else "↓"
    print(f"    测试期效果: {arrow}")
    print()

# ============================================================
# 画一只股票的对比图（海螺）
# ============================================================
df = pd.read_csv("data/600585.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
d_orig = backtest(df, MA, THRESHOLD)
d_stop = backtest(df, MA, THRESHOLD, TIME_STOP)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
    subplot_titles=("海螺水泥 — 时间止损对比", "持仓对比"))

fig.add_trace(go.Scatter(x=d_orig.index, y=d_orig["净值"]*100000, name="原版",
    line=dict(color="#d62728", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=d_stop.index, y=d_stop["净值"]*100000, name=f"时间止损{TIME_STOP}天",
    line=dict(color="#2ca02c", width=2)), row=1, col=1)
# 买持
bh = (1 + pd.to_numeric(df["close"]).pct_change()).cumprod() * 100000
fig.add_trace(go.Scatter(x=bh.index, y=bh, name="买持",
    line=dict(color="#999", width=0.8, dash="dash")), row=1, col=1)
fig.add_hline(y=100000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)

fig.add_trace(go.Scatter(x=d_orig.index, y=d_orig["持仓"].replace(0, np.nan)*45, name="原版持仓",
    line=dict(color="#d62728", width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=d_stop.index, y=d_stop["持仓"].replace(0, np.nan)*50, name="时间止损持仓",
    line=dict(color="#2ca02c", width=2)), row=2, col=1)

fig.update_layout(title="时间止损 — 拦截 100+ 天困局", hovermode="x unified")
fig.update_yaxes(title_text="净值", row=1, col=1)
fig.update_yaxes(title_text="持仓", row=2, col=1)
fig.show()
