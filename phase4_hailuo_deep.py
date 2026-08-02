"""
海螺水泥 600585 深度分析
问题：买持 -54%，策略 +45%，怎么做到的？
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("data/600585.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df["收盘"] = pd.to_numeric(df["close"])
df["ma"] = df["收盘"].rolling(20).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

# ---- 策略 ----
df["持仓"] = 0; in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev < -0.05: in_pos = True
    elif in_pos and dev > 0.05: in_pos = False
    df.iloc[i, df.columns.get_loc("持仓")] = int(in_pos)

pos = df["持仓"].shift(1).fillna(0)
df["策略日收"] = pos * df["收盘"].pct_change()
df["策略净值"] = (1 + df["策略日收"]).cumprod() * 100_000
df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * 100_000

# ---- 逐笔交易 ----
print("海螺水泥 — 每年走势")
print("=" * 60)
for y in range(2020, 2027):
    try:
        yr = df.loc[str(y)]
    except KeyError:
        continue
    if len(yr) == 0: continue
    s, e = yr["收盘"].iloc[0], yr["收盘"].iloc[-1]
    hi, lo = yr["收盘"].max(), yr["收盘"].min()
    print(f"  {y}: {s:.1f}→{e:.1f} ({(e/s-1)*100:+.1f}%)  振幅{(hi/lo-1)*100:.0f}%")

print(f"\n  全程: {df['收盘'].iloc[0]:.1f}→{df['收盘'].iloc[-1]:.1f} ({(df['收盘'].iloc[-1]/df['收盘'].iloc[0]-1)*100:+.1f}%)")

print("\n逐笔交易:")
print("-" * 60)
entry = None; t = 1; wins = losses = 0
for i in range(1, len(df)):
    prev, curr = df["持仓"].iloc[i-1], df["持仓"].iloc[i]
    if prev == 0 and curr == 1:
        entry = (df.index[i], df["收盘"].iloc[i])
    elif prev == 1 and curr == 0 and entry:
        chg = (df["收盘"].iloc[i] / entry[1] - 1) * 100
        days = (df.index[i] - entry[0]).days
        tag = "✅" if chg > 0 else "❌"
        if chg > 0: wins += 1
        else: losses += 1
        print(f"  #{t}: {entry[0].date()}→{df.index[i].date()}  {days:>3}天  {entry[1]:.1f}→{df['收盘'].iloc[i]:.1f}  {chg:>+5.1f}%  {tag}")
        t += 1; entry = None

print(f"\n  赢{wins}次 / 输{losses}次")
print(f"  策略: {(df['策略净值'].iloc[-1]/100000-1)*100:+.1f}%")
print(f"  买持: {(df['买持净值'].iloc[-1]/100000-1)*100:+.1f}%")

# ---- 画图 ----
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.6, 0.4],
    subplot_titles=("策略 vs 买入持有", "收盘价 + 偏离度"))

# 上行：净值对比
fig.add_trace(go.Scatter(x=df.index, y=df["策略净值"], name="策略净值",
    line=dict(color="#2ca02c", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["买持净值"], name="买入持有",
    line=dict(color="#999", width=1, dash="dash")), row=1, col=1)
fig.add_hline(y=100_000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)

# 标记买入点
entries = df[df["持仓"].diff() == 1]
exits = df[df["持仓"].diff() == -1]
fig.add_trace(go.Scatter(x=entries.index, y=df.loc[entries.index, "策略净值"],
    mode="markers", marker=dict(color="green", symbol="triangle-up", size=10),
    name="买入", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=exits.index, y=df.loc[exits.index, "策略净值"],
    mode="markers", marker=dict(color="red", symbol="triangle-down", size=10),
    name="卖出", showlegend=False), row=1, col=1)

# 下行：收盘价 + 偏离度
fig.add_trace(go.Scatter(x=df.index, y=df["收盘"], name="收盘价",
    line=dict(color="black", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["偏离"]*100, name="偏离度(%)",
    line=dict(color="#ff7f0e", width=0.8), yaxis="y2"), row=2, col=1)
fig.add_hline(y=-5, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
fig.add_hline(y=5, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

fig.update_layout(
    title="海螺水泥 600585 — 双向均值回归深度分析",
    hovermode="x unified", showlegend=True,
    font=dict(size=12),
)
fig.update_yaxes(title_text="净值 (元)", row=1, col=1)
fig.update_yaxes(title_text="价格 (元)", row=2, col=1)
fig.show()
