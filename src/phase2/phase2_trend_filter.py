"""
Phase 2.2 延伸 — 加趋势过滤器
问题：金叉在震荡期反复打脸，如果只在趋势期交易呢？
方法：用 MA60 判断"长期趋势"，只在趋势明确时启用金叉策略
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

SHORT, LONG, TREND = 5, 20, 60
INIT = 100_000

# 双均线
df["ma_short"] = df["收盘"].rolling(SHORT).mean()
df["ma_long"] = df["收盘"].rolling(LONG).mean()
df["ma_trend"] = df["收盘"].rolling(TREND).mean()

# 趋势判断：收盘价 > MA60 = 上升趋势，否则 = 震荡/下降
df["in_trend"] = df["收盘"] > df["ma_trend"]

# 金叉死叉
df["cross"] = df["ma_short"] - df["ma_long"]
df["cross_yesterday"] = df["cross"].shift(1)
df["信号"] = 0
df.loc[(df["cross_yesterday"] <= 0) & (df["cross"] > 0), "信号"] = 1
df.loc[(df["cross_yesterday"] >= 0) & (df["cross"] < 0), "信号"] = -1

# ---- 策略 A：无脑金叉（跟刚才一样）----
df["持仓A"] = df["信号"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# ---- 策略 B：加了趋势过滤 ----
# 只在趋势期间才看金叉信号，震荡期强制空仓
df["信号B"] = df["信号"].copy()
df.loc[~df["in_trend"], "信号B"] = 0                # 震荡期，所有信号无效
# 趋势结束时强制卖出
df.loc[df["in_trend"] == False, "信号B"] = 0
# 有一个问题：如果趋势结束时正在持仓，应该清仓
# 用 -2 标记"趋势结束强制清仓"
df.loc[(df["in_trend"].shift(1) == True) & (df["in_trend"] == False), "信号B"] = -2
df["持仓B"] = df["信号B"].replace(0, np.nan).replace(-2, 0).ffill().fillna(0).clip(lower=0)

# ---- 策略 C：只在趋势期间持仓，但用原策略（收盘价 > MA20）----
df["信号C"] = 0
df.loc[df["in_trend"] & (df["收盘"] > df["ma_long"]), "信号C"] = 1
df.loc[(df["in_trend"].shift(1) == True) & (df["in_trend"] == False), "信号C"] = -2
df["持仓C"] = df["信号C"].replace(0, np.nan).replace(-2, 0).ffill().fillna(0).clip(lower=0)

# 回测
for label, pos_col in [("A", "持仓A"), ("B", "持仓B"), ("C", "持仓C")]:
    pos = df[pos_col].shift(1).fillna(0)
    df[f"{label}_日收"] = pos * df["收盘"].pct_change()
    df[f"{label}_净值"] = (1 + df[f"{label}_日收"]).cumprod() * INIT

# 买入持有
bh_cum = (1 + df["收盘"].pct_change()).cumprod()
df["买持净值"] = bh_cum * INIT

# 算指标
def calc(net):
    ret = (net.iloc[-1] / INIT - 1) * 100
    days = len(net)
    ann = ((net.iloc[-1] / INIT) ** (252 / days) - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    trades = ((net.index.notnull()) & (net > 0)).sum()
    return ret, ann, dd

print("=" * 60)
print(f"  趋势过滤器测试 — 只在 MA{TREND} 上升趋势中交易")
print("=" * 60)

rA, aA, dA = calc(df["A_净值"])
rB, aB, dB = calc(df["B_净值"])
rC, aC, dC = calc(df["C_净值"])
bh_r, bh_a, _ = calc(df["买持净值"])

posA = (df["持仓A"].shift(1).fillna(0) == 1).sum()
posB = (df["持仓B"].shift(1).fillna(0) == 1).sum()
posC = (df["持仓C"].shift(1).fillna(0) == 1).sum()
trend_days = df["in_trend"].sum()

print(f"  在趋势中的天数: {trend_days} / {len(df)} ({trend_days/len(df)*100:.0f}%)")
print()
print(f"{'':<22} {'无脑金叉':<15} {'金叉+趋势过滤':<15} {'原策略+趋势过滤':<15}")
print(f"  总收益        {rA:>+10.2f}%      {rB:>+10.2f}%        {rC:>+10.2f}%")
print(f"  年化          {aA:>+10.2f}%      {aB:>+10.2f}%        {aC:>+10.2f}%")
print(f"  最大回撤      {dA:>+10.2f}%      {dB:>+10.2f}%        {dC:>+10.2f}%")
print(f"  持仓天数      {posA:>10}       {posB:>10}          {posC:>10}")
print(f"\n  买入持有: {bh_r:+.2f}%")

# 画图
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.5, color="#ccc", alpha=0.5)
ax1.plot(df.index, df["ma_trend"], linewidth=1.2, color="purple", label=f"MA{TREND}（趋势线）")
ax1.fill_between(df.index, df["收盘"].min(), df["收盘"].max(),
                 where=df["in_trend"], alpha=0.08, color="green", label="趋势期（策略活跃）")
ax1.fill_between(df.index, df["收盘"].min(), df["收盘"].max(),
                 where=~df["in_trend"], alpha=0.05, color="red", label="震荡期（策略休眠）")
ax1.set_title(f"市场状态识别：MA{TREND} 以上 = 趋势，以下 = 震荡", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df.index, df["A_净值"], linewidth=1.0, color="gray", alpha=0.6, label=f"无脑金叉 ({rA:+.1f}%)")
ax2.plot(df.index, df["B_净值"], linewidth=1.2, color="#2ca02c", label=f"金叉+趋势过滤 ({rB:+.1f}%)")
ax2.plot(df.index, df["C_净值"], linewidth=1.2, color="#1f77b4", label=f"原策略+趋势过滤 ({rC:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.4, label=f"买入持有 ({bh_r:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("资金曲线对比", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_trend_filter.png", dpi=150)
print("\n图已保存至 figures/phase2_trend_filter.png")
