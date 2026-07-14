"""
Phase 2.3 — 成交量过滤器 & 独立信号
A: 金叉 + 放量确认（量不够不买）
B: 缩量洗盘 → 放量阳线 = 独立买入信号
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

SHORT, LONG = 5, 20
INIT = 100_000

# 基础均线
df["ma_short"] = df["收盘"].rolling(SHORT).mean()
df["ma_long"] = df["收盘"].rolling(LONG).mean()
df["cross"] = df["ma_short"] - df["ma_long"]
df["cross_y"] = df["cross"].shift(1)

# 金叉信号
df["信号_gc"] = 0
df.loc[(df["cross_y"] <= 0) & (df["cross"] > 0), "信号_gc"] = 1
df.loc[(df["cross_y"] >= 0) & (df["cross"] < 0), "信号_gc"] = -1

# 成交量中位数（滚动 20 天）
df["vol_med"] = df["成交量"].rolling(20).median()
df["vol_ratio"] = df["成交量"] / df["vol_med"]    # >1 = 比平时活跃, <1 = 比平时冷清

# ===== A：金叉 + 放量确认 =====
df["信号A"] = df["信号_gc"].copy()
# 金叉那天成交量必须 > 中位数 × 1.2，否则信号作废
df.loc[(df["信号A"] == 1) & (df["vol_ratio"] < 1.2), "信号A"] = 0
df["持仓A"] = df["信号A"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# ===== B：缩量洗盘 → 放量阳线（独立信号，不依赖均线）=====
# 条件1：缩量阴跌 —— 前面至少 2 天都是缩量 + 阴线
# 条件2：放量阳线 —— 今天成交量 > 中位数 × 1.5 + 收阳
df["信号B"] = 0

for i in range(5, len(df)):
    # 过去 3 天：每天成交量 < 中位数且收盘下跌
    prev_days_chop = True
    for j in range(1, 4):
        if i - j < 0:
            prev_days_chop = False
            break
        vol_low = df["vol_ratio"].iloc[i - j] < 1.0
        price_down = df["收盘"].iloc[i - j] < df["收盘"].iloc[i - j - 1]
        if not (vol_low and price_down):
            prev_days_chop = False
            break

    # 今天：放量阳线
    if prev_days_chop:
        today_vol_spike = df["vol_ratio"].iloc[i] > 1.5
        today_up = df["收盘"].iloc[i] > df["开盘"].iloc[i]
        if today_vol_spike and today_up:
            df.iloc[i, df.columns.get_loc("信号B")] = 1

# B 策略：买进后持有 N 天再卖出（没有固定卖出信号）
df["持仓B"] = 0
hold_counter = 0
for i in range(len(df)):
    if df["信号B"].iloc[i] == 1:
        hold_counter = 10    # 持有 10 个交易日
    if hold_counter > 0:
        df.iloc[i, df.columns.get_loc("持仓B")] = 1
        hold_counter -= 1

# ---- 基准：无脑金叉 ----
df["持仓_base"] = df["信号_gc"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# 回测
for label, pos_col in [("base", "持仓_base"), ("A", "持仓A"), ("B", "持仓B")]:
    pos = df[pos_col].shift(1).fillna(0)
    df[f"{label}_日收"] = pos * df["收盘"].pct_change()
    df[f"{label}_净值"] = (1 + df[f"{label}_日收"]).cumprod() * INIT

df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * INIT

def calc(net):
    ret = (net.iloc[-1] / INIT - 1) * 100
    days = len(net)
    ann = ((net.iloc[-1] / INIT) ** (252 / days) - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    return ret, ann, dd

r0, a0, d0 = calc(df["base_净值"])
rA, aA, dA = calc(df["A_净值"])
rB, aB, dB = calc(df["B_净值"])
bh_r, _, _ = calc(df["买持净值"])

p0 = (df["持仓_base"].shift(1).fillna(0) == 1).sum()
pA = (df["持仓A"].shift(1).fillna(0) == 1).sum()
pB = (df["持仓B"].shift(1).fillna(0) == 1).sum()

sigB_count = (df["信号B"] == 1).sum()
gc_total = (df["信号_gc"] == 1).sum()
gc_filtered = ((df["信号_gc"] == 1) & (df["信号A"] == 1)).sum()

print("=" * 60)
print(f"  成交量过滤器 & 缩量洗盘信号")
print("=" * 60)
print(f"{'':<22} {'无脑金叉':<15} {'金叉+放量':<15} {'缩量洗盘':<15}")
print(f"  总收益        {r0:>+10.2f}%      {rA:>+10.2f}%         {rB:>+10.2f}%")
print(f"  最大回撤      {d0:>+10.2f}%      {dA:>+10.2f}%         {dB:>+10.2f}%")
print(f"  持仓天数      {p0:>10}       {pA:>10}          {pB:>10}")
print(f"\n  买入持有: {bh_r:+.2f}%")
print(f"  金叉总次数: {gc_total} → 过滤后: {gc_filtered}（淘汰 {gc_total - gc_filtered} 次没放量的）")
print(f"  缩量洗盘信号触发: {sigB_count} 次")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 第1行：成交量柱状图 + 信号标记
ax1 = axes[0]
colors = ["#2ca02c" if df["收盘"].iloc[i] >= df["开盘"].iloc[i] else "#d62728"
          for i in range(len(df))]
ax1.bar(df.index, df["成交量"].values / 1e6, color=colors, alpha=0.5, width=1)
ax1.axhline(y=df["vol_med"].mean() / 1e6, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
# 标记缩量洗盘信号
sigB_idx = df[df["信号B"] == 1].index
ax1.scatter(sigB_idx, [df["vol_med"].mean() / 1e6 * 2] * len(sigB_idx),
            marker="^", color="blue", s=80, zorder=5, label=f"缩量洗盘信号 ({sigB_count}次)")
ax1.set_ylabel("成交量（百万）")
ax1.set_title("成交量 & 缩量洗盘信号", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

# 第2行：价格 + 信号点对比
ax2 = axes[1]
ax2.plot(df.index, df["收盘"], linewidth=0.5, color="#ccc", alpha=0.5)
gc_buy = df[(df["信号A"] == 1)]
gc_skip = df[(df["信号_gc"] == 1) & (df["信号A"] == 0)]
ax2.scatter(gc_buy.index, gc_buy["收盘"], marker="^", color="green", s=50, zorder=5, label=f"金叉+放量 ({len(gc_buy)}次)")
ax2.scatter(gc_skip.index, gc_skip["收盘"], marker="^", color="gray", s=20, alpha=0.3, label=f"金叉被过滤 ({len(gc_skip)}次)")
ax2.scatter(sigB_idx, df.loc[sigB_idx, "收盘"], marker="o", color="blue", s=50, zorder=5, alpha=0.7, label=f"缩量洗盘 ({sigB_count}次)")
ax2.set_title("金叉信号：绿色=放量通过 / 灰色=缩量淘汰", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

# 第3行：资金曲线
ax3 = axes[2]
ax3.plot(df.index, df["base_净值"], linewidth=0.8, color="gray", alpha=0.4, label=f"无脑金叉 ({r0:+.1f}%)")
ax3.plot(df.index, df["A_净值"], linewidth=1.0, color="#2ca02c", label=f"金叉+放量 ({rA:+.1f}%)")
ax3.plot(df.index, df["B_净值"], linewidth=1.2, color="#1f77b4", label=f"缩量洗盘 ({rB:+.1f}%)")
ax3.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3, label=f"买入持有 ({bh_r:+.1f}%)")
ax3.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("资金曲线对比", fontsize=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_volume.png", dpi=150)
print("\n图已保存至 figures/phase2_volume.png")
