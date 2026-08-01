"""
Phase 3.2 实战 — 三种仓位管理对比

同一个双向均值回归策略，不同下注方式：
  A: 二值（0 or 100%）— 你现在的方式
  B: 固定比例（每次 25%）— 半凯利
  C: 信号强度（偏离越大仓位越大）
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

MA, INIT, THRESHOLD = 20, 100_000, 0.05
df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

# ============================================================
# 策略 A：二值（0/1）
# ============================================================
df["仓位A"] = 0
in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev <= -THRESHOLD:
        in_pos = True
    elif in_pos and dev >= THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("仓位A")] = float(in_pos)

# ============================================================
# 策略 B：固定比例（半凯利 ~25%）
# ============================================================
KELLY_FRAC = 0.25
df["仓位B"] = 0.0
in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev <= -THRESHOLD:
        in_pos = True
    elif in_pos and dev >= THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("仓位B")] = float(in_pos) * KELLY_FRAC

# ============================================================
# 策略 C：信号强度（偏离越大 → 仓位越大）
# ============================================================
# 偏离 -5% → 仓位 20%，偏离 -10% → 仓位 40%，偏离 -15% → 仓位 60%
# 最多加到 MAX_POS
MAX_POS = 0.60
df["仓位C"] = 0.0
in_pos = False
base_dev = 0  # 入场时的偏离

for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]

    if not in_pos and dev <= -THRESHOLD:
        in_pos = True
        base_dev = dev
        # 仓位 = min(|偏离| / 最大偏离阈值 × 最大仓位, 最大仓位)
        pos_size = min(abs(dev) / 0.20 * MAX_POS, MAX_POS)
        pos_size = max(pos_size, 0.10)  # 最少 10%
        df.iloc[i, df.columns.get_loc("仓位C")] = pos_size

    elif in_pos:
        # 持有期间可以根据偏离变化调整仓位
        remain_size = min(abs(dev) / 0.20 * MAX_POS, MAX_POS)
        remain_size = max(remain_size, 0.0)

        if dev >= THRESHOLD:
            in_pos = False
            df.iloc[i, df.columns.get_loc("仓位C")] = 0.0
        else:
            df.iloc[i, df.columns.get_loc("仓位C")] = remain_size

# ============================================================
# 回测
# ============================================================
for label, col in [("A", "仓位A"), ("B", "仓位B"), ("C", "仓位C")]:
    pos = df[col].shift(1).fillna(0)
    df[f"{label}_日收"] = pos * df["收盘"].pct_change()
    df[f"{label}_净值"] = (1 + df[f"{label}_日收"]).cumprod() * INIT

df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * INIT

def stats(net, pos_col):
    net = df[net]
    ret = (net.iloc[-1] / INIT - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    avg_pos = df[pos_col].mean() * 100
    return ret, dd, avg_pos

# ============================================================
# 输出
# ============================================================
print("=" * 65)
print("  Phase 3.2 实战 — 三种仓位管理")
print("=" * 65)
print()
print(f"{'方式':<20} {'收益':>8} {'回撤':>8} {'平均仓位':>8}")
print("-" * 50)

rA, dA, pA = stats("A_净值", "仓位A")
rB, dB, pB = stats("B_净值", "仓位B")
rC, dC, pC = stats("C_净值", "仓位C")
bh_r, _, _ = stats("买持净值", "仓位A")  # BH 不适用，随便传

print(f"  {'A: 二值(0/100%)':<20} {rA:>+7.2f}% {dA:>+7.2f}% {pA:>5.0f}%")
print(f"  {'B: 固定25%':<20} {rB:>+7.2f}% {dB:>+7.2f}% {pB:>5.0f}%")
print(f"  {'C: 信号强度(10-60%)':<20} {rC:>+7.2f}% {dC:>+7.2f}% {pC:>5.0f}%")
print(f"  {'买入持有':<20} {bh_r:>+7.2f}%")
print()

# 仓位分布统计
print(f"  策略 C 仓位分布：")
bins = [0, 0.05, 0.15, 0.25, 0.35, 0.50, 1.0]
labels = ["空仓", "微仓(5-15%)", "小仓(15-25%)", "中仓(25-35%)", "重仓(35-50%)", "满仓(>50%)"]
c_pos = df["仓位C"]
dist = pd.cut(c_pos, bins=bins, labels=labels, right=False).value_counts()
for label, count in dist.items():
    print(f"    {label}: {count} 天 ({count/len(df)*100:.0f}%)")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax1 = axes[0]
for net_col, label, color, lw in [
    ("A_净值", f"A: 二值 ({rA:+.1f}%)", "#d62728", 1.0),
    ("B_净值", f"B: 固定25% ({rB:+.1f}%)", "#1f77b4", 1.0),
    ("C_净值", f"C: 信号强度 ({rC:+.1f}%)", "#2ca02c", 1.5),
    ("买持净值", f"买入持有 ({bh_r:+.1f}%)", "#999", 0.8),
]:
    ax1.plot(df.index, df[net_col], linewidth=lw, color=color, alpha=0.85, label=label)
ax1.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax1.set_title("三种仓位管理的资金曲线", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.fill_between(df.index, 0, df["仓位C"] * 100, color="#2ca02c", alpha=0.3, label="策略C仓位")
ax2.plot(df.index, df["偏离"] * 100, linewidth=0.8, color="#d62728", alpha=0.5, label="偏离均线(%)")
ax2.axhline(y=-THRESHOLD * 100, color="blue", linewidth=0.5, linestyle="--", label=f"买入线({-THRESHOLD*100:.0f}%)")
ax2.axhline(y=THRESHOLD * 100, color="orange", linewidth=0.5, linestyle="--", label=f"卖出线({THRESHOLD*100:.0f}%)")
ax2.set_ylabel("%")
ax2.set_title("策略C仓位 vs 偏离均线", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase3_position_sizing.png", dpi=150)
print("\n图已保存至 figures/phase3_position_sizing.png")
