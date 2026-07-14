"""
Phase 2.3 延伸 — 趋势跟随 vs 均值回归
A: 追涨——价格 > 均线就买（你一直在做的）
B: 抄底——价格跌破均线 5% 才买，回到均线就卖（反过来）
C: 追涨+抄底混合——跌超5%买，涨超均线5%卖
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

MA = 20
THRESHOLD = 0.05    # 偏离均线 5%
INIT = 100_000

df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1    # >0 = 在均线上方, <0 = 在均线下方

# ---- 策略 A：追涨（价格 > 均线 → 持仓）----
df["持仓A"] = (df["偏离"] > 0).astype(int)

# ---- 策略 B：抄底（偏离 < -5% → 买，回到均线 → 卖）----
df["持仓B"] = 0
in_position = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    # 空仓时：跌到比均线低 5% → 买
    if not in_position and dev <= -THRESHOLD:
        in_position = True
    # 持仓时：回到均线 → 卖
    elif in_position and dev >= 0:
        in_position = False
    df.iloc[i, df.columns.get_loc("持仓B")] = int(in_position)

# ---- 策略 C：双向 —— 跌超买、涨超卖 ----
df["持仓C"] = 0
in_position_c = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_position_c and dev <= -THRESHOLD:
        in_position_c = True
    elif in_position_c and dev >= THRESHOLD:
        in_position_c = False
    df.iloc[i, df.columns.get_loc("持仓C")] = int(in_position_c)

# 回测
for label, pos_col in [("A", "持仓A"), ("B", "持仓B"), ("C", "持仓C")]:
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

rA, aA, dA = calc(df["A_净值"])
rB, aB, dB = calc(df["B_净值"])
rC, aC, dC = calc(df["C_净值"])
bh_r, _, _ = calc(df["买持净值"])

pA = (df["持仓A"].shift(1).fillna(0) == 1).sum()
pB = (df["持仓B"].shift(1).fillna(0) == 1).sum()
pC = (df["持仓C"].shift(1).fillna(0) == 1).sum()

# 数 B 策略交易次数
trades_B = 0
in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev <= -THRESHOLD:
        trades_B += 1
        in_pos = True
    elif in_pos and dev >= 0:
        in_pos = False

print("=" * 60)
print(f"  趋势跟随 vs 均值回归（MA{MA}，阈值 {THRESHOLD*100:.0f}%）")
print("=" * 60)
print(f"{'':<22} {'追涨':<15} {'抄底':<15} {'双向':<15}")
print(f"  总收益        {rA:>+10.2f}%      {rB:>+10.2f}%         {rC:>+10.2f}%")
print(f"  最大回撤      {dA:>+10.2f}%      {dB:>+10.2f}%         {dC:>+10.2f}%")
print(f"  持仓天数      {pA:>10}       {pB:>10}          {pC:>10}")
print(f"  交易次数      {'持续持仓':>10}       {trades_B:>10}")
print(f"\n  买入持有: {bh_r:+.2f}%")

# 画图
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.5, color="#ccc", alpha=0.5)
ax1.plot(df.index, df["ma"], linewidth=1.0, color="purple", label=f"MA{MA}")
# 买入区域
ax1.fill_between(df.index, df["收盘"].min(), df["收盘"].max(),
                 where=df["偏离"] <= -THRESHOLD, alpha=0.1, color="blue", label=f"抄底区（偏离<{-THRESHOLD*100:.0f}%）")
ax1.fill_between(df.index, df["收盘"].min(), df["收盘"].max(),
                 where=df["偏离"] >= THRESHOLD, alpha=0.1, color="red", label=f"高估区（偏离>{THRESHOLD*100:.0f}%）")
ax1.set_title("均值回归逻辑：蓝区抄底 → 回到均线卖出", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df.index, df["A_净值"], linewidth=1.0, color="#d62728", label=f"追涨 ({rA:+.1f}%)")
ax2.plot(df.index, df["B_净值"], linewidth=1.2, color="#1f77b4", label=f"抄底 ({rB:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3, label=f"买入持有 ({bh_r:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("追涨 vs 抄底", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

ax3 = axes[2]
ax3.plot(df.index, df["A_净值"], linewidth=0.8, color="gray", alpha=0.4, label=f"追涨 ({rA:+.1f}%)")
ax3.plot(df.index, df["B_净值"], linewidth=1.0, color="#1f77b4", alpha=0.7, label=f"抄底 ({rB:+.1f}%)")
ax3.plot(df.index, df["C_净值"], linewidth=1.2, color="#2ca02c", label=f"双向 ({rC:+.1f}%)")
ax3.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3, label=f"买入持有 ({bh_r:+.1f}%)")
ax3.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("追涨 vs 抄底 vs 双向", fontsize=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_mean_reversion.png", dpi=150)
print("\n图已保存至 figures/phase2_mean_reversion.png")
