"""
Phase 2.2 — 金叉 + 止损
问：金叉买入，亏 20% 止损，和死叉卖出比，哪个更好？
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

SHORT, LONG, STOP_LOSS = 5, 20, -0.20   # 止损 20%
INIT = 100_000

# 均线交叉
df["ma_short"] = df["收盘"].rolling(SHORT).mean()
df["ma_long"] = df["收盘"].rolling(LONG).mean()
df["cross"] = df["ma_short"] - df["ma_long"]
df["cross_y"] = df["cross"].shift(1)
df["信号"] = 0
df.loc[(df["cross_y"] <= 0) & (df["cross"] > 0), "信号"] = 1    # 金叉买
df.loc[(df["cross_y"] >= 0) & (df["cross"] < 0), "信号"] = -1   # 死叉卖

# ---- 策略 A：金叉买 / 死叉卖 ----
df["持仓A"] = df["信号"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# ---- 策略 B：金叉买 / 止损卖（没有死叉） ----
# 模拟：每次金叉买入，记录买入价。如果之后任何一天收盘价跌破买入价的 80%，止损。
# 金叉进来 → 要么下一次金叉重新买入（止损后再等新金叉）
# 持仓状态：0=空仓, 1=持仓
df["持仓B"] = 0
entry_price = np.nan
for i in range(1, len(df)):
    today_signal = df["信号"].iloc[i]
    today_close = df["收盘"].iloc[i]
    prev_close = df["收盘"].iloc[i - 1]
    prev_position = df["持仓B"].iloc[i - 1]

    # 空仓时：金叉 → 买入
    if prev_position == 0 and today_signal == 1:
        df.iloc[i, df.columns.get_loc("持仓B")] = 1
        entry_price = today_close
        continue

    # 持仓时：止损条件 或 金叉重新进入
    if prev_position == 1:
        # 止损：比买入价跌了 20%
        if entry_price > 0 and (today_close / entry_price - 1) <= STOP_LOSS:
            df.iloc[i, df.columns.get_loc("持仓B")] = 0
            entry_price = np.nan
            continue
        # 死叉卖出（仅对比用，策略 B 其实不用死叉，但你想对比所以保留）
        if today_signal == -1:
            df.iloc[i, df.columns.get_loc("持仓B")] = 0
            entry_price = np.nan
            continue
        # 继续持仓
        df.iloc[i, df.columns.get_loc("持仓B")] = 1

# ---- 策略 C：纯止损卖，不死叉 ----
df["持仓C"] = 0
entry_price_c = np.nan
for i in range(1, len(df)):
    today_signal = df["信号"].iloc[i]
    today_close = df["收盘"].iloc[i]
    prev_position = df["持仓C"].iloc[i - 1]

    # 空仓时：金叉买入
    if prev_position == 0 and today_signal == 1:
        df.iloc[i, df.columns.get_loc("持仓C")] = 1
        entry_price_c = today_close
        continue

    # 持仓时：只有止损，不死叉
    if prev_position == 1:
        if entry_price_c > 0 and (today_close / entry_price_c - 1) <= STOP_LOSS:
            df.iloc[i, df.columns.get_loc("持仓C")] = 0
            entry_price_c = np.nan
            continue
        df.iloc[i, df.columns.get_loc("持仓C")] = 1

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

posA = (df["持仓A"].shift(1).fillna(0) == 1).sum()
posB = (df["持仓B"].shift(1).fillna(0) == 1).sum()
posC = (df["持仓C"].shift(1).fillna(0) == 1).sum()

# 数止损触发次数
stop_count = ((df["持仓C"].shift(1) == 1) & (df["持仓C"] == 0) & (df["信号"] != -1)).sum()

print("=" * 60)
print(f"  金叉策略 — 止损 {abs(STOP_LOSS)*100:.0f}%")
print("=" * 60)
print(f"{'':<22} {'死叉卖出':<15} {'死叉+止损':<15} {'纯止损(无死叉)':<15}")
print(f"  总收益        {rA:>+10.2f}%      {rB:>+10.2f}%         {rC:>+10.2f}%")
print(f"  最大回撤      {dA:>+10.2f}%      {dB:>+10.2f}%         {dC:>+10.2f}%")
print(f"  持仓天数      {posA:>10}       {posB:>10}          {posC:>10}")
print(f"\n  买入持有: {bh_r:+.2f}%")
print(f"  纯止损触发次数: {stop_count} 次")

# 画图
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.5, color="#ccc", alpha=0.5)
ax1.plot(df.index, df["ma_short"], linewidth=0.6, alpha=0.5, label=f"MA{SHORT}")
ax1.plot(df.index, df["ma_long"], linewidth=1.0, alpha=0.5, label=f"MA{LONG}")
buy_sig = df[df["信号"] == 1]
ax1.scatter(buy_sig.index, buy_sig["收盘"], marker="^", color="green", s=30, alpha=0.6, label=f"金叉 ({len(buy_sig)}次)")
ax1.set_title("金叉买入信号", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df.index, df["A_净值"], linewidth=0.8, color="gray", alpha=0.5, label=f"死叉卖出 ({rA:+.1f}%)")
ax2.plot(df.index, df["B_净值"], linewidth=1.0, color="#ff7f0e", alpha=0.7, label=f"死叉+止损 ({rB:+.1f}%)")
ax2.plot(df.index, df["C_净值"], linewidth=1.2, color="#2ca02c", label=f"纯止损 ({rC:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3, label=f"买入持有 ({bh_r:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("资金曲线对比", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_stop_loss.png", dpi=150)
print("\n图已保存至 figures/phase2_stop_loss.png")
