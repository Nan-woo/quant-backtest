"""
Phase 3.3 — 移动止盈 vs 固定止损

对比三种退出方式：
  A: 固定止损（跌 15% 从入场价）
  B: 移动止盈（跌 15% 从最高点）
  C: 无退出（信号卖出）

在双向均值回归策略上跑
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

MA, INIT = 20, 100_000
STOP_LOSS = 0.15    # 15% 止损线

df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

# ============================================================
# 三种退出方式
# ============================================================

def backtest_trailing(data, entry_thresh, exit_thresh, stop_type="none", stop_pct=0.15):
    """
    stop_type: "none" | "fixed" | "trailing"
    """
    data = data.copy()
    data["持仓"] = 0
    data["退出原因"] = ""  # 记录为什么退出

    in_pos = False
    entry_price = 0
    highest_since_entry = 0

    for i in range(1, len(data)):
        dev = data["偏离"].iloc[i]
        price = data["收盘"].iloc[i]

        if not in_pos and dev <= -entry_thresh:
            # 买入
            in_pos = True
            entry_price = price
            highest_since_entry = price
            data.iloc[i, data.columns.get_loc("退出原因")] = "买入"

        elif in_pos:
            # 更新最高点
            if price > highest_since_entry:
                highest_since_entry = price

            # 信号卖出
            signal_sell = dev >= exit_thresh

            # 固定止损
            fixed_stop = (stop_type == "fixed" and
                          (price / entry_price - 1) < -stop_pct)

            # 移动止盈
            trailing_stop = (stop_type == "trailing" and
                             (price / highest_since_entry - 1) < -stop_pct)

            if signal_sell:
                in_pos = False
                data.iloc[i, data.columns.get_loc("退出原因")] = "信号卖出"
            elif fixed_stop:
                in_pos = False
                data.iloc[i, data.columns.get_loc("退出原因")] = "固定止损"
            elif trailing_stop:
                in_pos = False
                data.iloc[i, data.columns.get_loc("退出原因")] = "移动止盈"

        data.iloc[i, data.columns.get_loc("持仓")] = int(in_pos)

    pos = data["持仓"].shift(1).fillna(0)
    data["策略日收"] = pos * data["收盘"].pct_change()
    data["净值"] = (1 + data["策略日收"]).cumprod() * INIT
    return data

# 跑三种
THRESHOLD = 0.05
res_none = backtest_trailing(df, THRESHOLD, THRESHOLD, "none", STOP_LOSS)
res_fixed = backtest_trailing(df, THRESHOLD, THRESHOLD, "fixed", STOP_LOSS)
res_trail = backtest_trailing(df, THRESHOLD, THRESHOLD, "trailing", STOP_LOSS)

# ============================================================
# 评估
# ============================================================
def stats(data, name):
    net = data["净值"]
    ret = (net.iloc[-1] / INIT - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    holds = data["持仓"].shift(1).fillna(0).sum()
    exits = data["退出原因"].value_counts()
    print(f"\n  {name}:")
    print(f"    收益: {ret:+.2f}%  回撤: {dd:+.2f}%  持仓 {holds:.0f} 天")
    for reason, count in exits.items():
        if reason:
            print(f"    {reason}: {count} 次")

print("=" * 65)
print(f"  Phase 3.3 — 移动止盈（{STOP_LOSS*100:.0f}% 回撤线）")
print("=" * 65)
stats(res_none, "无退出保护")
stats(res_fixed, "固定止损")
stats(res_trail, "移动止盈")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 上：资金曲线
ax1 = axes[0]
for data, name, color in [
    (res_none, "无退出", "#999"),
    (res_fixed, "固定止损", "#d62728"),
    (res_trail, "移动止盈", "#2ca02c"),
]:
    ret = (data["净值"].iloc[-1] / INIT - 1) * 100
    ax1.plot(data.index, data["净值"], linewidth=1.2, color=color, alpha=0.8,
             label=f"{name} ({ret:+.1f}%)")
ax1.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax1.set_title(f"移动止盈 vs 固定止损（{STOP_LOSS*100:.0f}% 回撤线）", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.2)

# 下：展示一次交易的移动止盈逻辑
ax2 = axes[1]

# 找最近一次完整的交易来演示
# 用 res_trail 里最后一次买入-卖出
exit_mask = res_trail["退出原因"].isin(["移动止盈", "信号卖出", "固定止损"])
exit_dates = res_trail[exit_mask].index
entry_mask = res_trail["退出原因"] == "买入"
entry_dates = res_trail[entry_mask].index

if len(entry_dates) > 0 and len(exit_dates) > 0:
    # 取最后一段交易
    last_entry = None
    for ed in entry_dates:
        matching_exits = exit_dates[exit_dates > ed]
        if len(matching_exits) > 0:
            last_entry = ed
            last_exit = matching_exits[0]

    if last_entry is not None:
        # 放大这段
        start = max(0, df.index.get_loc(last_entry) - 10)
        end = min(len(df) - 1, df.index.get_loc(last_exit) + 5)
        zoom = df.iloc[start:end+1]

        ax2.plot(zoom.index, zoom["收盘"], linewidth=1.0, color="black", label="收盘价")
        ax2.scatter(last_entry, df.loc[last_entry, "收盘"], marker="^", color="green",
                   s=100, zorder=5, label=f"买入 ({last_entry.date()})")
        ax2.scatter(last_exit, df.loc[last_exit, "收盘"], marker="v", color="red",
                   s=100, zorder=5, label=f"{res_trail.loc[last_exit, '退出原因']} ({last_exit.date()})")

        # 画移动止盈线
        entry_p = df.loc[last_entry, "收盘"]
        segment = df.iloc[df.index.get_loc(last_entry):df.index.get_loc(last_exit)+1]
        highest = segment["收盘"].cummax()
        stop_line = highest * (1 - STOP_LOSS)
        ax2.plot(segment.index, stop_line, linewidth=1.0, color="orange", linestyle="--",
                label=f"移动止盈线 (最高×{1-STOP_LOSS:.0%})")
        ax2.fill_between(segment.index, stop_line, segment["收盘"].min(),
                         alpha=0.1, color="orange")

        ax2.set_title(f"一次交易的移动止盈演示（{last_entry.date()} → {last_exit.date()}）", fontsize=12)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase3_trailing_stop.png", dpi=150)
print("\n图已保存至 figures/phase3_trailing_stop.png")
