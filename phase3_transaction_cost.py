"""
Phase 3.1 — 交易成本对策略的真实侵蚀

把你 Day 2 的策略加上手续费、印花税、滑点，看还剩多少。
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
THRESHOLD = 0.05

# ============================================================
# 成本参数
# ============================================================
COMMISSION = 0.00025   # 佣金 万2.5（买卖都收）
STAMP_TAX = 0.0005     # 印花税 万5（仅卖出）
SLIPPAGE = 0.0001      # 滑点 万1（买卖都收）

BUY_COST = COMMISSION + SLIPPAGE           # 买入成本
SELL_COST = COMMISSION + STAMP_TAX + SLIPPAGE  # 卖出成本
ROUND_COST = BUY_COST + SELL_COST          # 一次完整买卖 ≈ 万10.5

print("=" * 65)
print("  Phase 3.1 — 交易成本分析")
print("=" * 65)
print(f"  买入成本: {BUY_COST*10000:.1f}‱  (佣金+滑点)")
print(f"  卖出成本: {SELL_COST*10000:.1f}‱  (佣金+印花税+滑点)")
print(f"  一次完整买卖: {ROUND_COST*10000:.1f}‱")
print()

# ============================================================
# 回测函数（带成本）
# ============================================================
df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

def backtest_with_cost(data, threshold, cost_on=True):
    """双向均值回归，可选是否扣除成本"""
    data = data.copy()
    data["信号"] = 0
    data["持仓"] = 0
    in_pos = False

    for i in range(1, len(data)):
        dev = data["偏离"].iloc[i]
        prev_in = in_pos
        if not in_pos and dev <= -threshold:
            in_pos = True
        elif in_pos and dev >= threshold:
            in_pos = False
        data.iloc[i, data.columns.get_loc("持仓")] = int(in_pos)

        # 检测状态变化 → 交易信号
        changed = in_pos != prev_in
        data.iloc[i, data.columns.get_loc("信号")] = 1 if changed else 0

    pos = data["持仓"].shift(1).fillna(0)
    daily_ret = pos * data["收盘"].pct_change()

    if cost_on:
        trades = data["信号"].shift(1).fillna(0)  # 昨天信号→今天执行
        # 买入信号（持仓从0变1）→ 扣买入成本
        # 卖出信号（持仓从1变0）→ 扣卖出成本
        buy_mask = (trades == 1) & (pos.diff() > 0)  # 今天开始持仓
        sell_mask = (trades == 1) & (pos.diff() < 0) # 今天结束持仓
        daily_ret = daily_ret.copy()
        daily_ret[buy_mask] = daily_ret[buy_mask] - BUY_COST
        daily_ret[sell_mask] = daily_ret[sell_mask] - SELL_COST

    data["策略日收"] = daily_ret
    data["净值"] = (1 + data["策略日收"]).cumprod() * INIT
    return data

# ============================================================
# 上策略：双向均值回归（你 Day 2 最好的）
# ============================================================
df_no_cost = backtest_with_cost(df, THRESHOLD, cost_on=False)
df_with_cost = backtest_with_cost(df, THRESHOLD, cost_on=True)

n_trades_no = int(df_no_cost["信号"].sum())
n_trades_cost = int(df_with_cost["信号"].sum())

def stats(data, label):
    net = data["净值"]
    ret = (net.iloc[-1] / INIT - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    trades = int(data["信号"].sum())
    cost_total = trades * ROUND_COST * INIT / 2  # 每半次交易 = 一次买卖的一半
    print(f"  {label}:")
    print(f"    收益: {ret:+.2f}%  回撤: {dd:+.2f}%  交易次数: {trades}")

stats(df_no_cost, "无成本")
stats(df_with_cost, "有成本（万10.5/次完整买卖）")

# 成本总侵蚀
ret_no = (df_no_cost["净值"].iloc[-1] / INIT - 1) * 100
ret_cost = (df_with_cost["净值"].iloc[-1] / INIT - 1) * 100
total_cost = ret_no - ret_cost
print(f"\n  交易成本总共吃掉: {total_cost:.2f}% 的收益")
print(f"  每次交易平均吞噬: {total_cost/n_trades_no*10000:.1f}‱")

# ============================================================
# 也测一下简单均线策略
# ============================================================
print()
print("─" * 65)
print("  对比：简单均线策略（收盘 > MA20 就持仓）")
print("─" * 65)

df_simple = df.copy()
df_simple["持仓"] = (df_simple["偏离"] > 0).astype(int)
df_simple["信号"] = df_simple["持仓"].diff().abs()  # 持仓变化 = 交易信号

for cost_on in [False, True]:
    pos = df_simple["持仓"].shift(1).fillna(0)
    dr = pos * df_simple["收盘"].pct_change()
    if cost_on:
        trades = df_simple["信号"].shift(1).fillna(0)
        buy_mask = (trades == 1) & (df_simple["持仓"].diff() > 0)
        sell_mask = (trades == 1) & (df_simple["持仓"].diff() < 0)
        dr = dr.copy()
        dr[buy_mask] = dr[buy_mask] - BUY_COST
        dr[sell_mask] = dr[sell_mask] - SELL_COST
    nv = (1 + dr).cumprod() * INIT
    ret = (nv.iloc[-1] / INIT - 1) * 100
    dd = ((nv - nv.cummax()) / nv.cummax() * 100).min()
    label = "有成本" if cost_on else "无成本"
    n_t = int(df_simple["信号"].sum())
    print(f"  {label}: 收益{ret:+.2f}%  回撤{dd:+.2f}%  交易{n_t}次")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, strategy_name, data_nc, data_wc in [
    (axes[0], "双向均值回归", df_no_cost, df_with_cost),
]:
    ax.plot(data_nc.index, data_nc["净值"], linewidth=1.0, color="#2ca02c",
            label=f"无成本 ({ret_no:+.1f}%)")
    ax.plot(data_wc.index, data_wc["净值"], linewidth=1.0, color="#d62728",
            label=f"有成本 ({ret_cost:+.1f}%)")
    ax.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"交易成本侵蚀：{strategy_name}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

# 右图：成本敏感性——不同交易频率下的成本侵蚀
ax2 = axes[1]
trade_counts = [1, 2, 5, 10, 20, 50, 100, 200]
costs_pct = [n * ROUND_COST * 100 for n in trade_counts]
ax2.plot(trade_counts, costs_pct, marker="o", color="#d62728", linewidth=1.5)
ax2.set_xlabel("交易次数")
ax2.set_ylabel("累计成本 (%)")
ax2.set_title("交易越频繁，成本越高", fontsize=12)
ax2.axhline(y=5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax2.text(trade_counts[-1]+5, 5, "5%", fontsize=8, color="gray")
ax2.grid(True, alpha=0.2)
# 标注你策略的位置
n_trades_mean_rev = n_trades_no
ax2.scatter([n_trades_mean_rev], [n_trades_mean_rev * ROUND_COST * 100],
            color="blue", s=100, zorder=5)
ax2.annotate(f"你的策略\n({n_trades_mean_rev}次交易)",
             (n_trades_mean_rev, n_trades_mean_rev * ROUND_COST * 100),
             textcoords="offset points", xytext=(10, -15), fontsize=9, color="blue")

plt.tight_layout()
plt.savefig("figures/phase3_transaction_cost.png", dpi=150)
print("\n图已保存至 figures/phase3_transaction_cost.png")
