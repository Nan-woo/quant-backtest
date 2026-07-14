"""
Phase 2.2 — 双均线金叉死叉策略
对比三种信号逻辑：
  A. 原来的 "收盘价 > 均线"（单均线）
  B. 真正的 金叉/死叉（双均线交叉）
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读数据
df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

SHORT, LONG = 5, 20
INIT = 100_000

# ===== 双均线计算 =====
df["ma_short"] = df["收盘"].rolling(SHORT).mean()
df["ma_long"] = df["收盘"].rolling(LONG).mean()

# ===== A：原来的单均线信号 =====
df["signal_old"] = 0
df.loc[df["收盘"] > df["ma_long"], "signal_old"] = 1

# ===== B：真正的金叉死叉 =====
# 金叉：昨天短均线 <= 长均线，今天短均线 > 长均线（交叉发生在今天）
# 死叉：昨天短均线 >= 长均线，今天短均线 < 长均线
df["cross"] = df["ma_short"] - df["ma_long"]          # 正值=短期在上，负值=短期在下
df["cross_yesterday"] = df["cross"].shift(1)           # 昨天的差值

# 金叉买入信号
df["signal_gc"] = 0
df.loc[(df["cross_yesterday"] <= 0) & (df["cross"] > 0), "signal_gc"] = 1   # 买入点
df.loc[(df["cross_yesterday"] >= 0) & (df["cross"] < 0), "signal_gc"] = -1  # 卖出点

# 把信号持久化：金叉之后一直持仓，直到死叉
# 用 cumsum 实现：买入=+1，卖出=-1，持仓状态 = 信号累加
df["position"] = df["signal_gc"].replace(0, np.nan).ffill().fillna(0)
df["position"] = df["position"].clip(lower=0)          # 0=空仓, 1=持仓

# ===== 回测对比 =====
# 旧策略
df["持仓旧"] = df["signal_old"].shift(1).fillna(0)
df["旧日收"] = df["持仓旧"] * df["收盘"].pct_change()
df["旧净值"] = (1 + df["旧日收"]).cumprod() * INIT

# 金叉策略
df["持仓新"] = df["position"].shift(1).fillna(0)
df["新日收"] = df["持仓新"] * df["收盘"].pct_change()
df["新净值"] = (1 + df["新日收"]).cumprod() * INIT

# 买入持有
bh_ret = (df["收盘"].pct_change().fillna(0) + 1).cumprod()
df["买持净值"] = bh_ret * INIT

# ===== 算指标 =====
def calc_metrics(net_val, init=INIT):
    total = (net_val.iloc[-1] / init - 1) * 100
    days = len(net_val)
    ann = ((net_val.iloc[-1] / init) ** (252 / days) - 1) * 100
    peak = net_val.cummax()
    dd = ((net_val - peak) / peak * 100).min()
    return total, ann, dd

old_ret, old_ann, old_dd = calc_metrics(df["旧净值"])
new_ret, new_ann, new_dd = calc_metrics(df["新净值"])
bh_ret_v, bh_ann, _ = calc_metrics(df["买持净值"])

# 交易次数
trades = (df["signal_gc"].abs() == 1).sum()
trade_days = (df["持仓新"] == 1).sum()

print("=" * 55)
print(f"  MA{SHORT}/{LONG} 双均线 — 策略对比")
print("=" * 55)
print(f"{'':<18} {'原策略(收盘>MA)':<18} {'金叉死叉':<18}")
print(f"  总收益        {old_ret:>+13.2f}%   {new_ret:>+13.2f}%")
print(f"  年化收益      {old_ann:>+13.2f}%   {new_ann:>+13.2f}%")
print(f"  最大回撤      {old_dd:>+13.2f}%   {new_dd:>+13.2f}%")
print()
print(f"  买入持有收益: {bh_ret_v:+.2f}%")
print(f"  金叉信号触发: {trades} 次（买入{trades//2}次 + 卖出{trades//2}次）")
print(f"  持仓天数: {trade_days} 天 ({trade_days/len(df)*100:.0f}%)")

# ===== 画图 =====
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 第1行：价格 + 双均线 + 金叉死叉标记
ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.6, color="#ccc", label="收盘价")
ax1.plot(df.index, df["ma_short"], linewidth=0.8, color="#1f77b4", label=f"MA{SHORT}（快）")
ax1.plot(df.index, df["ma_long"], linewidth=1.2, color="#d62728", label=f"MA{LONG}（慢）")
# 标记金叉点
buy_signals = df[df["signal_gc"] == 1]
sell_signals = df[df["signal_gc"] == -1]
ax1.scatter(buy_signals.index, buy_signals["收盘"], marker="^", color="green",
            s=60, zorder=5, label=f"金叉买入 ({len(buy_signals)}次)")
ax1.scatter(sell_signals.index, sell_signals["收盘"], marker="v", color="red",
            s=60, zorder=5, label=f"死叉卖出 ({len(sell_signals)}次)")
ax1.set_title(f"茅台 600519 — MA{SHORT}/{LONG} 金叉死叉", fontsize=13)
ax1.legend(loc="upper left", fontsize=7)
ax1.grid(True, alpha=0.2)

# 第2行：原策略资金曲线
ax2 = axes[1]
ax2.plot(df.index, df["旧净值"], linewidth=1.2, color="#1f77b4", label=f"原策略 ({old_ret:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.2, color="gray", alpha=0.6, label=f"买入持有 ({bh_ret_v:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("原策略：收盘价 > 均线即持仓", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

# 第3行：金叉策略资金曲线
ax3 = axes[2]
ax3.plot(df.index, df["新净值"], linewidth=1.2, color="#2ca02c", label=f"金叉死叉 ({new_ret:+.1f}%)")
ax3.plot(df.index, df["买持净值"], linewidth=1.2, color="gray", alpha=0.6, label=f"买入持有 ({bh_ret_v:+.1f}%)")
ax3.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("双均线金叉死叉策略", fontsize=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_golden_cross.png", dpi=150)
print("\n图已保存至 figures/phase2_golden_cross.png")
