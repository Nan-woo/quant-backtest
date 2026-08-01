"""
159516 半导体ETF — 牛市高抛低吸（用户实盘逻辑回测）

用户假说：
  1. 半导体长期看涨（牛市信念）
  2. 短期做反向T：以开盘价为锚
     - 盘中跌超 5% → 买入
     - 盘中涨超 5% → 卖出
  3. 规则目前是情绪驱动的，需要量化验证

日线近似：用最低价 vs 开盘价判断"跌了 5%"，用最高价 vs 开盘价判断"涨了 5%"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 参数 —— 用户可以改
# ============================================================
DROP_THRESHOLD = -0.05   # 盘中跌多少买（相对开盘价）
RISE_THRESHOLD = 0.05    # 盘中涨多少卖（相对开盘价）
INIT = 100_000
BASE_POS = 0.5           # 底仓 50%：长期看好的部分不动

# ============================================================
# 数据
# ============================================================
df = pd.read_csv("data/159516.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

# ============================================================
# 修复：7/10 1拆2，baostock 前复权未正确调整
# 将拆分前所有价格除以 2，使价格序列连续
# ============================================================
split_date = pd.to_datetime("2026-07-10")
pre_split = df.index < split_date
for col in ["开盘", "收盘", "最高", "最低"]:
    df.loc[pre_split, col] = df.loc[pre_split, col] / 2

# ============================================================
# 信号
# ============================================================
# 盘中最低价 vs 开盘价
df["盘中跌幅"] = df["最低"] / df["开盘"] - 1
# 盘中最高价 vs 开盘价
df["盘中涨幅"] = df["最高"] / df["开盘"] - 1

# 买入信号：盘中跌超阈值
df["买入信号"] = df["盘中跌幅"] <= DROP_THRESHOLD
# 卖出信号：盘中涨超阈值（但只有持仓时才有意义）
df["卖出信号"] = df["盘中涨幅"] >= RISE_THRESHOLD

# ============================================================
# 模拟：底仓 + 反向T
# 底仓始终持有 BASE_POS（50%），T仓在信号触发时进出
# ============================================================
df["底仓"] = BASE_POS
df["T仓"] = 0.0

in_trade = False
for i in range(len(df)):
    if not in_trade and df["买入信号"].iloc[i]:
        in_trade = True
    elif in_trade and df["卖出信号"].iloc[i]:
        in_trade = False
    df.iloc[i, df.columns.get_loc("T仓")] = float(in_trade)

df["总仓位"] = df["底仓"] + df["T仓"]

# 回测
df["日收益"] = df["收盘"].pct_change()
df["策略日收"] = df["总仓位"].shift(1).fillna(BASE_POS) * df["日收益"]
df["策略净值"] = (1 + df["策略日收"]).cumprod() * INIT
df["买持净值"] = (1 + df["日收益"]).cumprod() * INIT

# ============================================================
# 评估
# ============================================================
def calc(net):
    ret = (net.iloc[-1] / INIT - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    return ret, dd

s_ret, s_dd = calc(df["策略净值"])
b_ret, b_dd = calc(df["买持净值"])

buy_sig = df["买入信号"].sum()
sell_sig = df["卖出信号"].sum()
hold_t_pct = df["T仓"].mean() * 100

print("=" * 65)
print("  159516 半导体ETF — 反向T（高抛低吸）")
print("=" * 65)
print(f"  规则：开盘价锚定，跌{DROP_THRESHOLD*100:.0f}%买 → 涨{RISE_THRESHOLD*100:.0f}%卖")
print(f"  底仓: {BASE_POS*100:.0f}%  T仓: 0% or 100%")
print(f"  数据: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}天)")
print()
print(f"  买入信号: {buy_sig} 次")
print(f"  卖出信号: {sell_sig} 次")
print(f"  T仓持仓时间: {hold_t_pct:.1f}%")
print()
print(f"  策略收益: {s_ret:+.2f}%")
print(f"  最大回撤: {s_dd:+.2f}%")
print(f"  买入持有: {b_ret:+.2f}%")
print(f"  超额: {s_ret - b_ret:+.2f}%")

# ============================================================
# 看看每次 T 交易赚了多少
# ============================================================
print()
print("  T 交易记录（买入日 → 卖出日）：")
trades = []
entry_idx = None
for i in range(len(df)):
    if entry_idx is None and df["买入信号"].iloc[i]:
        entry_idx = i
    elif entry_idx is not None and df["卖出信号"].iloc[i]:
        entry_price = df["收盘"].iloc[entry_idx]
        exit_price = df["收盘"].iloc[i]
        ret = (exit_price / entry_price - 1) * 100
        trades.append((df.index[entry_idx], df.index[i], ret))
        entry_idx = None

if entry_idx is not None:
    trades.append((df.index[entry_idx], df.index[-1],
                   (df["收盘"].iloc[-1] / df["收盘"].iloc[entry_idx] - 1) * 100))

total_t = sum(t[2] for t in trades)
win_t = sum(1 for t in trades if t[2] > 0)
for t in trades:
    w = "✓" if t[2] > 0 else "✗"
    print(f"    {t[0].date()} → {t[1].date()}  {t[2]:+.2f}% {w}")
print(f"  T交易总盈亏: {total_t:+.2f}%  胜率: {win_t}/{len(trades)}")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.8, color="black", label="收盘价")
buy_idx = df[df["买入信号"]].index
sell_idx = df[df["卖出信号"]].index
ax1.scatter(buy_idx, df.loc[buy_idx, "收盘"], marker="^", color="green", s=60, zorder=5,
            label=f"买入信号 ({buy_sig}次)")
ax1.scatter(sell_idx, df.loc[sell_idx, "收盘"], marker="v", color="red", s=60, zorder=5,
            label=f"卖出信号 ({sell_sig}次)")
ax1.set_title(f"159516 半导体ETF — 反向T（跌{DROP_THRESHOLD*100:.0f}%买 涨{RISE_THRESHOLD*100:.0f}%卖）", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df.index, df["策略净值"], linewidth=1.2, color="#2ca02c",
         label=f"底仓50%+反向T ({s_ret:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3,
         label=f"买入持有 ({b_ret:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("资金曲线", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_159516_trade.png", dpi=150)
print("\n图已保存至 figures/phase2_159516_trade.png")
