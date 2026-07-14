"""
一个最简单的完整回测系统。
策略：20 日均线金叉买入，死叉卖出。
标的：贵州茅台 600519
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===== 第 1 步：拿数据 =====
df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
# 统一列名：baostock 用小写英文 → 改成中文
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

# ===== 第 2 步：算信号 =====
MA_PERIOD = 15                                      # ← 想改参数只改这一个数字
df["ma"] = df["收盘"].rolling(MA_PERIOD).mean()
df["signal"] = 0                                    # 0 = 空仓, 1 = 持仓
# 金叉：收盘价从下方上穿均线 → 买入
# 死叉：收盘价从上方下穿均线 → 卖出
df.loc[df["收盘"] > df["ma"], "signal"] = 1

# ===== 第 3 步：模拟交易 =====
INIT_CAPITAL = 100_000                              # 初始资金 10 万

df["持仓"] = df["signal"].shift(1)                   # 用昨天的信号决定今天的仓位
df["持仓"] = df["持仓"].fillna(0)                      # 第一天没信号，空仓

df["日收益率"] = df["收盘"].pct_change()              # 股票每天涨跌多少
df["策略日收益"] = df["持仓"] * df["日收益率"]          # 持仓时赚（亏），空仓时不赚不亏
df["策略净值"] = (1 + df["策略日收益"]).cumprod() * INIT_CAPITAL   # 资金曲线
df["买入持有净值"] = (1 + df["日收益率"]).cumprod() * INIT_CAPITAL

# ===== 第 4 步：算指标 =====
total_return = (df["策略净值"].iloc[-1] / INIT_CAPITAL - 1) * 100
bh_return = (df["买入持有净值"].iloc[-1] / INIT_CAPITAL - 1) * 100

# 年化收益率
days = len(df)
annual_return = ((df["策略净值"].iloc[-1] / INIT_CAPITAL) ** (252 / days) - 1) * 100

# 最大回撤
peak = df["策略净值"].cummax()
drawdown = (df["策略净值"] - peak) / peak * 100
max_dd = drawdown.min()

# 胜率：持仓日中，上涨的天数占比
hold_days = df[df["持仓"] == 1]
win_rate = (hold_days["日收益率"] > 0).sum() / len(hold_days) * 100 if len(hold_days) > 0 else 0

# ===== 第 5 步：输出 =====
print("=" * 50)
print(f"  茅台 {MA_PERIOD} 日均线策略 — 回测结果")
print("=" * 50)
print(f"  回测区间: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"  交易天数: {days}")
print(f"  持仓天数: {len(hold_days)} ({len(hold_days)/days*100:.0f}%)")
print()
print(f"  策略总收益:  {total_return:+.2f}%")
print(f"  买入持有:    {bh_return:+.2f}%")
print(f"  年化收益:    {annual_return:+.2f}%")
print(f"  最大回撤:    {max_dd:.2f}%")
print(f"  胜率:        {win_rate:.1f}%")

# ===== 第 6 步：画图 =====
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 上图：价格 + 均线
ax1 = axes[0]
ax1.plot(df.index, df["收盘"], linewidth=0.6, color="#999", label="收盘价")
ax1.plot(df.index, df["ma"], linewidth=1.0, color="#d62728", label=f"{MA_PERIOD} 日均线")
ax1.fill_between(df.index, df["收盘"].min(), df["收盘"].max(),
                 where=df["signal"] == 1, alpha=0.08, color="green", label="持仓区间")
ax1.set_title(f"茅台 600519 — 价格 & {MA_PERIOD} 日均线", fontsize=13)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

# 下图：资金曲线
ax2 = axes[1]
ax2.plot(df.index, df["策略净值"], linewidth=1.2, color="#1f77b4", label="策略净值")
ax2.plot(df.index, df["买入持有净值"], linewidth=1.2, color="#ff7f0e", label="买入持有")
ax2.axhline(y=INIT_CAPITAL, color="gray", linewidth=0.5, linestyle="--")
ax2.set_title("资金曲线对比", fontsize=13)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
filename = f"figures/backtest_ma{MA_PERIOD}.png"
plt.savefig(filename, dpi=150)
print(f"\n  图已保存至 {filename}")
