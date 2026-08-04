"""
Phase 2.1 — 参数敏感性：对比不同均线周期的策略表现
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
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

INIT = 100_000
results = {}

for ma in [5, 10, 20, 30, 60]:
    df["ma"] = df["收盘"].rolling(ma).mean()
    df["持仓"] = (df["收盘"] > df["ma"]).shift(1).fillna(0)
    df["策略日收益"] = df["持仓"] * df["收盘"].pct_change()
    df["净值"] = (1 + df["策略日收益"]).cumprod() * INIT
    ret = (df["净值"].iloc[-1] / INIT - 1) * 100
    dd = ((df["净值"] - df["净值"].cummax()) / df["净值"].cummax() * 100).min()
    results[ma] = {"净值": df["净值"].copy(), "收益": ret, "回撤": dd}
    print(f"MA{ma:2d}:  收益 {ret:+6.2f}%    最大回撤 {dd:+6.2f}%")

# 买入持有基准
df["买入持有净值"] = (1 + df["收盘"].pct_change()).cumprod() * INIT
bh_ret = (df["买入持有净值"].iloc[-1] / INIT - 1) * 100
print(f"\n买入持有: 收益 {bh_ret:+6.2f}%")

# 画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：各参数收益 vs 回撤散点图
for ma, r in results.items():
    ax1.scatter(r["回撤"], r["收益"], s=100, zorder=5)
    ax1.annotate(f"MA{ma}", (r["回撤"], r["收益"]),
                 textcoords="offset points", xytext=(8, -4), fontsize=10)
ax1.scatter(0, bh_ret, s=80, color="gray", marker="s", zorder=5)
ax1.annotate("买入持有", (0, bh_ret), textcoords="offset points", xytext=(8, -4), fontsize=9, color="gray")
ax1.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax1.set_xlabel("最大回撤 (%)")
ax1.set_ylabel("总收益 (%)")
ax1.set_title("不同均线周期：收益 vs 风险", fontsize=12)
ax1.grid(True, alpha=0.3)

# 右图：资金曲线
ax2.plot(df.index, df["买入持有净值"], linewidth=1.5, color="gray", label="买入持有", alpha=0.7)
colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"]
for (ma, r), c in zip(results.items(), colors):
    ax2.plot(r["净值"].index, r["净值"].values, linewidth=1.0, color=c, label=f"MA{ma}", alpha=0.85)
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_title("资金曲线对比", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_ma_comparison.png", dpi=150)
print("\n图已保存至 figures/phase2_ma_comparison.png")
