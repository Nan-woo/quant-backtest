"""
Phase 4.1 — 样本外检验
问题：MA23 在 2020-2026 上赚了 49%，这是真规律还是撞大运？
方法：把数据劈成两半，用前半段找参数，用后半段验证
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

INIT = 100_000

# ===== 核心：把数据劈成两半 =====
# 前半段：2020 ~ 2023-07（训练期，用来找参数）
# 后半段：2023-07 ~ 2026-07（测试期，只用来看表现，绝不回头调参数）
split_date = "2023-07-01"
train = df[df.index < split_date]
test  = df[df.index >= split_date]

print(f"训练期: {train.index[0].date()} ~ {train.index[-1].date()}  ({len(train)} 天)")
print(f"测试期: {test.index[0].date()} ~ {test.index[-1].date()}  ({len(test)} 天)")
print()

# ===== 第 1 轮：在训练期上找最好的 MA 参数 =====
print("【训练期】测 5~60 日每组 5 个，找最好参数...")
best_ma, best_train_ret = 0, -999
for ma in range(5, 65, 5):
    train_copy = train.copy()
    train_copy["ma"] = train_copy["收盘"].rolling(ma).mean()
    train_copy["持仓"] = (train_copy["收盘"] > train_copy["ma"]).shift(1).fillna(0)
    train_copy["策略日收益"] = train_copy["持仓"] * train_copy["收盘"].pct_change()
    train_copy["净值"] = (1 + train_copy["策略日收益"]).cumprod() * INIT
    ret = (train_copy["净值"].iloc[-1] / INIT - 1) * 100
    if ret > best_train_ret:
        best_ma, best_train_ret = ma, ret

print(f"  训练期最好参数: MA{best_ma} ({best_train_ret:+.2f}%)")

# ===== 第 2 轮：用这 ONE 个参数在测试期回测（绝不回头调） =====
print(f"\n【测试期】只测 MA{best_ma}，不调参数...")
test_copy = test.copy()
test_copy["ma"] = test_copy["收盘"].rolling(best_ma).mean()
test_copy["持仓"] = (test_copy["收盘"] > test_copy["ma"]).shift(1).fillna(0)
test_copy["策略日收益"] = test_copy["持仓"] * test_copy["收盘"].pct_change()
test_copy["净值"] = (1 + test_copy["策略日收益"]).cumprod() * INIT
test_ret = (test_copy["净值"].iloc[-1] / INIT - 1) * 100

# 买入持有基准（测试期）
test_copy["买持净值"] = (1 + test_copy["收盘"].pct_change()).cumprod() * INIT
bh_test_ret = (test_copy["买持净值"].iloc[-1] / INIT - 1) * 100

print(f"  测试期策略收益:  {test_ret:+.2f}%")
print(f"  测试期买入持有:  {bh_test_ret:+.2f}%")
print(f"  超额收益:        {test_ret - bh_test_ret:+.2f}%")

# ===== 对比：如果你在全段数据上直接挖 =====
print(f"\n【全段数据】挖出来的最好参数（之前的结果）...")
best_all_ma, best_all_ret = 0, -999
for ma in range(5, 65, 5):
    df2 = df.copy()
    df2["ma"] = df2["收盘"].rolling(ma).mean()
    df2["持仓"] = (df2["收盘"] > df2["ma"]).shift(1).fillna(0)
    df2["策略日收益"] = df2["持仓"] * df2["收盘"].pct_change()
    df2["净值"] = (1 + df2["策略日收益"]).cumprod() * INIT
    ret = (df2["净值"].iloc[-1] / INIT - 1) * 100
    if ret > best_all_ret:
        best_all_ma, best_all_ret = ma, ret

print(f"  全段最好: MA{best_all_ma} ({best_all_ret:+.2f}%)  ← 这个数字有水分")
print(f"  训练最优 + 测试验证: MA{best_ma} 训练期{best_train_ret:+.2f}% / 测试期{test_ret:+.2f}%  ← 可信得多")

# ===== 输出结论 =====
print()
print("=" * 50)
print("结论")
print("=" * 50)
print(f"  全段挖出的 MA{best_all_ma} 收益 {best_all_ret:+.2f}% 含运气成分")
print(f"  样本外验证的 MA{best_ma} 测试期实际跑出 {test_ret:+.2f}%")
if test_ret > bh_test_ret:
    print(f"  策略确实跑赢了买入持有，超额 {test_ret - bh_test_ret:+.2f}%")
else:
    print(f"  策略在样本外输给了买入持有 —— 之前的'好'是过拟合")

# ===== 画图 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：训练期资金曲线
train_plot = train.copy()
train_plot["ma"] = train_plot["收盘"].rolling(best_ma).mean()
train_plot["持仓"] = (train_plot["收盘"] > train_plot["ma"]).shift(1).fillna(0)
train_plot["策略日收益"] = train_plot["持仓"] * train_plot["收盘"].pct_change()
train_plot["净值"] = (1 + train_plot["策略日收益"]).cumprod() * INIT
train_plot["买持净值"] = (1 + train_plot["收盘"].pct_change()).cumprod() * INIT

ax1.plot(train_plot.index, train_plot["净值"], color="#1f77b4", label=f"策略 MA{best_ma}")
ax1.plot(train_plot.index, train_plot["买持净值"], color="gray", alpha=0.7, label="买入持有")
ax1.axvline(x=pd.to_datetime(split_date), color="red", linewidth=0.8, linestyle="--", alpha=0.5)
ax1.set_title(f"训练期（找参数）→ MA{best_ma} 最优", fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

# 右图：测试期
ax2.plot(test_copy.index, test_copy["净值"], color="#1f77b4", label=f"策略 MA{best_ma}")
ax2.plot(test_copy.index, test_copy["买持净值"], color="gray", alpha=0.7, label="买入持有")
ax2.set_title("测试期（只看不动手）", fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase4_out_of_sample.png", dpi=150)
print("\n图已保存至 figures/phase4_out_of_sample.png")
