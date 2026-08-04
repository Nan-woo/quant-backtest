"""
双向均值回归策略 — 双重检验
1. 样本外：训练期找最优阈值 → 测试期验证
2. 随机基准：1000次随机交易，看策略排第几
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
df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

def run_strategy(data, threshold):
    """双向均值回归：偏离 < -threshold 买，偏离 > +threshold 卖"""
    data = data.copy()
    data["持仓"] = 0
    in_pos = False
    for i in range(1, len(data)):
        dev = data["偏离"].iloc[i]
        if not in_pos and dev <= -threshold:
            in_pos = True
        elif in_pos and dev >= threshold:
            in_pos = False
        data.iloc[i, data.columns.get_loc("持仓")] = int(in_pos)
    pos = data["持仓"].shift(1).fillna(0)
    data["策略日收"] = pos * data["收盘"].pct_change()
    data["净值"] = (1 + data["策略日收"]).cumprod() * INIT
    ret = (data["净值"].iloc[-1] / INIT - 1) * 100
    peak = data["净值"].cummax()
    dd = ((data["净值"] - peak) / peak * 100).min()
    return ret, dd, data["净值"]

# ============================================================
# 检验 1：样本外
# ============================================================
split_date = "2023-07-01"
train = df[df.index < split_date].copy()
test  = df[df.index >= split_date].copy()

print("=" * 60)
print("  检验 1：样本外测试")
print("=" * 60)
print(f"  训练期: {train.index[0].date()} ~ {train.index[-1].date()}  ({len(train)}天)")
print(f"  测试期: {test.index[0].date()} ~ {test.index[-1].date()}  ({len(test)}天)")
print()

# 训练期：测不同阈值，找最优
print("  训练期 — 测不同阈值...")
best_thresh, best_train_ret = 0, -999
for t in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    ret, dd, _ = run_strategy(train, t)
    if ret > best_train_ret:
        best_thresh, best_train_ret = t, ret

print(f"  训练期最优阈值: {best_thresh*100:.0f}%  (收益 {best_train_ret:+.2f}%)")

# 测试期：只测最优阈值
test_ret, test_dd, test_nv = run_strategy(test, best_thresh)
test_bh = (test["收盘"].pct_change().fillna(0) + 1).cumprod() * INIT
test_bh_ret = (test_bh.iloc[-1] / INIT - 1) * 100

print(f"  测试期实际表现:")
print(f"    策略: {test_ret:+.2f}%  回撤: {test_dd:+.2f}%")
print(f"    买入持有: {test_bh_ret:+.2f}%")
print(f"    超额: {test_ret - test_bh_ret:+.2f}%")

# 全段验证
full_ret, full_dd, _ = run_strategy(df, best_thresh)
bh_ret = (df["收盘"].pct_change().fillna(0) + 1).cumprod() * INIT
bh_ret_v = (bh_ret.iloc[-1] / INIT - 1) * 100
print(f"\n  全段（训练期最优阈值 {best_thresh*100:.0f}%）: {full_ret:+.2f}%")
print(f"  买入持有全段: {bh_ret_v:+.2f}%")

# ============================================================
# 检验 2：随机基准（用全段回测的阈值）
# ============================================================
print()
print("=" * 60)
print("  检验 2：随机基准（p 值）")
print("=" * 60)

full_data = df.copy()
# 重新跑策略，获取持仓列
_, _, _ = run_strategy(df, best_thresh)  # 先跑一遍保证数据有持仓列
strat_ret = full_ret  # already calculated

hold_pct = 0.40   # 简单取个持仓比例（训练期回测约40%时间持仓）
np.random.seed(42)
random_rets = []
for _ in range(1000):
    rand_pos = (np.random.rand(len(full_data)) < hold_pct).astype(int)
    rand_daily = rand_pos * full_data["收盘"].pct_change()
    rand_nv = (1 + rand_daily).cumprod() * INIT
    random_rets.append((rand_nv.iloc[-1] / INIT - 1) * 100)

random_rets = np.array(random_rets)
better_than = (random_rets < strat_ret).sum()
pct_rank = better_than / 1000 * 100

print(f"  策略收益: {strat_ret:+.2f}%  (持仓 {hold_pct*100:.0f}% 时间)")
print(f"  随机平均: {random_rets.mean():+.2f}%")
print(f"  随机中位数: {np.median(random_rets):+.2f}%")
print(f"  随机 95 分位: {np.percentile(random_rets, 95):+.2f}%")
print(f"  随机 99 分位: {np.percentile(random_rets, 99):+.2f}%")
print(f"  策略跑赢了 {better_than}/1000 ({pct_rank:.0f}%) 个随机策略")
print()
if pct_rank >= 99:
    print("  → p < 0.01：极大概率不是运气（但可能是过拟合）")
elif pct_rank >= 95:
    print("  → p < 0.05：统计上显著")
elif pct_rank >= 90:
    print("  → 边缘显著，需要更多证据")
else:
    print(f"  → p = {1 - pct_rank/100:.2f}，不显著。收益可能来自随机蹭趋势。")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：样本外
ax1 = axes[0]
_, _, train_nv = run_strategy(train, best_thresh)
ax1.plot(train.index, train_nv, color="#1f77b4", label=f"训练期 ({best_thresh*100:.0f}%)")
ax1.plot(test.index, test_nv, color="#2ca02c", label=f"测试期 ({test_ret:+.1f}%)")
ax1.axvline(x=pd.to_datetime(split_date), color="red", linewidth=0.8, linestyle="--")
bh_train = (train["收盘"].pct_change().fillna(0) + 1).cumprod() * INIT
ax1.plot(train.index, bh_train, color="gray", alpha=0.5)
ax1.plot(test.index, test_bh, color="gray", alpha=0.5, label="买入持有")
ax1.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax1.set_title(f"样本外检验：训练→阈值{best_thresh*100:.0f}%→测试", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

# 右：随机分布
ax2 = axes[1]
ax2.hist(random_rets, bins=50, color="#ccc", edgecolor="#999", alpha=0.7)
ax2.axvline(x=strat_ret, color="#2ca02c", linewidth=2, label=f"策略 {strat_ret:+.1f}%")
ax2.axvline(x=np.median(random_rets), color="red", linewidth=1, linestyle="--", label=f"随机中位数 {np.median(random_rets):+.1f}%")
ax2.axvline(x=np.percentile(random_rets, 95), color="orange", linewidth=1, linestyle="--", label=f"95分位")
ax2.set_title(f"随机基准：跑赢 {pct_rank:.0f}% 的随机策略", fontsize=12)
ax2.set_xlabel("总收益 (%)")
ax2.set_ylabel("出现次数")
ax2.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("figures/phase2_mr_validation.png", dpi=150)
print("\n图已保存至 figures/phase2_mr_validation.png")
