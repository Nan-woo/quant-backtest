"""
我的第一个原创因子 —— "恐慌反弹"

逻辑：放量暴跌 = 情绪宣泄 → 超卖 → 次日反弹概率高
条件：
  1. 今天跌幅 < -3%（恐慌）
  2. 今天成交量 > 20日均量的 1.5 倍（放量 = 有人在割肉）
  3. 不是连续暴跌（连着暴跌说明是真趋势，不是恐慌）
触发后：次日开盘买入，持有 N 天卖出
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 数据加载
# ============================================================
df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

INIT = 100_000

# ============================================================
# 因子参数 —— 你想改就改这几个数
# ============================================================
PANIC_DROP = -0.03       # 跌幅超过 3% 才算恐慌
VOL_SPIKE = 1.5          # 成交量超过中位数几倍算"放量"
HOLD_DAYS = 5            # 买入后持有几天
SKIP_CONSECUTIVE = True  # 是否跳过连续暴跌

# ============================================================
# 计算基础数据
# ============================================================
df["日收益"] = df["收盘"].pct_change()
df["vol_med"] = df["成交量"].rolling(20).median()
df["量比"] = df["成交量"] / df["vol_med"]

# ============================================================
# 因子信号：识别恐慌日
# ============================================================
df["恐慌信号"] = 0

for i in range(5, len(df)):
    # 条件 1：今天暴跌
    is_panic = df["日收益"].iloc[i] < PANIC_DROP
    # 条件 2：放量
    is_high_vol = df["量比"].iloc[i] > VOL_SPIKE
    # 条件 3：昨天不是暴跌（跳过连续暴跌）
    if SKIP_CONSECUTIVE:
        yesterday_panic = df["日收益"].iloc[i - 1] < PANIC_DROP
    else:
        yesterday_panic = False

    if is_panic and is_high_vol and not yesterday_panic:
        df.iloc[i, df.columns.get_loc("恐慌信号")] = 1

# ============================================================
# 模拟交易：信号触发 → 次日买入 → 持有 HOLD_DAYS 天卖出
# ============================================================
df["持仓"] = 0
hold_counter = 0

for i in range(len(df)):
    # 昨天出信号 → 今天买
    if i > 0 and df["恐慌信号"].iloc[i - 1] == 1:
        hold_counter = HOLD_DAYS

    if hold_counter > 0:
        df.iloc[i, df.columns.get_loc("持仓")] = 1
        hold_counter -= 1

# 回测
pos = df["持仓"].shift(1).fillna(0)
df["策略日收"] = pos * df["日收益"]
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
sig_count = int(df["恐慌信号"].sum())
hold_days = int(df["持仓"].shift(1).fillna(0).sum())
trade_count = int(df["恐慌信号"].sum())  # 每信号 = 一次交易

print("=" * 60)
print("  我的第一个原创因子：恐慌反弹")
print("=" * 60)
print(f"  规则：跌超{PANIC_DROP*100:.0f}% + 放量{VOL_SPIKE}倍 → 次日买 → 持{HOLD_DAYS}天卖")
print(f"  信号触发: {sig_count} 次（6 年）")
print(f"  持仓天数: {hold_days} 天（{hold_days/len(df)*100:.1f}% 时间）")
print(f"  策略收益: {s_ret:+.2f}%")
print(f"  最大回撤: {s_dd:+.2f}%")
print(f"  买入持有: {b_ret:+.2f}%")
print(f"  超额收益: {s_ret - b_ret:+.2f}%")

# ============================================================
# 看每次交易的收益
# ============================================================
print(f"\n  每次交易详情：")
trade_dates = df[df["恐慌信号"] == 1].index
for td in trade_dates:
    t_idx = df.index.get_loc(td)
    # 买入价 = 次日开盘
    if t_idx + 1 < len(df):
        buy_price = df["开盘"].iloc[t_idx + 1]
        # 卖出价 = 持有期结束后收盘
        sell_idx = min(t_idx + 1 + HOLD_DAYS, len(df) - 1)
        sell_price = df["收盘"].iloc[sell_idx]
        trade_ret = (sell_price / buy_price - 1) * 100
        win = "✓" if trade_ret > 0 else "✗"
        print(f"    {td.date()} 恐慌日跌{df['日收益'].iloc[t_idx]*100:+.2f}%  "
              f"→ 次日买{buy_price:.2f} → {HOLD_DAYS}天后卖{sell_price:.2f}  "
              f"{trade_ret:+.2f}% {win}")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
colors = ["#2ca02c" if df["日收益"].iloc[i] >= 0 else "#d62728"
          for i in range(len(df))]
ax1.bar(df.index, df["日收益"].values * 100, color=colors, alpha=0.4, width=1)
for td in trade_dates:
    ax1.axvline(x=td, color="blue", linewidth=0.8, linestyle="--", alpha=0.7)
ax1.axhline(y=PANIC_DROP * 100, color="red", linewidth=0.5, linestyle="--",
            label=f"恐慌线 ({PANIC_DROP*100:.0f}%)")
ax1.set_ylabel("日收益率 (%)")
ax1.set_title(f"恐慌反弹因子：跌超{PANIC_DROP*100:.0f}%+放量 → 蓝线=买入信号", fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2)

ax2 = axes[1]
ax2.plot(df.index, df["策略净值"], linewidth=1.2, color="#2ca02c",
         label=f"恐慌反弹 ({s_ret:+.1f}%)")
ax2.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3,
         label=f"买入持有 ({b_ret:+.1f}%)")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
for td in trade_dates:
    ax2.axvline(x=td, color="blue", linewidth=0.5, linestyle="--", alpha=0.4)
ax2.set_title("资金曲线", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_my_factor.png", dpi=150)
print("\n图已保存至 figures/phase2_my_factor.png")
