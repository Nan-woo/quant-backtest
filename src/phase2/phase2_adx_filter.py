"""
Phase 2.2 — ADX 趋势过滤器
ADX（Average Directional Index）：一个数字告诉你"现在有没有趋势"
  ADX > 25 → 趋势明确，金叉信号可信
  ADX < 20 → 震荡，金叉信号忽略
  ADX 在 20~25 → 灰色地带，保持原仓位不动
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

SHORT, LONG, ADX_PERIOD = 5, 20, 14
INIT = 100_000

# ===== 双均线 =====
df["ma_short"] = df["收盘"].rolling(SHORT).mean()
df["ma_long"] = df["收盘"].rolling(LONG).mean()

# ===== ADX 计算（只关心趋势"有多强"，不关心方向）=====
# 第1步：方向运动
df["up"] = df["最高"] - df["最高"].shift(1)       # 今天高点比昨天高多少
df["down"] = df["最低"].shift(1) - df["最低"]      # 今天低点比昨天低多少
df["+DM"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0)
df["-DM"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0)

# 第2步：真实波幅（今天波动的总幅度）
df["tr1"] = df["最高"] - df["最低"]
df["tr2"] = (df["最高"] - df["收盘"].shift(1)).abs()
df["tr3"] = (df["最低"] - df["收盘"].shift(1)).abs()
df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

# 第3步：平滑（用 Wilder 的 EMA，alpha = 1/N）
df["ATR"] = df["TR"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
df["+DI"] = df["+DM"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"] * 100
df["-DI"] = df["-DM"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df["ATR"] * 100

# 第4步：ADX = 方向强度的"净值"
df["DX"] = (df["+DI"] - df["-DI"]).abs() / (df["+DI"] + df["-DI"]) * 100
df["ADX"] = df["DX"].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

# ===== 趋势判断（只看 ADX）=====
df["趋势"] = "震荡"
df.loc[df["ADX"] > 25, "趋势"] = "趋势"
df.loc[df["ADX"] < 20, "趋势"] = "震荡"  # 20~25 之间留灰色地带，不改变状态

# ===== 金叉死叉 =====
df["cross"] = df["ma_short"] - df["ma_long"]
df["cross_y"] = df["cross"].shift(1)
df["信号"] = 0
df.loc[(df["cross_y"] <= 0) & (df["cross"] > 0), "信号"] = 1    # 金叉
df.loc[(df["cross_y"] >= 0) & (df["cross"] < 0), "信号"] = -1   # 死叉

# ---- 策略 A：无脑金叉 ----
df["持仓A"] = df["信号"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# ---- 策略 B：ADX 过滤 ----
# 只在 ADX > 25（趋势明确）时接受金叉信号
# ADX < 20（震荡）时保持空仓
df["信号B"] = df["信号"].copy()
df.loc[df["ADX"] < 20, "信号B"] = 0   # 震荡期忽略信号
df["持仓B"] = df["信号B"].replace(0, np.nan).ffill().fillna(0).clip(lower=0)

# 死叉不忽略——如果趋势结束（ADX跌破20），强制清仓
df.loc[(df["ADX"].shift(1) >= 20) & (df["ADX"] < 20), "持仓B"] = 0

# 回测
for label, pos_col in [("A", "持仓A"), ("B", "持仓B")]:
    pos = df[pos_col].shift(1).fillna(0)
    df[f"{label}_日收"] = pos * df["收盘"].pct_change()
    df[f"{label}_净值"] = (1 + df[f"{label}_日收"]).cumprod() * INIT

df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * INIT

# 算指标
def calc(net):
    ret = (net.iloc[-1] / INIT - 1) * 100
    days = len(net)
    ann = ((net.iloc[-1] / INIT) ** (252 / days) - 1) * 100
    peak = net.cummax()
    dd = ((net - peak) / peak * 100).min()
    return ret, ann, dd

rA, aA, dA = calc(df["A_净值"])
rB, aB, dB = calc(df["B_净值"])
bh_r, _, _ = calc(df["买持净值"])

posA = (df["持仓A"].shift(1).fillna(0) == 1).sum()
posB = (df["持仓B"].shift(1).fillna(0) == 1).sum()
trend_d = (df["ADX"] > 25).sum()
chop_d = (df["ADX"] < 20).sum()
grey_d = ((df["ADX"] >= 20) & (df["ADX"] <= 25)).sum()

print("=" * 60)
print(f"  ADX 趋势过滤器")
print("=" * 60)
print(f"  趋势日 (ADX>25): {trend_d} 天")
print(f"  震荡日 (ADX<20): {chop_d} 天")
print(f"  灰色地带:       {grey_d} 天")
print()
print(f"{'':<22} {'无脑金叉':<15} {'ADX过滤':<15}")
print(f"  总收益        {rA:>+10.2f}%      {rB:>+10.2f}%")
print(f"  年化          {aA:>+10.2f}%      {aB:>+10.2f}%")
print(f"  最大回撤      {dA:>+10.2f}%      {dB:>+10.2f}%")
print(f"  持仓天数      {posA:>10}       {posB:>10}")
print(f"\n  买入持有: {bh_r:+.2f}%")

# 额外统计：ADX 过滤后，金叉信号的胜率变化
# 在趋势期发出的金叉，后续 20 天平均收益
df["买入信号A"] = df["信号"] == 1
df["买入信号B"] = (df["信号"] == 1) & (df["ADX"] > 25)

# 画图
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# 第1行：ADX + 趋势区间
ax1 = axes[0]
ax1.plot(df.index, df["ADX"], linewidth=0.8, color="#9467bd", label="ADX")
ax1.axhline(y=25, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="趋势阈值 25")
ax1.axhline(y=20, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="震荡阈值 20")
ax1.fill_between(df.index, df["ADX"].max(), 25,
                 where=df["ADX"] > 25, alpha=0.08, color="green")
ax1.fill_between(df.index, 20, 0,
                 where=df["ADX"] < 20, alpha=0.08, color="red")
ax1.set_title("ADX — 趋势强度计（不判断方向，只判断力度）", fontsize=12)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.2)

# 第2行：价格 + 金叉标记（区分趋势期金叉vs震荡期金叉）
ax2 = axes[1]
ax2.plot(df.index, df["收盘"], linewidth=0.5, color="#ccc", alpha=0.5)
buy_trend = df[(df["信号"] == 1) & (df["ADX"] > 25)]
buy_chop = df[(df["信号"] == 1) & (df["ADX"] <= 25)]
ax2.scatter(buy_trend.index, buy_trend["收盘"], marker="^", color="green",
            s=50, zorder=5, alpha=0.8, label=f"金叉-趋势期 ({len(buy_trend)}次)")
ax2.scatter(buy_chop.index, buy_chop["收盘"], marker="^", color="red",
            s=30, zorder=5, alpha=0.4, label=f"金叉-震荡期 ({len(buy_chop)}次)")
ax2.set_title("金叉信号：绿色=ADX过滤保留 / 红色=ADX过滤丢弃", fontsize=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.2)

# 第3行：资金曲线
ax3 = axes[2]
ax3.plot(df.index, df["A_净值"], linewidth=1.0, color="gray", alpha=0.5, label=f"无脑金叉 ({rA:+.1f}%)")
ax3.plot(df.index, df["B_净值"], linewidth=1.2, color="#2ca02c", label=f"ADX过滤 ({rB:+.1f}%)")
ax3.plot(df.index, df["买持净值"], linewidth=1.0, color="black", alpha=0.3, label=f"买入持有 ({bh_r:+.1f}%)")
ax3.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax3.set_title("资金曲线对比", fontsize=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase2_adx_filter.png", dpi=150)
print("\n图已保存至 figures/phase2_adx_filter.png")
