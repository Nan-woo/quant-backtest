"""
Phase 3.2 — 仓位管理直觉

游戏：你有一个硬币，正面概率 60%，反面 40%。
赢：下注额翻倍（赚 100%）。输：下注额归零（亏 100%）。
你可以玩 100 次。每次你决定下注 %（总资金的多少）。
初始资金 10000，你每把押多少？

试试看：押 10%、押 25%、押 50%、押 100%——哪个最后最有钱？
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
INIT = 10_000
GAMES = 100
WIN_RATE = 0.60

# 生成 100 次游戏结果
results = np.random.rand(GAMES) < WIN_RATE  # True = 赢

print("=" * 65)
print("  仓位管理直觉 — 硬币游戏")
print("  " + "-" * 45)
print(f"  赢面 60%, 赢了翻倍, 输了全赔, 玩 {GAMES} 次")
print("=" * 65)
print()
print(f"{'押注比例':<12} {'最终资金':>10} {'最终收益率':>10} {'破产?':>8}")
print("-" * 45)

# 不同的押注比例
bets = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
histories = {}

for bet_pct in bets:
    capital = INIT
    history = [capital]

    for won in results:
        bet_amount = capital * bet_pct
        if won:
            capital += bet_amount    # 翻倍
        else:
            capital -= bet_amount    # 归零
        history.append(capital)
        if capital <= 0:
            break

    histories[bet_pct] = history
    ret = (capital / INIT - 1) * 100
    bust = "☠ 破产" if capital <= 1 else ""
    print(f"  {bet_pct*100:>4.0f}%        {capital:>10.0f}     {ret:>+9.1f}%     {bust}")

# 理论最优：Kelly 公式 f* = p - q = 0.6 - 0.4 = 0.2 (20%)
print()
print(f"  Kelly 公式: f* = p - q = {WIN_RATE} - {1-WIN_RATE} = {WIN_RATE - (1-WIN_RATE):.0%}")
print(f"  理论最优押注: 20%")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：几条代表性路径
ax1 = axes[0]
for bet, hist in histories.items():
    if bet in [0.05, 0.20, 0.50, 1.0]:
        label = f"{bet*100:.0f}%"
        alpha = 0.8 if bet == 0.20 else 0.5
        lw = 2.0 if bet == 0.20 else 1.0
        ax1.plot(range(len(hist)), hist, linewidth=lw, alpha=alpha, label=label)
ax1.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax1.set_yscale("log")
ax1.set_xlabel("游戏次数")
ax1.set_ylabel("资金（对数坐标）")
ax1.set_title("不同押注比例的财富增长", fontsize=12)
ax1.legend(title="每次押注", fontsize=8)
ax1.grid(True, alpha=0.2)

# 右：最终资金 vs 押注比例
ax2 = axes[1]
final_capitals = [histories[b][-1] for b in bets]
ax2.plot([b*100 for b in bets], final_capitals, marker="o", linewidth=1.5, color="#1f77b4")
ax2.axvline(x=20, color="red", linewidth=0.8, linestyle="--", label="Kelly 最优 20%")
ax2.axhline(y=INIT, color="black", linewidth=0.5, linestyle="--")
ax2.set_xlabel("每次押注比例 (%)")
ax2.set_ylabel("最终资金")
ax2.set_title("押注越多 ≠ 赚得越多", fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/phase3_kelly_demo.png", dpi=150)
print("\n图已保存至 figures/phase3_kelly_demo.png")

print()
print("  你看到了什么？")
print("  1. 押 5% → 赚太少，资金增长慢")
print("  2. 押 20% → Kelly 最优，长期资金最多")
print("  3. 押 50% → 波动太大，某次连续输就可能伤筋动骨")
print("  4. 押 100% → 一把输了就没了")
print()
print("  仓位管理的本质：不是猜涨跌，是控制'赌多少才能一直赌下去'")
