"""
海螺水泥 600585 样本外检验
问题：策略在海螺上 p=0.023，但这是我们偷看全段数据后说的。
真正的检验：劈数据，用前半找参数，用后半验证一次。
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("data/600585.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df["收盘"] = pd.to_numeric(df["close"])

# ============================================================
# 劈数据：2020-2023 训练 / 2024-2026 测试
# ============================================================
split_date = "2024-01-01"
train = df[df.index < split_date].copy()
test = df[df.index >= split_date].copy()

print(f"训练期: {train.index[0].date()} ~ {train.index[-1].date()} ({len(train)}天)")
print(f"测试期: {test.index[0].date()} ~ {test.index[-1].date()} ({len(test)}天)")
print()

# ============================================================
# 训练期：网格搜索 MA + 阈值
# ============================================================
def run_strategy(data, ma, threshold):
    d = data.copy()
    d["ma"] = d["收盘"].rolling(ma).mean()
    d["偏离"] = d["收盘"] / d["ma"] - 1
    d["持仓"] = 0; in_pos = False
    for i in range(1, len(d)):
        dev = d["偏离"].iloc[i]
        if not in_pos and dev < -threshold:
            in_pos = True
        elif in_pos and dev > threshold:
            in_pos = False
        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)
    pos = d["持仓"].shift(1).fillna(0)
    d["日收"] = pos * d["收盘"].pct_change()
    d["净值"] = (1 + d["日收"]).cumprod()
    # 买持
    bh = (1 + d["收盘"].pct_change()).cumprod()
    ret = d["净值"].iloc[-1]
    bh_ret = bh.iloc[-1]
    alpha = (ret / bh_ret - 1)  # 超额
    return ret, alpha, d["持仓"].sum() / (d["持仓"].diff()==1).sum() if (d["持仓"].diff()==1).sum() > 0 else 0

mas = [10, 20, 30, 40]
thresholds = [0.03, 0.05, 0.07, 0.10]
print("训练期网格搜索（MA × 阈值）:")
print(f"{'MA':<6} {'阈值':<8} {'策略收益':<10} {'超额(alpha)':<12} {'交易次数':<8}")
print("-" * 55)
best = None; best_alpha = -999

for ma in mas:
    for th in thresholds:
        ret, alpha, avg_hold = run_strategy(train, ma, th)
        label = ""
        if alpha > best_alpha:
            best_alpha = alpha; best = (ma, th)
            label = " ←"
        print(f"  {ma:<4}  {th*100:<5.0f}%    {ret*100:>+6.1f}%      {alpha*100:>+7.1f}%        {label}")

print(f"\n训练期最优: MA={best[0]}, 阈值={best[1]*100:.0f}%")

# ============================================================
# 测试期：只跑最优参数，一次
# ============================================================
print(f"\n{'='*55}")
print(f"测试期验证（MA={best[0]}, 阈值={best[1]*100:.0f}%）")
print(f"{'='*55}")

test_ret, test_alpha, _ = run_strategy(test, best[0], best[1])
test_bh_ret = (1 + test["收盘"].pct_change()).cumprod().iloc[-1]

# 完整跑一遍画图
ma, th = best
for data, name in [(train, "训练期"), (test, "测试期")]:
    d = data.copy()
    d["ma"] = d["收盘"].rolling(ma).mean()
    d["偏离"] = d["收盘"] / d["ma"] - 1
    d["持仓"] = 0; in_pos = False
    for i in range(1, len(d)):
        dev = d["偏离"].iloc[i]
        if not in_pos and dev < -th:
            in_pos = True
        elif in_pos and dev > th:
            in_pos = False
        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)
    pos = d["持仓"].shift(1).fillna(0)
    d["日收"] = pos * d["收盘"].pct_change()
    d["策略净值"] = (1 + d["日收"]).cumprod() * 100_000
    d["买持净值"] = (1 + d["收盘"].pct_change()).cumprod() * 100_000
    if name == "训练期":
        train_res = d
    else:
        test_res = d

print(f"\n  全段6年（偷看答案）: 策略 +44.7%  买持 -54.0%  p=0.023")
print(f"  训练期 MA{ma} 阈值{th*100:.0f}%: 策略 {test_ret*100:+.1f}% (净值因子)")
print(f"  测试期 实际alpha: {(test_ret/test_bh_ret-1)*100:+.1f}%")
print(f"  测试期 策略收益率: {test_ret*100-100:+.1f}%")
print(f"  测试期 买入持有:    {test_bh_ret*100-100:+.1f}%")

# ============================================================
# 画图：训练期 + 测试期
# ============================================================
fig = make_subplots(rows=1, cols=2,
    subplot_titles=(f"训练期 (2020-2023) MA{ma} 阈值{th*100:.0f}%", "测试期 (2024-2026) 只看不动手"),
    shared_yaxes=True)

for col, (res, label) in enumerate([(train_res, "训练期"), (test_res, "测试期")], 1):
    fig.add_trace(go.Scatter(x=res.index, y=res["策略净值"], name=f"{label}策略",
        line=dict(color="#2ca02c", width=2), showlegend=(col==1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=res.index, y=res["买持净值"], name=f"{label}买持",
        line=dict(color="#999", width=1, dash="dash"), showlegend=(col==1)), row=1, col=col)
    fig.add_hline(y=100_000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=col)

fig.update_layout(title=f"海螺水泥 样本外检验 — 训练期选参，测试期验证一次", hovermode="x unified")
fig.show()
