"""
Phase 5.1 — RSI 超卖确认过滤器

逻辑：偏离度跌到位 + RSI 确认超卖 = 双确认买入
      不是"震荡期才能交易"，而是"跌够了且砸不动了"
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
# RSI 计算（14天标准周期）
# ============================================================
N = 14
df["涨跌"] = df["收盘"].diff()                       # 今天 - 昨天

# 涨的部分（涨了记正数，跌了记0）
df["涨"] = df["涨跌"].clip(lower=0)                  # clip(lower=0) = 负数变成0
# 跌的部分（跌了记正数绝对值，涨了记0）
df["跌"] = (-df["涨跌"]).clip(lower=0)               # 取负数再 clip = 只保留正跌幅

# 平均涨幅 / 平均跌幅
df["avg_gain"] = df["涨"].rolling(N).mean()
df["avg_loss"] = df["跌"].rolling(N).mean()

# RS = 平均涨幅 ÷ 平均跌幅
df["rs"] = df["avg_gain"] / df["avg_loss"]

# RSI = 100 - 100/(1+RS)
df["rsi"] = 100 - (100 / (1 + df["rs"]))

# ============================================================
# 策略参数
# ============================================================
MA, THRESHOLD, RSI_THRESHOLD = 20, 0.05, 30
df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

# ============================================================
# 策略 1：原版（无 RSI）
# ============================================================
df["持仓A"] = 0; in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev < -THRESHOLD:
        in_pos = True
    elif in_pos and dev > THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("持仓A")] = int(in_pos)

# ============================================================
# 策略 2：RSI 确认版
# ============================================================
df["持仓B"] = 0; in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    rsi_val = df["rsi"].iloc[i]

    # ====================================================
    # 【你填】买入条件：偏离度 + RSI 双确认
    # 提示：偏离度跌破 -5% AND RSI < 30（超卖）
    # ====================================================
    buy_signal = dev < -THRESHOLD and rsi_val < RSI_THRESHOLD
    # ====================================================

    if not in_pos and buy_signal:
        in_pos = True
    elif in_pos and dev > THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("持仓B")] = int(in_pos)

# ============================================================
# 算收益
# ============================================================
for label, col in [("A_原版", "持仓A"), ("B_RSI过滤", "持仓B")]:
    pos = df[col].shift(1).fillna(0)
    df[f"{label}_日收"] = pos * df["收盘"].pct_change()
    df[f"{label}_净值"] = (1 + df[f"{label}_日收"]).cumprod() * 100_000

df["买持净值"] = (1 + df["收盘"].pct_change()).cumprod() * 100_000

# ============================================================
# 评估
# ============================================================
def stats(net_col, pos_col):
    net = df[net_col]
    daily = df[net_col].pct_change().dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    peak = net.cummax()
    mdd = ((net - peak) / peak * 100).min()
    total_ret = (net.iloc[-1] / 100_000 - 1) * 100
    trades = (df[pos_col].diff() == 1).sum()
    return total_ret, sharpe, mdd, trades

print("=" * 72)
print("  Phase 5.1 — RSI 超卖确认过滤器")
print("=" * 72)
print(f"  规则: 偏离度 < -{THRESHOLD*100:.0f}%  AND  RSI < {RSI_THRESHOLD}（超卖）  →  买入")
print()

for label, net_col, pos_col in [
    ("原版均值回归", "A_原版_净值", "持仓A"),
    ("RSI 确认版", "B_RSI过滤_净值", "持仓B"),
]:
    ret, sh, dd, tr = stats(net_col, pos_col)
    print(f"  {label}:  总收益{ret:+.1f}%  夏普{sh:.2f}  回撤{dd:+.1f}%  {tr:.0f}次交易")

# ============================================================
# 画图
# ============================================================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.4, 0.3, 0.3],
    subplot_titles=("策略对比", "RSI 超卖/超买", "持仓对比"))

fig.add_trace(go.Scatter(x=df.index, y=df["A_原版_净值"], name="原版",
    line=dict(color="#d62728", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["B_RSI过滤_净值"], name="RSI确认",
    line=dict(color="#2ca02c", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["买持净值"], name="买持",
    line=dict(color="#999", width=0.8, dash="dash")), row=1, col=1)
fig.add_hline(y=100_000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
    line=dict(color="#ff7f0e", width=1)), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5,
    annotation_text="超卖线 RSI=30", row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5,
    annotation_text="超买线 RSI=70", row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["持仓A"].replace(0, np.nan)*45, name="原版持仓",
    line=dict(color="#d62728", width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["持仓B"].replace(0, np.nan)*50, name="RSI持仓",
    line=dict(color="#2ca02c", width=2)), row=3, col=1)

fig.update_layout(title="海螺水泥 — RSI 超卖确认过滤器", hovermode="x unified", showlegend=True)
fig.update_yaxes(title_text="净值", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_yaxes(title_text="持仓", row=3, col=1)
fig.show()
