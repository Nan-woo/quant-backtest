"""
Phase 5.1 — ADX 市场状态过滤器

问题：海螺后半段持仓陷了 100+ 天，能不能用 ADX 识别趋势期、提前避开？
方法：只在震荡期（ADX < 20）交易，趋势期空仓等待
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# 数据准备
# ============================================================
df = pd.read_csv("data/600585.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df["收盘"] = pd.to_numeric(df["close"])
df["最高"] = pd.to_numeric(df["high"])
df["最低"] = pd.to_numeric(df["low"])

# ============================================================
# 计算 ADX（14天标准周期）
# ============================================================
n = 14
# True Range
df["tr"] = np.maximum(
    df["最高"] - df["最低"],
    np.maximum(
        abs(df["最高"] - df["收盘"].shift(1)),
        abs(df["最低"] - df["收盘"].shift(1))
    )
)
df["atr"] = df["tr"].rolling(n).mean()

# +DM / -DM
df["up"] = df["最高"] - df["最高"].shift(1)
df["dn"] = df["最低"].shift(1) - df["最低"]
df["+dm"] = np.where((df["up"] > df["dn"]) & (df["up"] > 0), df["up"], 0)
df["-dm"] = np.where((df["dn"] > df["up"]) & (df["dn"] > 0), df["dn"], 0)

# 平滑 +DI / -DI
df["+di"] = (df["+dm"].rolling(n).mean() / df["atr"]) * 100
df["-di"] = (df["-dm"].rolling(n).mean() / df["atr"]) * 100

# ADX
df["dx"] = abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"]) * 100
df["adx"] = df["dx"].rolling(n).mean()

# ============================================================
# 策略参数
# ============================================================
MA, THRESHOLD, ADX_THRESHOLD = 20, 0.05, 20
df["ma"] = df["收盘"].rolling(MA).mean()
df["偏离"] = df["收盘"] / df["ma"] - 1

# ============================================================
# 策略 1：原版均值回归（无 ADX 过滤）
# ============================================================
df["持仓A"] = 0; in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    if not in_pos and dev < -THRESHOLD:    # ← 只靠偏离度买入
        in_pos = True
    elif in_pos and dev > THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("持仓A")] = int(in_pos)

# ============================================================
# 策略 2：ADX 过滤版（只在震荡期交易）
# ============================================================
df["持仓B"] = 0; in_pos = False
for i in range(1, len(df)):
    dev = df["偏离"].iloc[i]
    adx_val = df["adx"].iloc[i]
    # ====================================================
    # 【你填】ADX 条件：震荡期才能交易
    # ====================================================
    is_ranging = adx_val < ADX_THRESHOLD   # ADX < 20 = 震荡
    # ====================================================

    if not in_pos and dev < -THRESHOLD and is_ranging:
        in_pos = True
    elif in_pos and dev > THRESHOLD:
        in_pos = False
    df.iloc[i, df.columns.get_loc("持仓B")] = int(in_pos)

# ============================================================
# 算收益
# ============================================================
for label, col in [("A_原版", "持仓A"), ("B_ADX过滤", "持仓B")]:
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
    calmar = ann_ret / abs(mdd/100) if mdd != 0 else 0
    total_ret = (net.iloc[-1] / 100_000 - 1) * 100
    trades = (df[pos_col].diff() == 1).sum()
    avg_hold = df[pos_col].sum() / trades if trades > 0 else 0
    return total_ret, sharpe, mdd, calmar, trades, avg_hold

print("=" * 72)
print("  Phase 5.1 — ADX 市场状态过滤器")
print("=" * 72)
print(f"  规则: 只在 ADX < {ADX_THRESHOLD}（震荡期）时允许买入")
print()

for label, net_col, pos_col in [
    ("原版均值回归", "A_原版_净值", "持仓A"),
    ("ADX 过滤版", "B_ADX过滤_净值", "持仓B"),
]:
    ret, sh, dd, cm, tr, ah = stats(net_col, pos_col)
    print(f"  {label}:")
    print(f"    总收益{ret:+.1f}%  夏普{sh:.2f}  回撤{dd:+.1f}%  卡玛{cm:.2f}  {tr:.0f}次交易  均持{ah:.0f}天")

# ============================================================
# ADX 分年统计
# ============================================================
print()
print("  ADX 年平均值:")
for y in range(2020, 2027):
    try:
        yr = df.loc[str(y)]
        print(f"    {y}: ADX均值 {yr['adx'].mean():.0f}  震荡天数 {(yr['adx']<20).sum()}/{len(yr)}")
    except KeyError:
        continue

# ============================================================
# 画图
# ============================================================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.4, 0.3, 0.3],
    subplot_titles=("策略对比", "ADX 趋势强度", "持仓对比"))

# 上行：净值
fig.add_trace(go.Scatter(x=df.index, y=df["A_原版_净值"], name="原版",
    line=dict(color="#d62728", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["B_ADX过滤_净值"], name="ADX过滤",
    line=dict(color="#2ca02c", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["买持净值"], name="买持",
    line=dict(color="#999", width=0.8, dash="dash")), row=1, col=1)
fig.add_hline(y=100_000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)

# 中行：ADX
fig.add_trace(go.Scatter(x=df.index, y=df["adx"], name="ADX",
    line=dict(color="#1f77b4", width=1)), row=2, col=1)
fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5,
    annotation_text="震荡/趋势分界", row=2, col=1)
fig.add_hline(y=25, line_dash="dot", line_color="red", opacity=0.3, row=2, col=1)

# 下行：持仓对比（用阴影区域）
df["持仓A_shift"] = df["持仓A"].replace(0, np.nan) * 45  # 缩放方便看
df["持仓B_shift"] = df["持仓B"].replace(0, np.nan) * 50
fig.add_trace(go.Scatter(x=df.index, y=df["持仓A_shift"], name="原版持仓",
    line=dict(color="#d62728", width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["持仓B_shift"], name="ADX过滤持仓",
    line=dict(color="#2ca02c", width=2)), row=3, col=1)

fig.update_layout(title="海螺水泥 — ADX 市场状态过滤器", hovermode="x unified", showlegend=True)
fig.update_yaxes(title_text="净值", row=1, col=1)
fig.update_yaxes(title_text="ADX", row=2, col=1)
fig.update_yaxes(title_text="持仓", row=3, col=1)
fig.show()
