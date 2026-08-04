"""
因子2 — 放量反弹

买入: 偏离 < 阈值 AND 最近5天>=3天放量 AND 价格<MA20
卖出: 价格>MA20后 量比>1.5天量+次日缩量 或 连续2天量比>1.0
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SPLIT = "2024-01-01"

def backtest(data, dev_th, vol_entry_th, vol_exit_th):
    """vol_entry_th: 放量标准(量比>?), vol_exit_th: 天量标准(量比>?)"""
    d = data.copy()
    d["收盘"] = pd.to_numeric(d["close"])
    d["成交量"] = pd.to_numeric(d["volume"])
    d["vol_ma"] = d["成交量"].rolling(20).mean()
    d["量比"] = d["成交量"] / d["vol_ma"]
    d["放量天数"] = (d["量比"] > vol_entry_th).rolling(5).sum()
    d["ma20"] = d["收盘"].rolling(20).mean()
    d["偏离"] = d["收盘"] / d["ma20"] - 1

    d["持仓"] = 0
    in_pos = False
    exit_reason = ""

    for i in range(1, len(d)):
        dev = d["偏离"].iloc[i]
        price = d["收盘"].iloc[i]
        vol_r = d["量比"].iloc[i]
        vol_r_prev = d["量比"].iloc[i - 1] if i >= 1 else 1
        above_ma = price > d["ma20"].iloc[i]

        # ===== 买入 =====
        buy_entry = (
            dev < -dev_th
            and d["放量天数"].iloc[i] >= 3
            and price < d["ma20"].iloc[i]
        )

        # ===== 卖出 =====
        if in_pos:
            # 信号卖出（价格回到均线以上）
            sig_sell = dev > dev_th

            # 天量后缩量卖出: 昨天量比>vol_exit_th 且 今天量比<1.0
            vol_spike = above_ma and vol_r_prev > vol_exit_th and vol_r < 1.0

            # 连续超量卖出: 价格>MA 且 连续2天量比>1.0
            double_vol = above_ma and vol_r > 1.0 and vol_r_prev > 1.0

            if sig_sell:
                in_pos = False
            elif vol_spike:
                in_pos = False
            elif double_vol:
                in_pos = False

        elif buy_entry:
            in_pos = True

        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)

    pos = d["持仓"].shift(1).fillna(0)
    d["日收"] = pos * d["收盘"].pct_change()
    d["净值"] = (1 + d["日收"]).cumprod()
    return d

def stats(d):
    net = d["净值"].iloc[-1]
    daily = d["日收"].dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sh = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    mdd = ((d["净值"] - d["净值"].cummax()) / d["净值"].cummax() * 100).min()
    tr = (d["持仓"].diff() == 1).sum()
    avg_hold = d["持仓"].sum() / tr if tr > 0 else 0
    return (net - 1) * 100, sh, mdd, tr, avg_hold

# ===== 参数扫描 =====
dev_ths = [0.01, 0.02, 0.03, 0.05]
vol_entries = [1.0, 1.2, 1.5]
vol_exits = [1.5, 2.0]

print("=" * 72)
print("  因子2 — 参数扫描（海螺训练期 2020-2023）")
print("=" * 72)

df = pd.read_csv("data/600585.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
train = df[df.index < SPLIT]

best_sharpe = -99
best_params = None

print(f'{"偏离阈值":<10} {"放量标准":<10} {"天量标准":<10} {"收益":>8} {"夏普":>6} {"交易":>5} {"均持":>5}')
print("-" * 58)

for dt in dev_ths:
    for ve in vol_entries:
        for vx in vol_exits:
            d = backtest(train, dt, ve, vx)
            ret, sh, dd, tr, ah = stats(d)
            if sh > best_sharpe:
                best_sharpe = sh
                best_params = (dt, ve, vx)
            print(f"  {dt*100:<8.0f}%  量>{ve:<7.1f}   量>{vx:<7.1f}   {ret:>+6.1f}% {sh:>5.2f} {tr:>4.0f}次 {ah:>4.0f}天")

print(f"\n  训练期最优: 偏离{best_params[0]*100:.0f}%  放量>{best_params[1]:.1f}  天量>{best_params[2]:.1f}")

# ===== 测试期验证最优参数 =====
print(f"\n{'='*72}")
print(f"  测试期验证（偏离{best_params[0]*100:.0f}% 放量>{best_params[1]:.1f} 天量>{best_params[2]:.1f}）")
print("=" * 72)

test = df[df.index >= SPLIT]
d_test = backtest(test, *best_params)
ret, sh, dd, tr, ah = stats(d_test)
print(f"  测试期: 收益{ret:+.1f}%  夏普{sh:.2f}  回撤{dd:+.1f}%  {tr:.0f}次  均持{ah:.0f}天")

# 对比原版
d_orig = backtest(test, 0.05, 1.0, 99)  # 99 = 永不触发天量退出
orig_ret, orig_sh, orig_dd, orig_tr, orig_ah = stats(d_orig)
print(f"  原版对比: {orig_ret:+.1f}% S{orig_sh:.2f} → {ret:+.1f}% S{sh:.2f}")

# ===== 画图 =====
d_full = backtest(df, *best_params)
d_orig_full = backtest(df, 0.05, 1.0, 99)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
    subplot_titles=("因子2 vs 原版 vs 买持", "成交量 + 量比"))

fig.add_trace(go.Scatter(x=d_full.index, y=d_full["净值"] * 100000, name=f"因子2",
    line=dict(color="#2ca02c", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=d_orig_full.index, y=d_orig_full["净值"] * 100000, name="原版偏离度",
    line=dict(color="#d62728", width=1)), row=1, col=1)
bh = (1 + pd.to_numeric(df["close"]).pct_change()).cumprod() * 100000
fig.add_trace(go.Scatter(x=bh.index, y=bh, name="买持",
    line=dict(color="#999", width=0.8, dash="dash")), row=1, col=1)
fig.add_hline(y=100000, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["成交量"], name="成交量", opacity=0.3), row=2, col=1)
fig.add_trace(go.Scatter(x=d_full.index, y=d_full["量比"], name="量比",
    line=dict(color="orange", width=1)), row=2, col=1)
fig.add_hline(y=1, line_dash="dot", line_color="gray", row=2, col=1)
fig.add_hline(y=best_params[2], line_dash="dash", line_color="red", row=2, col=1)

fig.update_layout(title=f"因子2 — 放量反弹（偏离{best_params[0]*100:.0f}% 放量>{best_params[1]:.1f} 天量>{best_params[2]:.1f}）",
    hovermode="x unified")
fig.show()
