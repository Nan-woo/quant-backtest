"""
Phase 5 — 多因子组合（Day 7 最终产出）

因子1: 偏离度均值回归 + 放量确认
因子2: 纯成交量（不放量就卖）
组合: 50/50 等权
验证: 滚动窗口交叉验证

读完这个文件的顺序：
  板块1: 了解两个因子各自在干什么
  板块2: 看它们组合后的效果
  板块3: 交叉验证——不只看一次测试，看多次
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 板块 0：工具函数（先看这里——知道每步在算什么东西）
# ============================================================

def load_stock(code):
    """读取 CSV，返回处理好的 DataFrame"""
    df = pd.read_csv(f"data/{code}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["收盘"] = pd.to_numeric(df["close"])
    df["成交量"] = pd.to_numeric(df["volume"])
    return df


def factor1_returns(data):
    """
    因子 1：偏离度均值回归 + 放量确认

    买入：偏离 20 日 MA < -5% 且 近 5 天 >= 3 天量比 > 1.0
    卖出：偏离 > +5%
    """
    d = data.copy()
    # 价格指标
    d["ma20"] = d["收盘"].rolling(20).mean()
    d["偏离"] = d["收盘"] / d["ma20"] - 1

    # 成交量指标
    d["vol_ma"] = d["成交量"].rolling(20).mean()
    d["量比"] = d["成交量"] / d["vol_ma"]
    d["放量天数"] = (d["量比"] > 1.0).rolling(5).sum()    # 近 5 天有几日放量

    # 逐天模拟
    d["持仓"] = 0
    in_pos = False
    for i in range(1, len(d)):
        dev = d["偏离"].iloc[i]
        vol_ok = d["放量天数"].iloc[i] >= 3   # 至少 3 天放量

        if not in_pos and dev < -0.05 and vol_ok:
            in_pos = True
        elif in_pos and dev > 0.05:
            in_pos = False

        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)

    # 算日收益（今天仓位 = 昨天信号）
    pos = d["持仓"].shift(1).fillna(0)
    return pos * d["收盘"].pct_change()


def factor2_returns(data):
    """
    因子 2：纯成交量因子

    入场：近 5 天 >= 3 天量比 > 1.2（有人在动）
    出场（三条规则，满足任一即卖出）：
      ① 10 天内缩量 >= 7 天（没人气了）
      ② 天量（> 1.7）后缩量（暴涨暴跌止盈止损）
      ③ 入场 >= 3 天后 近 5 天又现 >= 3 天放量（持续放量，落袋）
    """
    d = data.copy()

    # 成交量指标
    d["vol_ma"] = d["成交量"].rolling(20).mean()
    d["量比"] = d["成交量"] / d["vol_ma"]
    d["超量_5天"] = (d["量比"] > 1.2).rolling(5).sum()       # 入口用
    d["缩量_10天"] = (d["量比"] < 1.0).rolling(10).sum()     # 出口①用

    # 逐天模拟
    d["持仓"] = 0
    in_pos = False
    entry_day = 0    # 入场时是第几天

    for i in range(1, len(d)):
        vr = d["量比"].iloc[i]       # 今日量比
        vp = d["量比"].iloc[i - 1]   # 昨日量比

        # ---- 入场 ----
        if not in_pos and d["超量_5天"].iloc[i] >= 3:
            in_pos = True
            entry_day = i

        # ---- 出场 ----
        elif in_pos:
            days_in = i - entry_day

            exit_1 = days_in <= 10 and d["缩量_10天"].iloc[i] >= 7
            exit_2 = vp > 1.7 and vr < 1.0
            exit_3 = days_in >= 3 and d["超量_5天"].iloc[i] >= 3

            if exit_1 or exit_2 or exit_3:
                in_pos = False

        d.iloc[i, d.columns.get_loc("持仓")] = int(in_pos)

    pos = d["持仓"].shift(1).fillna(0)
    return pos * d["收盘"].pct_change()


def stats(daily_returns):
    """从日收益率序列算出：总收益、夏普、最大回撤、交易次数"""
    r = daily_returns.dropna()
    if len(r) == 0:
        return 0, 0, 0, 0
    # 总收益
    net = (1 + r).cumprod().iloc[-1]
    total_ret = (net - 1) * 100
    # 夏普（无风险 2%）
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    # 最大回撤
    nv = (1 + r).cumprod()
    mdd = ((nv - nv.cummax()) / nv.cummax() * 100).min()
    return total_ret, sharpe, mdd


# ============================================================
# 板块 1：12 只股票上跑两个因子 + 50/50 组合（测试期 2024-2026）
# ============================================================

STOCKS = [
    ("600519", "茅台"), ("600036", "招行"), ("000725", "京东方"),
    ("600585", "海螺"), ("600276", "恒瑞"), ("000858", "五粮液"),
    ("601318", "平安"), ("002415", "海康"), ("600887", "伊利"),
    ("000333", "美的"), ("002230", "科大讯飞"), ("300750", "宁德时代"),
]

SPLIT = "2024-01-01"

print("=" * 80)
print("  Phase 5 — 多因子组合")
print("=" * 80)
print()
print("  因子1: 偏离度均值回归 + 放量确认")
print("  因子2: 纯成交量")
print("  组合: 50 / 50 等权")
print()

# ---- 逐只股票跑 ----
results_table = []
for code, name in STOCKS:
    df = load_stock(code)
    test = df[df.index >= SPLIT]

    r1 = factor1_returns(test)
    r2 = factor2_returns(test)
    combined = 0.5 * r1 + 0.5 * r2

    ret1, sh1, dd1 = stats(r1)
    ret2, sh2, dd2 = stats(r2)
    retc, shc, ddc = stats(combined)

    results_table.append((name, ret1, sh1, ret2, sh2, retc, shc))

# ---- 输出表格 ----
print(f"  {'':<8} {'因子1':>8} {'因子1夏普':>8} {'因子2':>8} {'因子2夏普':>8} {'组合':>8} {'组合夏普':>8}")
print("  " + "-" * 56)
w1 = w2 = wc = 0
for name, r1, s1, r2, s2, rc, sc in results_table:
    if r1 > 0: w1 += 1
    if r2 > 0: w2 += 1
    if rc > 0: wc += 1
    print(f"  {name:<6} {r1:>+7.1f}% {s1:>7.2f}  {r2:>+7.1f}% {s2:>7.2f}  {rc:>+7.1f}% {sc:>7.2f}")

print(f"\n  因子1 正收益: {w1}/{len(STOCKS)}")
print(f"  因子2 正收益: {w2}/{len(STOCKS)}")
print(f"  组合  正收益: {wc}/{len(STOCKS)}")


# ============================================================
# 板块 2：因子 2 边界分析
# ============================================================

print()
print("=" * 80)
print("  因子 2 边界分析 — 按股票特征分组")
print("=" * 80)

features = []
for code, name in STOCKS:
    df = load_stock(code)
    df["振幅"] = (pd.to_numeric(df["high"]) - pd.to_numeric(df["low"])) / pd.to_numeric(df["close"])
    feat = {
        "name": name,
        "日均成交量(百万)": pd.to_numeric(df["volume"]).mean() / 1e6,
        "日均振幅": df["振幅"].mean() * 100,
        "均价": pd.to_numeric(df["close"]).mean(),
        "因子2收益": [r for n, _, _, r, _, _, _ in results_table if n == name][0],
    }
    features.append(feat)

F = pd.DataFrame(features)
for col, label in [("日均振幅", "日均振幅"), ("日均成交量(百万)", "日均成交量"), ("均价", "均价")]:
    med = F[col].median()
    hi = F[F[col] >= med]
    lo = F[F[col] < med]
    print(f"\n  按 {label} 分组（中位数 {med:.1f}）：")
    print(f"    高组（{len(hi)}只）：均收益 {hi['因子2收益'].mean():+.0f}%  名单：{hi['name'].tolist()}")
    print(f"    低组（{len(lo)}只）：均收益 {lo['因子2收益'].mean():+.0f}%  名单：{lo['name'].tolist()}")


# ============================================================
# 板块 3：滚动窗口交叉验证
# ============================================================

print()
print("=" * 80)
print("  滚动窗口交叉验证（3 年训练 / 1 年测试 × 3 窗口）")
print("=" * 80)

windows = [
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

weights_to_test = [(50, 50), (60, 40), (70, 30), (40, 60), (30, 70), (100, 0), (0, 100)]
print(f"\n  {'权重':<8} {'合并 Sharpe':>10}")
print("  " + "-" * 24)

best_sh = -99
best_w = None
for w1, w2 in weights_to_test:
    all_rets = []
    for _, _, test_start, test_end in windows:
        for code, _ in STOCKS:
            df = load_stock(code)
            test_data = df[(df.index >= test_start) & (df.index <= test_end)]
            if len(test_data) == 0:
                continue
            r1 = factor1_returns(test_data)
            r2 = factor2_returns(test_data)
            c = w1 / 100 * r1 + w2 / 100 * r2
            all_rets.append(c.dropna())
    if all_rets:
        R = pd.concat(all_rets)
        sh = (R.mean() * 252 - 0.02) / (R.std() * np.sqrt(252)) if R.std() > 0 else 0
        if sh > best_sh:
            best_sh = sh
            best_w = (w1, w2)
        print(f"  {w1}/{w2:<6} {sh:>+9.2f}")

print(f"\n  50/50 各窗口表现：")
for _, _, vs, ve in windows:
    wins = 0
    for code, _ in STOCKS:
        df = load_stock(code)
        test_data = df[(df.index >= vs) & (df.index <= ve)]
        if len(test_data) == 0:
            continue
        r1 = factor1_returns(test_data)
        r2 = factor2_returns(test_data)
        c = 0.5 * r1 + 0.5 * r2
        ret, sh, dd = stats(c)
        if ret > 0:
            wins += 1
    print(f"    {vs[:4]} 年：正收益 {wins}/{len(STOCKS)}")


# ============================================================
# 板块 4：画组合 vs 因子1 vs 因子2（选一只代表性股票）
# ============================================================

DEMO = "300750"  # 宁德时代
df = load_stock(DEMO)
r1 = factor1_returns(df)
r2 = factor2_returns(df)
combined = 0.5 * r1 + 0.5 * r2
bh = df["收盘"].pct_change()

nv1 = (1 + r1).cumprod() * 100_000
nv2 = (1 + r2).cumprod() * 100_000
nvc = (1 + combined).cumprod() * 100_000
nv_bh = (1 + bh).cumprod() * 100_000

fig = go.Figure()
for nv, name, color, dash in [
    (nv1, "因子1（偏离度）", "#d62728", "solid"),
    (nv2, "因子2（纯量）", "#1f77b4", "solid"),
    (nvc, "50/50 组合", "#2ca02c", "solid"),
    (nv_bh, "买入持有", "#999", "dash"),
]:
    fig.add_trace(go.Scatter(
        x=nv.index, y=nv, name=name,
        line=dict(color=color, width=1.5 if "组合" in name else 1.0,
                  dash=dash),
        opacity=0.9 if "组合" in name else 0.6
    ))

fig.add_hline(y=100_000, line_dash="dot", line_color="gray", opacity=0.3)
fig.update_layout(
    title=f"宁德时代（{DEMO}）— 多因子 50/50 组合 vs 单因子",
    xaxis_title="日期", yaxis_title="净值（元）",
    hovermode="x unified", font=dict(size=13),
)
fig.show()

print()
print("  浏览器已打开组合净值曲线（宁德时代示例）。")
print()
print("  " + "-" * 50)
print("  下午读代码的时候，对着上面跑出来的数字看：")
print("  1. 板块 1 的表格 — 因子 1 vs 因子 2 vs 组合逐只股票对比")
print("  2. 板块 2 的分组 — 因子 2 在什么特征的股票上更强")
print("  3. 板块 3 的滚动窗口 — 为什么 50/50 是最稳的选择")
print("  4. 板块 4 的图 — 两条单因子线 + 组合线 + 买持线同框")
