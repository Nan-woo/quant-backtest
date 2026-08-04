"""
K 线形态扫描 —— 帮你在茅台 6 年数据里找有趣的规律
输出各种值得观察的片段，方便你找因子灵感
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})

df["日收益"] = df["收盘"].pct_change() * 100
df["振幅"] = (df["最高"] - df["最低"]) / df["开盘"] * 100
df["实体"] = (df["收盘"] - df["开盘"]) / df["开盘"] * 100    # 阳线正, 阴线负
df["成交量_20日均"] = df["成交量"].rolling(20).mean()
df["量比"] = df["成交量"] / df["成交量_20日均"]
df["涨跌"] = np.where(df["收盘"] >= df["开盘"], "阳", "阴")

ma20 = df["收盘"].rolling(20).mean()
df["偏离MA"] = (df["收盘"] / ma20 - 1) * 100

print("=" * 70)
print("  茅台 600519 K 线形态扫描（2020-2026）")
print("=" * 70)

# ===== 1. 极端日 =====
print("\n" + "─" * 50)
print("【1. 涨跌幅最大的 10 天】")
print("─" * 50)
top_volatile = df.nlargest(5, "日收益")
top_volatile = pd.concat([top_volatile, df.nsmallest(5, "日收益")])
top_volatile = top_volatile.sort_index()
for _, row in top_volatile.iterrows():
    bar = "█" * int(abs(row["日收益"]))
    direction = "↑" if row["日收益"] > 0 else "↓"
    print(f"  {row.name.date()}  {direction}{abs(row['日收益']):+.2f}%  "
          f"振幅{row['振幅']:.2f}%  {bar}")

# ===== 2. 连涨连跌 =====
print("\n" + "─" * 50)
print("【2. 连涨 / 连跌 序列】")
print("─" * 50)

streak_dir = 0
streak_start = None
streak_count = 0
long_streaks = []

for i in range(len(df)):
    ret = df["日收益"].iloc[i]
    if ret > 0:
        current_dir = 1
    elif ret < 0:
        current_dir = -1
    else:
        continue

    if current_dir == streak_dir:
        streak_count += 1
    else:
        if streak_count >= 4 and streak_start is not None:
            cum = (1 + df["日收益"].iloc[max(0, i - streak_count - 1):i] / 100).prod() - 1
            label = "阳" if streak_dir == 1 else "阴"
            long_streaks.append((streak_start, streak_count, cum * 100, label))
        streak_dir = current_dir
        streak_start = df.index[i]
        streak_count = 1

if streak_count >= 4 and streak_start is not None:
    i_last = len(df) - 1
    cum = (1 + df["日收益"].iloc[i_last - streak_count:i_last + 1] / 100).prod() - 1
    label = "阳" if streak_dir == 1 else "阴"
    long_streaks.append((streak_start, streak_count, cum * 100, label))

# 按累计收益率排序，各取前 5
up_streaks = sorted([s for s in long_streaks if s[3] == "阳"], key=lambda x: x[2], reverse=True)[:5]
down_streaks = sorted([s for s in long_streaks if s[3] == "阴"], key=lambda x: x[2])[:5]

print("  最强的连涨序列：")
for start, cnt, cum, label in up_streaks:
    print(f"    {start.date()}  连涨{cnt}天  累计{cum:+.2f}%")

print("  最惨的连跌序列：")
for start, cnt, cum, label in down_streaks:
    print(f"    {start.date()}  连跌{cnt}天  累计{cum:+.2f}%")

# ===== 3. 放量日 =====
print("\n" + "─" * 50)
print("【3. 放量最大的阳线和阴线】")
print("─" * 50)
top_vol_up = df[df["涨跌"] == "阳"].nlargest(5, "量比")
top_vol_down = df[df["涨跌"] == "阴"].nlargest(5, "量比")

print("  放量阳线（量比越大 = 成交越狂热）：")
for _, row in top_vol_up.iterrows():
    next_day = df[df.index > row.name]
    next_ret = next_day.iloc[0]["日收益"] if len(next_day) > 0 else 0
    print(f"    {row.name.date()}  量比{row['量比']:.1f}x  实体{row['实体']:+.2f}%  "
          f"次日{next_ret:+.2f}%")

print("  放量阴线：")
for _, row in top_vol_down.iterrows():
    next_day = df[df.index > row.name]
    next_ret = next_day.iloc[0]["日收益"] if len(next_day) > 0 else 0
    print(f"    {row.name.date()}  量比{row['量比']:.1f}x  实体{row['实体']:+.2f}%  "
          f"次日{next_ret:+.2f}%")

# ===== 4. 缩量日 =====
print("\n" + "─" * 50)
print("【4. 缩量阳线和阴线（量比 < 0.5）】")
print("─" * 50)
low_vol = df[df["量比"] < 0.5]
low_vol_up = low_vol[low_vol["涨跌"] == "阳"].head(5)
low_vol_down = low_vol[low_vol["涨跌"] == "阴"].head(5)

print("  缩量阳线（冷清清地涨）：")
for _, row in low_vol_up.iterrows():
    next_day = df[df.index > row.name]
    next_ret = next_day.iloc[0]["日收益"] if len(next_day) > 0 else 0
    print(f"    {row.name.date()}  量比{row['量比']:.2f}x  实体{row['实体']:+.2f}%  "
          f"次日{next_ret:+.2f}%")

print("  缩量阴线（冷清清地跌）：")
for _, row in low_vol_down.iterrows():
    next_day = df[df.index > row.name]
    next_ret = next_day.iloc[0]["日收益"] if len(next_day) > 0 else 0
    print(f"    {row.name.date()}  量比{row['量比']:.2f}x  实体{row['实体']:+.2f}%  "
          f"次日{next_ret:+.2f}%")

# ===== 5. 高振幅日 =====
print("\n" + "─" * 50)
print("【5. 振幅最大的 10 天（多空激烈博弈）】")
print("─" * 50)
top_amp = df.nlargest(10, "振幅")
for _, row in top_amp.iterrows():
    next_day = df[df.index > row.name]
    next_ret = next_day.iloc[0]["日收益"] if len(next_day) > 0 else 0
    print(f"    {row.name.date()}  振幅{row['振幅']:.2f}%  实体{row['实体']:+.2f}%  "
          f"次日{next_ret:+.2f}%  ({row['涨跌']})")

# ===== 6. 均线偏离极值 =====
print("\n" + "─" * 50)
print("【6. 偏离 MA20 最多的时刻】")
print("─" * 50)
above = df.nlargest(5, "偏离MA")
below = df.nsmallest(5, "偏离MA")
print("  远高于均线（超买）：")
for _, row in above.iterrows():
    print(f"    {row.name.date()}  偏离{row['偏离MA']:+.2f}%  价格{row['收盘']:.2f}")

print("  远低于均线（超卖）：")
for _, row in below.iterrows():
    print(f"    {row.name.date()}  偏离{row['偏离MA']:+.2f}%  价格{row['收盘']:.2f}")

# ===== 7. 大阳次日和大阴次日 =====
print("\n" + "─" * 50)
print("【7. 大涨 >5% 的次日表现】")
print("─" * 50)
big_days = df[df["日收益"] > 5]
up_after = (df["日收益"].shift(-1)[big_days.index[:-1]] > 0).sum()
total = len(big_days) - 1
avg_next = df["日收益"].shift(-1)[big_days.index[:-1]].mean()
print(f"  大涨 >5% 共 {len(big_days)} 天")
print(f"  次日上涨概率: {up_after}/{total} ({up_after/total*100:.0f}%)")
print(f"  次日平均: {avg_next:+.2f}%")

print("\n" + "─" * 50)
print("【8. 大跌 < -3% 的次日表现】")
print("─" * 50)
big_down = df[df["日收益"] < -3]
up_after_d = (df["日收益"].shift(-1)[big_down.index[:-1]] > 0).sum()
total_d = len(big_down) - 1
avg_next_d = df["日收益"].shift(-1)[big_down.index[:-1]].mean()
print(f"  大跌 < -3% 共 {len(big_down)} 天")
print(f"  次日上涨概率: {up_after_d}/{total_d} ({up_after_d/total_d*100:.0f}%)")
print(f"  次日平均: {avg_next_d:+.2f}%")

# ===== 9. 月收益概览 =====
print("\n" + "─" * 50)
print("【9. 每年每月收益率总览】")
print("─" * 50)
monthly = df["日收益"].resample("ME").apply(lambda x: (1 + x / 100).prod() - 1) * 100
years = sorted(set(monthly.index.year))
for y in years:
    months_str = ""
    for m in range(1, 13):
        try:
            v = monthly.loc[f"{y}-{m:02d}"]
            if v > 5:
                months_str += f" {m}月+{v:.0f}%"
            elif v < -5:
                months_str += f" {m}月{v:.0f}%"
        except:
            pass
    if months_str:
        print(f"  {y}:{months_str}")

print()
print("=" * 70)
print("  看完这些，有没有哪个现象引起了你的注意？")
print("  比如：")
print("  - 放量阳线之后，第二天是涨得多还是跌得多？")
print("  - 连跌 4 天以上之后，第 5 天反弹的概率大吗？")
print("  - 缩量阴线之后放量阳线的那天，后续行情有规律吗？")
print("  - 偏离均线 10% 以上时，接下来是回归还是继续偏离？")
print("=" * 70)
