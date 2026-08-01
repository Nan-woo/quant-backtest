"""
节假日效应分析 —— 茅台在节日前后有没有规律？
逻辑：春节/中秋/国庆是白酒消费高峰 → 节前备货/送礼 → 可能影响股价
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("data/600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df = df.rename(columns={
    "open": "开盘", "close": "收盘", "high": "最高",
    "low": "最低", "volume": "成交量", "amount": "成交额"
})
df["日收益"] = df["收盘"].pct_change() * 100

# 中国主要节假日（2020-2026），只列交易日附近的日期
# 格式：(节日名, 节前最后一个交易日附近, 节后第一个交易日附近)
holidays = {
    "春节": [
        ("2020-01-24", "2020-01-31"),  # 春节 1.25
        ("2021-02-11", "2021-02-18"),  # 春节 2.12
        ("2022-01-31", "2022-02-07"),  # 春节 2.1
        ("2023-01-21", "2023-01-30"),  # 春节 1.22
        ("2024-02-10", "2024-02-19"),  # 春节 2.10
        ("2025-01-29", "2025-02-05"),  # 春节 1.29
        ("2026-02-17", "2026-02-20"),  # 春节 2.17
    ],
    "国庆": [
        ("2020-09-30", "2020-10-09"),
        ("2021-09-30", "2021-10-08"),
        ("2022-09-30", "2022-10-10"),
        ("2023-09-28", "2023-10-09"),
        ("2024-09-30", "2024-10-08"),
        ("2025-09-30", "2025-10-09"),
    ],
    "中秋": [
        ("2020-09-30", "2020-10-09"),  # 中秋 10.1, 和国庆重合
        ("2021-09-17", "2021-09-22"),  # 中秋 9.21
        ("2022-09-09", "2022-09-13"),  # 中秋 9.10
        ("2023-09-28", "2023-10-09"),  # 中秋 9.29, 和国庆挨着
        ("2024-09-13", "2024-09-18"),  # 中秋 9.17
        ("2025-10-06", "2025-10-09"),  # 中秋 10.6, 和国庆重合
    ],
    "五一": [
        ("2020-04-30", "2020-05-06"),
        ("2021-04-30", "2021-05-06"),
        ("2022-04-29", "2022-05-05"),
        ("2023-04-28", "2023-05-04"),
        ("2024-04-30", "2024-05-06"),
        ("2025-04-30", "2025-05-06"),
    ],
    "端午": [
        ("2020-06-24", "2020-06-29"),
        ("2021-06-11", "2021-06-15"),
        ("2022-06-02", "2022-06-06"),
        ("2023-06-21", "2023-06-26"),
        ("2024-06-07", "2024-06-11"),
        ("2025-05-30", "2025-06-03"),
    ],
    "清明": [
        ("2020-04-03", "2020-04-07"),
        ("2021-04-02", "2021-04-06"),
        ("2022-04-01", "2022-04-06"),
        ("2023-04-04", "2023-04-06"),  # 清明 4.5 本身就是交易日
        ("2024-04-03", "2024-04-08"),
        ("2025-04-03", "2025-04-07"),
    ],
}

print("=" * 65)
print("  茅台 600519 — 节假日效应分析")
print("=" * 65)

def find_nearest_trading_day(target_str, df_idx, direction="before"):
    """找到目标日期最近的实际交易日"""
    target = pd.to_datetime(target_str)
    for offset in range(10):
        if direction == "before":
            check = target - pd.Timedelta(days=offset)
        else:
            check = target + pd.Timedelta(days=offset)
        if check in df_idx:
            return check
    return None

def get_return_window(data, center_date, days_before, days_after):
    """获取某个日期前后 N 天的累计收益"""
    idx = data.index.get_loc(center_date)
    start = max(0, idx - days_before)
    end = min(len(data) - 1, idx + days_after)
    rets = data["日收益"].iloc[start:end+1]
    cum = (1 + rets / 100).prod() - 1
    return cum * 100

# 按节日汇总
all_results = []

for holiday_name, date_pairs in holidays.items():
    pre_rets = []   # 节前 N 天收益
    post_rets = []  # 节后 N 天收益
    pre_5 = []      # 节前一周
    post_5 = []     # 节后一周
    pre_1 = []      # 节前一天
    post_1 = []     # 节后一天
    pre_vol = []    # 节前成交量
    post_vol = []   # 节后成交量

    for pre_str, post_str in date_pairs:
        pre_date = find_nearest_trading_day(pre_str, df.index, "before")
        post_date = find_nearest_trading_day(post_str, df.index, "after")

        if pre_date is None or post_date is None:
            continue

        # 节前 5 个交易日
        if pre_date in df.index:
            pre_start = df.index.get_loc(pre_date) - 4
            if pre_start >= 0:
                seg = df.iloc[pre_start:df.index.get_loc(pre_date)+1]
                pre_5.append((1 + seg["日收益"] / 100).prod() - 1)
            # 节前 1 天
            if df.index.get_loc(pre_date) >= 0:
                pre_1.append(df["日收益"].iloc[df.index.get_loc(pre_date)] / 100)

        # 节后 5 个交易日
        if post_date in df.index:
            post_start = df.index.get_loc(post_date)
            post_end = min(len(df) - 1, post_start + 4)
            if post_end >= post_start:
                seg = df.iloc[post_start:post_end+1]
                post_5.append((1 + seg["日收益"] / 100).prod() - 1)
            # 节后 1 天
            post_1.append(df["日收益"].iloc[df.index.get_loc(post_date)] / 100)

    if pre_5 or post_5:
        all_results.append({
            "节日": holiday_name,
            "年份数": len(pre_5) or len(post_5),
            "节前1天均值": np.mean(pre_1) * 100 if pre_1 else None,
            "节前5天均值": np.mean(pre_5) * 100 if pre_5 else None,
            "节前5天胜率": sum(1 for r in pre_5 if r > 0) / len(pre_5) * 100 if pre_5 else None,
            "节后1天均值": np.mean(post_1) * 100 if post_1 else None,
            "节后5天均值": np.mean(post_5) * 100 if post_5 else None,
            "节后5天胜率": sum(1 for r in post_5 if r > 0) / len(post_5) * 100 if post_5 else None,
        })

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("节后5天均值", ascending=False)

print(f"\n{'节日':<8} {'年份':<6} {'节前1天':>8} {'节前5天':>10} {'节前胜率':>8} "
      f"{'节后1天':>8} {'节后5天':>10} {'节后胜率':>8}")
print("-" * 70)
for _, row in results_df.iterrows():
    def fmt(v):
        if v is None or pd.isna(v):
            return "    N/A"
        return f"{v:+7.2f}%"

    def fmt_pct(v):
        if v is None or pd.isna(v):
            return "   N/A"
        return f"{v:5.0f}%"

    print(f"{row['节日']:<8} {row['年份数']:<6} "
          f"{fmt(row['节前1天均值']):>8} {fmt(row['节前5天均值']):>10} {fmt_pct(row['节前5天胜率']):>8} "
          f"{fmt(row['节后1天均值']):>8} {fmt(row['节后5天均值']):>10} {fmt_pct(row['节后5天胜率']):>8}")

# ===== 详细展开每年每个节日 =====
print("\n" + "─" * 65)
print("  逐年详情")
print("─" * 65)

for holiday_name in ["春节", "国庆", "中秋"]:
    print(f"\n  === {holiday_name} ===")
    detail = []
    for pre_str, post_str in holidays[holiday_name]:
        year = pre_str[:4]
        pre_date = find_nearest_trading_day(pre_str, df.index, "before")
        post_date = find_nearest_trading_day(post_str, df.index, "after")

        if pre_date is None or post_date is None:
            continue

        # 节前 5 天
        pre_cum = None
        if pre_date in df.index:
            pre_idx = df.index.get_loc(pre_date)
            if pre_idx >= 4:
                pre_cum = (1 + df["日收益"].iloc[pre_idx-4:pre_idx+1] / 100).prod() - 1

        # 节后 5 天
        post_cum = None
        if post_date in df.index:
            post_idx = df.index.get_loc(post_date)
            post_end = min(len(df) - 1, post_idx + 4)
            post_cum = (1 + df["日收益"].iloc[post_idx:post_end+1] / 100).prod() - 1

        pre_str_val = f"节前5天{pre_cum*100:+.1f}%" if pre_cum is not None else "无数据"
        post_str_val = f"节后5天{post_cum*100:+.1f}%" if post_cum is not None else "无数据"
        print(f"    {year}  {pre_date.date() if pre_date else '?'} {pre_str_val}  →  "
              f"{post_date.date() if post_date else '?'} {post_str_val}")

# ===== 所有节日合并，看整体规律 =====
print("\n" + "─" * 65)
print("  所有节日合并统计")
print("─" * 65)

all_pre_1, all_pre_5, all_post_1, all_post_5 = [], [], [], []
pre5_by_year, post5_by_year = {}, {}

for holiday_name, date_pairs in holidays.items():
    for pre_str, post_str in date_pairs:
        pre_date = find_nearest_trading_day(pre_str, df.index, "before")
        post_date = find_nearest_trading_day(post_str, df.index, "after")
        year = int(pre_str[:4])

        if pre_date is not None and pre_date in df.index:
            pre_idx = df.index.get_loc(pre_date)
            all_pre_1.append(df["日收益"].iloc[pre_idx] / 100)
            if pre_idx >= 4:
                cum = (1 + df["日收益"].iloc[pre_idx-4:pre_idx+1] / 100).prod() - 1
                all_pre_5.append(cum)
                pre5_by_year.setdefault(year, []).append(cum)

        if post_date is not None and post_date in df.index:
            post_idx = df.index.get_loc(post_date)
            all_post_1.append(df["日收益"].iloc[post_idx] / 100)
            post_end = min(len(df) - 1, post_idx + 4)
            cum = (1 + df["日收益"].iloc[post_idx:post_end+1] / 100).prod() - 1
            all_post_5.append(cum)
            post5_by_year.setdefault(year, []).append(cum)

print(f"\n  节前 1 天：均值 {np.mean(all_pre_1)*100:+.2f}%  "
      f"胜率 {sum(1 for r in all_pre_1 if r>0)/len(all_pre_1)*100:.0f}%  "
      f"样本 {len(all_pre_1)}")
print(f"  节前 5 天：均值 {np.mean(all_pre_5)*100:+.2f}%  "
      f"胜率 {sum(1 for r in all_pre_5 if r>0)/len(all_pre_5)*100:.0f}%  "
      f"样本 {len(all_pre_5)}")
print(f"  节后 1 天：均值 {np.mean(all_post_1)*100:+.2f}%  "
      f"胜率 {sum(1 for r in all_post_1 if r>0)/len(all_post_1)*100:.0f}%  "
      f"样本 {len(all_post_1)}")
print(f"  节后 5 天：均值 {np.mean(all_post_5)*100:+.2f}%  "
      f"胜率 {sum(1 for r in all_post_5 if r>0)/len(all_post_5)*100:.0f}%  "
      f"样本 {len(all_post_5)}")

# 对比：同期非节假日随机采样
print(f"\n  对比 — 非节假日任意 5 天窗口：")
np.random.seed(42)
rand_5day = []
all_dates = list(df.index)
for _ in range(1000):
    r_idx = np.random.randint(4, len(df) - 5)
    cum = (1 + df["日收益"].iloc[r_idx:r_idx+5] / 100).prod() - 1
    rand_5day.append(cum)
rand_5day = np.array(rand_5day)
print(f"    随机 5 天平均：{rand_5day.mean()*100:+.2f}%")
print(f"    随机 5 天胜率：{(rand_5day > 0).sum()/10:.0f}%")
print(f"    节后 5 天 vs 随机：{np.mean(all_post_5)*100:+.2f}% vs {rand_5day.mean()*100:+.2f}%")

print()
