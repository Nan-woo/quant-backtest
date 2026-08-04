import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import requests
# 禁用 Windows 系统代理检测（代理会打断东方财富 API 的 HTTPS 连接）
# 必须在任何使用 requests 的库之前执行
os.environ["NO_PROXY"] = "*"
# 同时禁止 requests 从 Windows 注册表读取代理
requests.Session().trust_env = False

import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --------------- 第 1 步：获取 A 股日线数据 ---------------
# stock_zh_a_hist: 东方财富历史日线
# symbol="600519" → 茅台
# adjust="qfq"    → 前复权（把历史价格按分红送股折算）
df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20200101",
                        end_date="20260713", adjust="qfq")

# --------------- 第 2 步：看一眼数据 ---------------
print("========== df.head() ==========")
print(df.head())

print("\n========== df.info() ==========")
df.info()

print(f"\n总行数: {len(df)}")

# --------------- 第 3 步：整理 ---------------
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期")
df.set_index("日期", inplace=True)

# --------------- 第 4 步：画图 ---------------
close = df["收盘"]

plt.figure(figsize=(12, 5))
plt.plot(close.index, close.values, linewidth=0.8, color="#1f77b4")
plt.title("贵州茅台（600519）收盘价走势", fontsize=14)
plt.xlabel("日期")
plt.ylabel("前复权收盘价（元）")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/day1_maotai_price.png", dpi=150)
print("\n图已保存至 figures/day1_maotai_price.png")

print(f"\n期间最高价: {close.max():.2f}")
print(f"期间最低价: {close.min():.2f}")
print(f"最新收盘价: {close.iloc[-1]:.2f}")
