# Python 基础语法

## 变量和基本类型

```python
price = 1850.5          # 小数
name = "茅台"            # 字符串
count = 100             # 整数
is_active = True        # 布尔值（True/False，首字母大写）
```

变量不需要声明类型，Python 自己推断。

---

## 缩进就是代码块

```python
if price > 1000:
    print("贵")
    print("买不起")
else:
    print("还行")
```

冒号 `:` 开头，下一行缩进 4 格表示属于这个 if。同一层缩进必须对齐。没有大括号。

---

## 列表

```python
stocks = ["茅台", "宁德", "比亚迪"]   # 有序存放

stocks[0]       # → "茅台"（从 0 开始）
stocks[-1]      # → "比亚迪"（-1 倒数第一）
stocks.append("腾讯")   # 末尾追加
len(stocks)     # → 4（列表长度）
```

---

## 字典

```python
stock = {"name": "茅台", "price": 1850, "code": "600519"}

stock["price"]   # → 1850
stock["name"]    # → "茅台"
```

给每个值贴一个名字标签，用名字去取。

---

## 循环

```python
for s in stocks:
    print(s)                # 挨个打印
```

"对于 stocks 里的每一个元素，叫它 s，执行下面缩进的代码。"

---

## 函数

```python
def 算收益率(买入价, 卖出价):
    return (卖出价 - 买入价) / 买入价

r = 算收益率(100, 110)   # r = 0.1
```

---

## import

```python
import pandas as pd
import matplotlib.pyplot as plt
```

引入别人写好的工具箱。`as pd` 是起短名，后面用 `pd.xxx`。

---

## 常用操作一览

```python
# 转整数/小数/字符串
int("123")       # → 123
float("3.14")    # → 3.14
str(100)         # → "100"

# 字符串拼接
"收盘价: " + str(1850)   # → "收盘价: 1850"
f"价格是{price}元"       # → "价格是1850.5元"（推荐这种写法）
```

---

## pandas 快速参考

```python
import pandas as pd

df = pd.read_csv("数据.csv")     # 读 CSV，df = Excel 表格
df.head()                        # 看前 5 行
df.info()                        # 看每列的类型和是否缺数据
df["收盘价"]                      # 取一列
df["日期"] = pd.to_datetime(df["日期"])  # 把文本转成日期
df.set_index("日期", inplace=True)       # 把日期设为索引
df["收益率"] = df["收盘价"].pct_change()  # 算每日涨跌幅
df.dropna()                      # 删掉有空缺的行
```

---

## matplotlib 快速参考

```python
import matplotlib.pyplot as plt

plt.plot(df["收盘价"])              # 画线
plt.title("茅台收盘价")             # 加标题
plt.xlabel("日期")
plt.ylabel("价格")
plt.savefig("figures/price.png")   # 保存图片
plt.show()                         # 显示图
```
