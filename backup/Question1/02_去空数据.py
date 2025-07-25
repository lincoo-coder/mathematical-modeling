import pandas as pd

df = pd.read_csv("合并后的奥运数据.csv")

# 如果其中有男性比率为空的行，则删除该行
df = df[df["男性比率"].notnull()]

df.to_csv("第一问终版训练数据.csv", index=False)