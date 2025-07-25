import pandas as pd
import numpy as np


# 日志函数
def log(message):
    print(f"[INFO] {message}")


# 加载数据
def load_data():
    log("正在加载数据...")
    host_city_data = pd.read_csv("../../data/第一问处理的数据/第二阶段/主办城市数据_编码结果.csv")
    medal_data = pd.read_csv("../../data/第一问处理的数据/第二阶段/奖牌数据_编码结果.csv")
    athlete_data = pd.read_csv("../../data/第一问处理的数据/第三阶段/运动员统计特征.csv", encoding='gbk')
    sports_data = pd.read_csv("../../data/第一问处理的数据/第二阶段/项目数据_编码结果.csv")
    log("数据加载完成")

    return host_city_data, medal_data, athlete_data, sports_data



# 转换 sports_data 格式，使其可以合并
def transform_sports_data(sports_data):
    log("正在转换项目数据格式...")
    # 使用 melt 将年份列转换为行
    sports_data_melted = sports_data.melt(id_vars=["Sport"], var_name="Year", value_name="Sports_Participation_Count")

    # 确保年份列是数值型数据
    sports_data_melted["Year"] = pd.to_numeric(sports_data_melted["Year"], errors="coerce")

    log("项目数据格式转换完成")
    return sports_data_melted


# 合并所有数据
def merge_data(host_city_data, medal_data, athlete_data, sports_data):
    # 转换 sports_data 格式
    sports_data_melted = transform_sports_data(sports_data)

    # 合并主办城市数据与奖牌数据，左连接
    merged_data = pd.merge(medal_data, host_city_data, on=["Year"], how="left")

    # 删除右表没有匹配值的行
    merged_data = merged_data.dropna(subset=['Host_City'], how='all')  # 假设 Host_City 列是 host_city_data 中的列名

    # 合并项目数据（转换后的项目数据），左连接
    merged_data = pd.merge(merged_data, sports_data_melted, on=["Year"], how="left")

    # 删除右表没有匹配值的行
    merged_data = merged_data.dropna(subset=['Sport'], how='all')  # 假设 Sport 列是 sports_data_melted 中的列名

    # 合并运动员数据（包括性别、运动项目、奖牌等），左连接
    merged_data = pd.merge(merged_data, athlete_data, on=["Year", "NOC_x", "Sport"], how="left")


    # 保存结果
    merged_data.to_csv("merged_data.csv", index=False)

#
# # 生成前一年的特征
# def generate_previous_year_features(df):
#     log("正在生成前一年的特征...")
#     # 前一年的获奖运动员数量
#     df['前一年的获奖运动员数量'] = df.groupby('NOC')['Awarded_Athletes_Count'].shift(1)
#
#     # 前一年的获奖运动员占其总运动员数量的比例
#     df['前一年的获奖运动员占其总运动员数量的比例'] = df.groupby('NOC')['Awarded_Athletes_Count'].shift(1) / \
#                                                      df.groupby('NOC')['Total_Athletes_Count'].shift(1)
#
#     # 前一年的运动员在所有Sport上的平均获奖比例
#     df['前一年的运动员在所有Sport上的平均获奖比例'] = df.groupby('NOC')['Athletes_Avg_Awarded_Percentage'].shift(1)
#
#     # 前一年的男女性别比例
#     df['前一年的男女性别比例'] = df.groupby('NOC')['Sex_Ratio'].shift(1)
#
#     # 前一年的参赛运动员数量
#     df['前一年的参赛运动员数量'] = df.groupby('NOC')['Competing_Athletes_Count'].shift(1)
#
#     # 前一年的参赛Sport数量
#     df['前一年的参赛Sport数量'] = df.groupby('NOC')['Competing_Sports_Count'].shift(1)
#
#     # 前一年的金银铜牌数量
#     df['前一年的金牌数量'] = df.groupby('NOC')['Gold'].shift(1)
#     df['前一年的银牌数量'] = df.groupby('NOC')['Silver'].shift(1)
#     df['前一年的铜牌数量'] = df.groupby('NOC')['Bronze'].shift(1)
#
#     # 前三年金银铜牌数量的平均值（包括当前年份）
#     df['前三年金牌平均数量'] = df.groupby('NOC')['Gold'].transform(lambda x: x.rolling(3, min_periods=1).mean())
#     df['前三年银牌平均数量'] = df.groupby('NOC')['Silver'].transform(lambda x: x.rolling(3, min_periods=1).mean())
#     df['前三年铜牌平均数量'] = df.groupby('NOC')['Bronze'].transform(lambda x: x.rolling(3, min_periods=1).mean())
#
#     log("前一年的特征生成完成")
#     return df


# 主函数
def main():
    log("数据处理开始...")
    # 加载数据
    host_city_data, medal_data, athlete_data, sports_data = load_data()

    # 打印每个数据的标题
    print("\n数据集标签：")
    for dataset in [host_city_data, medal_data, athlete_data, sports_data]:
        print(dataset.head())

    # 合并数据
    merge_data(host_city_data, medal_data, athlete_data, sports_data)


# 运行主函数
main()
