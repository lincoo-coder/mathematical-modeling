import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

# 加载已保存的XGBoost模型
xgb_model_gold = joblib.load('../../model/xgb_model_Gold.pkl')
xgb_model_silver = joblib.load('../../model/xgb_model_Silver.pkl')
xgb_model_bronze = joblib.load('../../model/xgb_model_Bronze.pkl')

def load_noc_mapping(noc_mapping_file):
    """
    加载NOC映射文件，将NOC原始和NOC编码的关系读取为字典。
    :param noc_mapping_file: 映射文件的路径
    :return: NOC原始与NOC编码的映射字典
    """
    noc_df = pd.read_csv(noc_mapping_file, encoding='utf-8')
    noc_mapping = dict(zip(noc_df['NOC_原始'], noc_df['NOC_编码']))
    noc_mapping_reverse = dict(zip(noc_df['NOC_编码'], noc_df['NOC_原始']))  # 反向映射
    return noc_mapping, noc_mapping_reverse


def preprocess_2024_data(df, country_code, noc_mapping):
    """
    预处理2024年数据，使标签名与训练时的标签一致，并调整列顺序
    :param df: 包含2024年数据的DataFrame
    :param country_code: 需要处理的国家代码
    :param noc_mapping: NOC编码映射字典
    :return: 处理后的DataFrame
    """
    # 筛选出2024年的数据，并过滤出特定国家的数据
    df_2024 = df[(df['举办年份'] == 2024) & (df['国家标签'] == country_code)]

    if df_2024.empty:
        print(f"没有找到国家代码 {country_code} 的数据")
        return None

    # 修改列名，符合XGBoost模型要求的格式
    df_2024_for_prediction = df_2024.rename(columns={
        '国家排名': '前一届国家排名',
        '金牌数量': '前一届金牌数量',
        '银牌数量': '前一届银牌数量',
        '铜牌数量': '前一届铜牌数量',
        '总计数量': '前一届总计数量',
        '参赛运动员数量': '前一届参赛运动员数量',
        '获奖运动员数量': '前一届获奖运动员数量',
        '获奖运动员占参赛比例': '前一届获奖运动员占参赛比例',
        '运动员获奖比例平均值': '前一届运动员获奖比例平均值',
        '男性比率': '前一届男性比率',
        '参加Sport项数': '前一届参加Sport项数',
        '参加体育项目标签': '前一届参加体育项目标签',
        '参加该体育项目数量标签': '前一届参加该体育项目数量标签',
        '前三年金牌平均值': '前三年金牌平均值',
        '前三年银牌平均值': '前三年银牌平均值',
        '前三年铜牌平均值': '前三年铜牌平均值',
        '举办国家标签': '举办国家标签',
        '举办城市标签': '举办城市标签',
        '国家标签': '国家标签',
        '举办年份': '举办年份'
    })

    # 使用NOC映射将国家标签转换为NOC编码
    df_2024_for_prediction['国家标签'] = df_2024_for_prediction['国家标签'].map(noc_mapping)

    # 确保列顺序与训练时的顺序一致
    training_columns = [
        '前一届国家排名', '前一届金牌数量', '前一届银牌数量', '前一届铜牌数量', '前一届总计数量',
        '前一届参赛运动员数量', '前一届获奖运动员数量', '前一届获奖运动员占参赛比例', '前一届运动员获奖比例平均值',
        '前一届男性比率', '前一届参加Sport项数', '前一届参加体育项目标签', '前一届参加该体育项目数量标签',
        '前三年金牌平均值', '前三年银牌平均值', '前三年铜牌平均值',
        '举办国家标签', '举办城市标签',
        '国家标签', '举办年份'
    ]

    # 按照训练数据的列顺序重新排列2024年数据
    df_2024_for_prediction = df_2024_for_prediction[training_columns]

    # 为预测数据设置特定的举办信息（2028年洛杉矶奥运会）
    df_2024_for_prediction['举办国家标签'] = 265  # 洛杉矶举办国家标签
    df_2024_for_prediction['举办城市标签'] = 9  # 洛杉矶举办城市标签
    df_2024_for_prediction['举办年份'] = 2028  # 预测的举办年份是2028年

    return df_2024_for_prediction


def predict_medals(xgb_model_gold, xgb_model_silver, xgb_model_bronze, df_for_prediction):
    """
    使用XGBoost模型进行奖牌预测
    :param xgb_model_gold: 训练好的金牌XGBoost模型
    :param xgb_model_silver: 训练好的银牌XGBoost模型
    :param xgb_model_bronze: 训练好的铜牌XGBoost模型
    :param df_for_prediction: 预处理后的数据
    :return: 预测的金、银、铜奖牌数
    """
    # 分别使用三个模型进行金、银、铜奖牌的预测
    predicted_gold = xgb_model_gold.predict(df_for_prediction)
    predicted_silver = xgb_model_silver.predict(df_for_prediction)
    predicted_bronze = xgb_model_bronze.predict(df_for_prediction)

    return predicted_gold, predicted_silver, predicted_bronze


def monte_carlo_simulation(df, noc_mapping, noc_mapping_reverse, n_simulations=100):
    simulation_results = []
    unique_country_codes = df[df['举办年份'] == 2024]['国家标签'].unique()

    for country_code in unique_country_codes:
        print(f"正在进行 {country_code} 的蒙特卡洛模拟...")

        # 预处理该国家的2024年数据
        df_2024_for_prediction = preprocess_2024_data(df, country_code, noc_mapping)

        if df_2024_for_prediction is None:
            continue  # 如果该国家没有数据则跳过

        gold_preds, silver_preds, bronze_preds = [], [], []

        # 执行蒙特卡洛模拟
        for _ in range(n_simulations):
            predicted_gold, predicted_silver, predicted_bronze = predict_medals(
                xgb_model_gold, xgb_model_silver, xgb_model_bronze, df_2024_for_prediction
            )

            # 对每个奖牌数添加高斯扰动（扰动范围为1）
            gold_with_noise = np.random.normal(loc=predicted_gold[0], scale=1, size=1)[0]
            silver_with_noise = np.random.normal(loc=predicted_silver[0], scale=1, size=1)[0]
            bronze_with_noise = np.random.normal(loc=predicted_bronze[0], scale=1, size=1)[0]

            # 如果有负值则设定为0，并四舍五入为整数
            gold_preds.append(max(round(gold_with_noise), 0))
            silver_preds.append(max(round(silver_with_noise), 0))
            bronze_preds.append(max(round(bronze_with_noise), 0))

        # 计算奖牌数区间和均值
        gold_min, gold_max, gold_mean = min(gold_preds), max(gold_preds), np.mean(gold_preds)
        silver_min, silver_max, silver_mean = min(silver_preds), max(silver_preds), np.mean(silver_preds)
        bronze_min, bronze_max, bronze_mean = min(bronze_preds), max(bronze_preds), np.mean(bronze_preds)
        total_min = gold_min + silver_min + bronze_min
        total_max = gold_max + silver_max + bronze_max
        total_mean = gold_mean + silver_mean + bronze_mean

        # 获取国家名称
        country_name = noc_mapping_reverse.get(country_code, '未知国家')

        simulation_results.append([
            country_name, country_code, gold_min, gold_max, gold_mean, silver_min, silver_max, silver_mean,
            bronze_min, bronze_max, bronze_mean, total_min, total_max, total_mean
        ])

    # 将结果保存为DataFrame
    columns = ['国家名称', '国家NOC', '金牌数区间最小值', '金牌数区间最大值', '金牌数均值', '银牌数区间最小值',
               '银牌数区间最大值', '银牌数均值',
               '铜牌数区间最小值', '铜牌数区间最大值', '铜牌数均值', '总奖牌数区间最小值', '总奖牌数区间最大值',
               '总奖牌数均值']

    df_simulation = pd.DataFrame(simulation_results, columns=columns)

    # 保存为Excel文件
    df_simulation.to_excel('2028_LA_Olympics_Monte_Carlo_Simulation_with_noise.xlsx', index=False)
    print("蒙特卡洛模拟完成，并已保存为Excel文件。")


# 主函数
def main():
    noc_mapping_file = 'D:/云工作站/所有竞赛内容/competition/数学建模比赛合集/MCM数学建模美赛/mcm2024-mathematical-modeling/代码+论文/代码（除编程手不要编辑）/data/第一问处理的数据/最终结果/映射标准/NOC_映射.csv'
    noc_mapping, noc_mapping_reverse = load_noc_mapping(noc_mapping_file)
    df = pd.read_csv('第一问终版训练数据.csv', encoding='gbk')

    monte_carlo_simulation(df, noc_mapping, noc_mapping_reverse)


if __name__ == '__main__':
    main()
