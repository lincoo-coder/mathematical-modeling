import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


# 日志函数
def log(message):
    print(f"[INFO] {message}")


# 保存映射并返回映射字典
def save_mapping(mapping, column_name, save_path):
    log(f"保存 {column_name} 映射关系到文件")
    mapping_df = pd.DataFrame(list(mapping.items()), columns=[f"{column_name}_原始", f"{column_name}_编码"])
    mapping_df.to_csv(os.path.join(save_path, f"{column_name}_映射.csv"), index=False)


# 创建标签编码并统一数字映射规则
def create_label_encoding_with_global_mapping(df, columns, save_path, noc_mapping=None, sport_mapping=None):
    """
    对数据框中的指定列进行标签编码，并保存映射结果
    :param df: 原始数据框
    :param columns: 需要转换为标签编码的列名列表
    :param save_path: 保存路径
    :param noc_mapping: NOC列的数字映射规则
    :param sport_mapping: Sport列的数字映射规则
    :return: 转换后的数据框
    """
    label_encoder = LabelEncoder()
    mapping_info = {}

    # 为NOC列应用数字映射
    if noc_mapping is not None and 'NOC' in columns:
        df['NOC'] = df['NOC'].map(noc_mapping).fillna(df['NOC'])
        save_mapping(noc_mapping, 'NOC', save_path)  # 保存NOC映射

    # 为Sport列应用数字映射
    if sport_mapping is not None and 'Sport' in columns:
        df['Sport'] = df['Sport'].map(sport_mapping).fillna(df['Sport'])
        save_mapping(sport_mapping, 'Sport', save_path)  # 保存Sport映射

    # 为所有需要编码的列进行标签编码
    for col in columns:
        if col not in ['NOC', 'Sport']:  # NOC和Sport已经有了单独的映射规则
            log(f"为列 {col} 执行标签编码")
            df[col] = label_encoder.fit_transform(df[col].astype(str))  # 使用LabelEncoder进行标签编码
            mapping_info[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            save_mapping(mapping_info[col], col, save_path)  # 保存映射关系

    # 保存处理后的数据框
    log(f"保存处理后的数据框 {df.name}_编码结果.csv")
    df.to_csv(os.path.join(save_path, f"{df.name}_编码结果.csv"), index=False)

    return df, mapping_info


# ---------------------------------------------------------------main部分----------------------------------------------------------------

def preprocess_data():
    # 读取数据
    log("开始读取数据")
    athletes_df = pd.read_csv("./data/summerOly_athletes.csv")
    hosts_df = pd.read_csv("./data/summerOly_hosts.csv")
    medals_df = pd.read_csv("./data/summerOly_medal_counts_processed.csv")
    programs_df = pd.read_csv("./data/summerOly_programs.csv")

    # 去掉无用的列
    log("去掉无用的列 'Unnamed: 3'")
    hosts_df = hosts_df.drop(columns=['Unnamed: 3'], errors='ignore')

    # 设置保存路径
    save_path = './pre_data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 给数据框设置名称以便保存时使用
    athletes_df.name = '运动员数据'
    hosts_df.name = '主办城市数据'
    medals_df.name = '奖牌数据'
    programs_df.name = '项目数据'

    # 提取并合并所有文件中的NOC和Sport列
    noc_values = pd.concat([athletes_df['NOC'], hosts_df['NOC'], medals_df['NOC']]).unique()
    sport_values = pd.concat([athletes_df['Sport'], programs_df['Sport']]).unique()

    log(f"提取并合并 NOC 和 Sport 的值，去重后进行编码。")

    # 为NOC列创建独立的编码
    noc_encoder = LabelEncoder()
    noc_encoded = noc_encoder.fit_transform(noc_values)
    noc_mapping = dict(zip(noc_values, noc_encoded))
    save_mapping(noc_mapping, 'NOC', save_path)

    # 为Sport列创建独立的编码
    sport_encoder = LabelEncoder()
    sport_encoded = sport_encoder.fit_transform(sport_values)
    sport_mapping = dict(zip(sport_values, sport_encoded))
    save_mapping(sport_mapping, 'Sport', save_path)

    # 提取每个文件中的NOC和Sport列并进行标签编码
    log("对每个文件中的 NOC 和 Sport 列进行标签编码")
    athletes_df['NOC'] = athletes_df['NOC'].map(noc_mapping).fillna(athletes_df['NOC'])
    hosts_df['NOC'] = hosts_df['NOC'].map(noc_mapping).fillna(hosts_df['NOC'])
    medals_df['NOC'] = medals_df['NOC'].map(noc_mapping).fillna(medals_df['NOC'])
    programs_df['Sport'] = programs_df['Sport'].map(sport_mapping).fillna(programs_df['Sport'])

    # 创建标签编码并保存
    athletes_df_encoded, athletes_mapping = create_label_encoding_with_global_mapping(
        athletes_df, ['NOC', 'Sex', 'City', 'Sport', 'Event', 'Medal'], save_path, noc_mapping, sport_mapping)
    hosts_df_encoded, hosts_mapping = create_label_encoding_with_global_mapping(
        hosts_df, ['NOC', 'Host_City'], save_path, noc_mapping, sport_mapping)
    medals_df_encoded, medals_mapping = create_label_encoding_with_global_mapping(
        medals_df, ['NOC'], save_path, noc_mapping, sport_mapping)
    programs_df_encoded, programs_mapping = create_label_encoding_with_global_mapping(
        programs_df, ['Sport'], save_path, noc_mapping, sport_mapping)

    log("处理后的数据保存完毕")


# 调用数据处理函数
preprocess_data()
