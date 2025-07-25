import pandas as pd

# 读取数据
raw_data = pd.read_csv('./pre_data/运动员数据_编码结果.csv')

# 从中分离出获奖的数据
awarded_data = raw_data[raw_data['Medal'] != 2]

# 去除重复记录，确保每个运动员每年只计算一次
raw_data_unique = raw_data.drop_duplicates(subset=['NOC', 'Year', 'Name'])
awarded_data_unique = awarded_data.drop_duplicates(subset=['NOC', 'Year', 'Name'])

# 1. 参赛运动员数量
participant_counts = raw_data_unique.groupby(['NOC', 'Year'])['Name'].nunique().reset_index()
participant_counts.rename(columns={'Name': '参赛运动员数量'}, inplace=True)

# 2. 获奖运动员数量
awarded_participant_counts = awarded_data_unique.groupby(['NOC', 'Year'])['Name'].nunique().reset_index()
awarded_participant_counts.rename(columns={'Name': '获奖运动员数量'}, inplace=True)

# 3. 获奖运动员占参赛比例
merged_counts = pd.merge(participant_counts, awarded_participant_counts, on=['NOC', 'Year'], how='left')
merged_counts['获奖运动员占参赛比例'] = (merged_counts['获奖运动员数量'] / merged_counts['参赛运动员数量']).fillna(0).round(4)

# 4. 运动员获奖比例平均值（按Sport项计算）
def calculate_sport_award_ratio(group):
    sport_award_ratio = group.groupby('Sport').apply(lambda x: (x['Medal'] != 2).sum() / len(x['Name'].unique()) if len(x['Name'].unique()) > 0 else 0)
    return sport_award_ratio.mean() if len(sport_award_ratio) > 0 else 0

award_ratio_avg = awarded_data.groupby(['NOC', 'Year']).apply(calculate_sport_award_ratio).reset_index(name='运动员获奖比例平均值')

# 合并运动员获奖比例平均值
merged_counts = pd.merge(merged_counts, award_ratio_avg, on=['NOC', 'Year'], how='left')
merged_counts['运动员获奖比例平均值'] = merged_counts['运动员获奖比例平均值'].fillna(0).round(4)

# 5. 男性比率（每年每国的男性运动员占比）
def calculate_male_ratio(group):
    male_count = (group['Sex'] == 1).sum()  # 计算男性运动员的数量
    total_count = len(group)  # 计算总参赛人数
    return male_count / total_count if total_count > 0 else 0

male_ratio = raw_data.groupby(['NOC', 'Year']).apply(calculate_male_ratio).reset_index(name='男性比率')

# 合并男性比率
merged_counts = pd.merge(merged_counts, male_ratio, on=['NOC', 'Year'], how='left')
merged_counts['男性比率'] = merged_counts['男性比率'].fillna(0).round(4)

# 6. 参加Sport项数
sport_count = raw_data_unique.groupby(['NOC', 'Year'])['Sport'].nunique().reset_index()
sport_count.rename(columns={'Sport': '参加Sport项数'}, inplace=True)

# 合并运动项目数
final_data = pd.merge(merged_counts, sport_count, on=['NOC', 'Year'], how='left')

# 填充缺失值为0，并确保小数保留四位
final_data = final_data.fillna(0).round(4)

# 打印最终的结果
print(final_data)

# 保存结果到CSV文件
final_data.to_csv('运动员统计特征.csv', index=False, encoding='utf-8')
