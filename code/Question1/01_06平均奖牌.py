import pandas as pd
import numpy as np
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 加载数据
data_file = './第一问终版训练数据.csv'
data = pd.read_csv(data_file, encoding="gbk")

# 按年份排序
data = data.sort_values(by='举办年份')

# 计算前三年金银铜奖牌的平均值
logger.info("开始计算前三年金银铜奖牌的平均值")
start_time = time.time()

# 添加新的列用于存储前三年奖牌的平均值
data['前三年金牌平均值'] = np.nan
data['前三年银牌平均值'] = np.nan
data['前三年铜牌平均值'] = np.nan

# 对数据进行迭代计算前三年金银铜奖牌的平均值
for i in range(3, len(data)):
    try:
        data.loc[i, '前三年金牌平均值'] = np.mean([data.loc[i-1, '金牌数量'], data.loc[i-2, '金牌数量'], data.loc[i-3, '金牌数量']])
        data.loc[i, '前三年银牌平均值'] = np.mean([data.loc[i-1, '银牌数量'], data.loc[i-2, '银牌数量'], data.loc[i-3, '银牌数量']])
        data.loc[i, '前三年铜牌平均值'] = np.mean([data.loc[i-1, '铜牌数量'], data.loc[i-2, '铜牌数量'], data.loc[i-3, '铜牌数量']])
    except Exception as e:
        logger.warning(f"无法计算第{i+1}行的前三年奖牌平均值，跳过该行。错误信息: {e}")

# 删除无法计算的行（前三年金银铜奖牌平均值为NaN的行）
data_cleaned = data.dropna(subset=['前三年金牌平均值', '前三年银牌平均值', '前三年铜牌平均值'])

# 输出处理完成的时间
end_time = time.time()
logger.info(f"前三年金银铜奖牌的平均值计算完毕，耗时 {end_time - start_time:.2f}秒")

# 将清理后的数据写回源文件
data_cleaned.to_csv(data_file, index=False, encoding="gbk")
logger.info(f"清理后的数据已经写回到 {data_file}")
