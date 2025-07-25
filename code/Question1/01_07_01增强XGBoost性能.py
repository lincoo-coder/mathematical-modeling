import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib  # 用于保存和加载模型

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 1. 函数：加载数据
def load_data(file_path):
    logger.info(f"加载数据: {file_path}")
    return pd.read_csv(file_path, encoding='gbk')

# 2. 函数：获取前一届数据
def get_previous_edition_data(df):
    logger.info("提取前一届数据...")
    df = df.sort_values(by=['国家标签', '举办年份'])
    df['前一届国家排名'] = df.groupby('国家标签')['国家排名'].shift(1)
    df['前一届金牌数量'] = df.groupby('国家标签')['金牌数量'].shift(1)
    df['前一届银牌数量'] = df.groupby('国家标签')['银牌数量'].shift(1)
    df['前一届铜牌数量'] = df.groupby('国家标签')['铜牌数量'].shift(1)
    df['前一届总计数量'] = df.groupby('国家标签')['总计数量'].shift(1)
    df['前一届参赛运动员数量'] = df.groupby('国家标签')['参赛运动员数量'].shift(1)
    df['前一届获奖运动员数量'] = df.groupby('国家标签')['获奖运动员数量'].shift(1)
    df['前一届获奖运动员占参赛比例'] = df.groupby('国家标签')['获奖运动员占参赛比例'].shift(1)
    df['前一届运动员获奖比例平均值'] = df.groupby('国家标签')['运动员获奖比例平均值'].shift(1)
    df['前一届男性比率'] = df.groupby('国家标签')['男性比率'].shift(1)
    df['前一届参加Sport项数'] = df.groupby('国家标签')['参加Sport项数'].shift(1)
    df['前一届参加体育项目标签'] = df.groupby('国家标签')['参加体育项目标签'].shift(1)
    df['前一届参加该体育项目数量标签'] = df.groupby('国家标签')['参加该体育项目数量标签'].shift(1)
    df['前三年金牌平均值'] = df.groupby('国家标签')['前三年金牌平均值'].shift(1)
    df['前三年银牌平均值'] = df.groupby('国家标签')['前三年银牌平均值'].shift(1)
    df['前三年铜牌平均值'] = df.groupby('国家标签')['前三年铜牌平均值'].shift(1)

    logger.info("前一届数据提取完毕")
    return df.dropna()

# 3. 函数：预处理数据
def preprocess_data(df):
    features = [
        '前一届国家排名', '前一届金牌数量', '前一届银牌数量', '前一届铜牌数量',
        '前一届总计数量', '前一届参赛运动员数量', '前一届获奖运动员数量', '前一届获奖运动员占参赛比例',
        '前一届运动员获奖比例平均值', '前一届男性比率', '前一届参加Sport项数', '前一届参加体育项目标签',
        '前一届参加该体育项目数量标签', '前三年金牌平均值', '前三年银牌平均值', '前三年铜牌平均值',
        '举办国家标签', '举办城市标签', '国家标签', '举办年份'
    ]
    target = ['金牌数量', '银牌数量', '铜牌数量']
    X = df[features]
    y = df[target]
    return X, y

# 4. 训练并评估XGBoost模型，记录MSE
def train_and_evaluate_xgb_model(X_train, y_train):
    """
    Train XGBoost model for each medal type (Gold, Silver, Bronze).
    :param X_train: Training features
    :param y_train: Training targets
    :return: MSE for XGBoost models during training
    """
    # XGBoost model configuration
    xgb_best_model = xgb.XGBRegressor(objective='reg:squarederror',
                                      learning_rate=0.12973169683940733,
                                      max_depth=10,
                                      n_estimators=30000,
                                      subsample=0.9559945203362027,
                                      tree_method='hist',
                                      n_jobs=-1)

    xgb_train_mse = []

    for medal_index in range(3):  # 0: Gold, 1: Silver, 2: Bronze
        logger.info(f"Training XGBoost model for {['Gold', 'Silver', 'Bronze'][medal_index]} medal...")

        # Ensure y_train is in the correct format
        y_train_col = y_train.iloc[:, medal_index].values if isinstance(y_train, pd.DataFrame) else y_train[:, medal_index]

        # Train XGBoost model
        xgb_model = xgb_best_model.fit(X_train, y_train_col)

        # Calculate and record the training MSE
        xgb_mse = mean_squared_error(y_train_col, xgb_model.predict(X_train))

        # Append MSE values for XGBoost model
        xgb_train_mse.append(xgb_mse)

        logger.info(f"XGBoost MSE: {xgb_mse}")

        # Save the trained XGBoost model
        joblib.dump(xgb_model, f'xgb_model_{["Gold", "Silver", "Bronze"][medal_index]}.pkl')

    return xgb_train_mse


# 主流程
def main():
    file_path = '第一问终版训练数据.csv'  # 修改为你的数据路径
    df = load_data(file_path)
    df = get_previous_edition_data(df)
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost models and evaluate MSE
    xgb_train_mse = train_and_evaluate_xgb_model(X_train, y_train)

    logger.info(f"XGBoost MSE for Gold, Silver, Bronze: {xgb_train_mse}")

if __name__ == '__main__':
    main()
