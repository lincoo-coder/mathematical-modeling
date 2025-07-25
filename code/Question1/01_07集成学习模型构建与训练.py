import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import logging
import joblib  # 用于保存和加载模型
import torch  # 用于检查CUDA是否可用

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 打印当前设备（CPU或GPU）
def check_device():
    if torch.cuda.is_available():
        logger.info(f"当前设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        logger.info("当前设备: CPU")

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

# 4. Function: Train and evaluate models, record MSE per iteration
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train three models (XGBoost, Random Forest, and Stacking) for each medal type.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training targets
    :param y_test: Test targets
    :return: MSE for XGBoost, Random Forest, and Stacking models during training
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    check_device()  # Print the current device

    logger.info("Starting model training...")

    # XGBoost model configuration
    xgb_best_model = xgb.XGBRegressor(objective='reg:squarederror',
                                      learning_rate=0.12973169683940733,
                                      max_depth=10,
                                      n_estimators=200,
                                      subsample=0.9559945203362027,
                                      tree_method='hist',
                                      n_jobs=-1)

    # Random Forest model configuration
    rf_best_model = RandomForestRegressor(n_estimators=100,
                                          min_samples_split=5,
                                          max_depth=20,
                                          n_jobs=-1)

    # Track MSE for each model
    xgb_train_mse = []
    rf_train_mse = []
    stacking_train_mse = []

    for medal_index in range(3):  # 0: Gold, 1: Silver, 2: Bronze
        logger.info(f"Training model for {['Gold', 'Silver', 'Bronze'][medal_index]} medal...")

        # Train XGBoost, Random Forest, and Stacking models
        xgb_model = xgb_best_model.fit(X_train, y_train[:, medal_index])
        rf_model = rf_best_model.fit(X_train, y_train[:, medal_index])
        stacking_model = StackingRegressor(
            estimators=[('xgb', xgb_best_model), ('rf', rf_best_model)],
            final_estimator=LinearRegression()
        )
        stacking_model.fit(X_train, y_train[:, medal_index])

        # Calculate and record the training MSE
        xgb_mse = mean_squared_error(y_train[:, medal_index], xgb_model.predict(X_train))
        rf_mse = mean_squared_error(y_train[:, medal_index], rf_model.predict(X_train))
        stacking_mse = mean_squared_error(y_train[:, medal_index], stacking_model.predict(X_train))

        # Append MSE values for each model
        xgb_train_mse.append(xgb_mse)
        rf_train_mse.append(rf_mse)
        stacking_train_mse.append(stacking_mse)

        logger.info(f"XGBoost MSE: {xgb_mse}, Random Forest MSE: {rf_mse}, Stacking MSE: {stacking_mse}")

        # Save models for each medal type
        joblib.dump(xgb_model, f'xgb_model_{["Gold", "Silver", "Bronze"][medal_index]}.pkl')
        joblib.dump(rf_model, f'rf_model_{["Gold", "Silver", "Bronze"][medal_index]}.pkl')
        joblib.dump(stacking_model, f'stacking_model_{["Gold", "Silver", "Bronze"][medal_index]}.pkl')

    logger.info("Model evaluation completed.")
    return xgb_train_mse, rf_train_mse, stacking_train_mse

# 5. Plot training MSE decline for all medal types in one graph
def plot_training_mse(xgb_train_mse, rf_train_mse, stacking_train_mse):
    """
    Plot the training MSE decline for each model.
    :param xgb_train_mse: XGBoost training MSE
    :param rf_train_mse: Random Forest training MSE
    :param stacking_train_mse: Stacking model training MSE
    """
    logger.info(f"Plotting all medal types training MSE decline...")

    plt.figure(figsize=(10, 6))

    # MSE values for XGBoost, Random Forest, and Stacking
    medals = ['Gold', 'Silver', 'Bronze']
    xgb_mse_values = xgb_train_mse
    rf_mse_values = rf_train_mse
    stacking_mse_values = stacking_train_mse

    # 奥林匹克奖牌配色
    gold_color = '#FFD700'  # 金色
    silver_color = '#C0C0C0'  # 银色
    bronze_color = '#CD7F32'  # 铜色

    # Plot the MSE values
    bar_width = 0.25
    index = np.arange(len(medals))

    plt.bar(index, xgb_mse_values, bar_width, label='XGBoost', color=gold_color)
    plt.bar(index + bar_width, rf_mse_values, bar_width, label='Random Forest', color=silver_color)
    plt.bar(index + 2 * bar_width, stacking_mse_values, bar_width, label='Stacking', color=bronze_color)

    plt.xlabel('Medal Type')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Model Training MSE for Gold, Silver, and Bronze Medals')
    plt.xticks(index + bar_width, medals)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主流程
def main():
    file_path = '第一问终版训练数据.csv'  # 修改为你的数据路径
    df = load_data(file_path)
    df = get_previous_edition_data(df)
    X, y = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and evaluate MSE
    xgb_train_mse, rf_train_mse, stacking_train_mse = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Plot the MSE decline for all medal types in one chart
    plot_training_mse(xgb_train_mse, rf_train_mse, stacking_train_mse)

if __name__ == '__main__':
    main()
