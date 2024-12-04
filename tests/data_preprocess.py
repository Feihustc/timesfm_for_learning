import pandas as pd
import numpy as np
from scipy import stats

def process_data(file_path, predicted_cols, begin_time, end_time, k=3):
    """
    处理数据的主函数
    
    Parameters:
    file_path: CSV文件路径
    predicted_cols: 需要处理的预测特征列表
    k: 选择相关性最强的特征数量
    
    Returns:
    处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 1. 按时间排序
    df['data_time'] = pd.to_datetime(df['data_time'])
    df = df.sort_values('data_time')
    
    # 2. 计算相关性并选择最相关的k个特征
    # correlations = {}
    # for col in predicted_cols:
    #     correlation = stats.pearsonr(df[col], df['real_power'])[0]
    #     correlations[col] = abs(correlation)
    
    # top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:k]
    # selected_features = [feature[0] for feature in top_features]
    
    # # 保留必要的列
    # columns_to_keep = ['data_time'] + selected_features + ['real_power', 'predicted_power']
    # df = df[columns_to_keep]
    
    # 3. 将预测值向前移动一天
    # for col in selected_features:
    #     df[col] = df[col].shift(-96)
        
    for col in predicted_cols:
        df[col] = df[col].shift(-96)
    
    # 删除最后一行的NaN值
    # df = df.dropna(subset=selected_features)
    
    # df = df[df['data_time'] >= begin_time]
    # df = df[df['data_time'] < end_time]
    
    return df

if __name__=='__main__':
    csv_file="/root/autodl-tmp/timesfm/datasets/gnrx/weather_long_power_dropnull.csv"
    
    data_df = pd.read_csv(csv_file)
    predicted_cols = [col for col in data_df.columns if col.startswith('predict') and col != 'predicted_power']
    # predicted_cols = ['predicted_radiation', 'predicted_direction', 'predicted_speed', 
                    # 'predicted_temperature', 'predicted_humidity', 'predicted_pressure']
    begin_time = pd.Timestamp('2024-04-01 00:00:00')
    end_time = pd.Timestamp('2024-09-24 00:00:00')
    df = process_data(csv_file, predicted_cols, begin_time, end_time, k=3)
    df.to_csv('/root/autodl-tmp/timesfm/datasets/gnrx/processed_train_data.csv', index=False)