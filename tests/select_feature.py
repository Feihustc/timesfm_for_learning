import pandas as pd
import numpy as np

def feature_selection(soure_file, tgt_col, cov_cols, k=3):
    data_df = pd.read_csv(soure_file)


    from sklearn.feature_selection import SelectKBest
    selector = SelectKBest(k=k)

    X = data_df[cov_cols]
    y = data_df[tgt_col]
    selector.fit(X, y)

    selected_features_mask = selector.get_support()  # 获取布尔掩码
    selected_features = X.columns[selected_features_mask].tolist()
    
    return selected_features

    
if __name__=='__main__':
    csv_file="/root/autodl-tmp/weather_long_power_several_month.csv"
    k = 3
    data_df = pd.read_csv(csv_file)
    datetime_col = "data_time"
    target_col = "real_power"
    cov_cols = [col for col in data_df.columns if col != datetime_col and col.startswith('predict') and col != 'predicted_power']
    cov_features = feature_selection(csv_file, k, target_col, cov_cols)
    print(cov_features)