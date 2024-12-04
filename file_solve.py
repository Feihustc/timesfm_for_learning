from collections import defaultdict
import pandas as pd
datetime_col='data_time'
df = pd.read_csv("/root/autodl-tmp/timesfm/tests/result/16/my16_predictions.csv")
a = len(df)
print(a)
df[datetime_col] = pd.to_datetime(df[datetime_col])
data_df = df.sort_values(datetime_col)
cutoff_date = pd.Timestamp('2024-04-01 00:00:00')  # 更改为你需要的时间点
df_filtered = data_df[data_df['data_time'] > cutoff_date]
cutoff_date = pd.Timestamp('2024-09-25 00:00:00') 
df_filtered = df_filtered[df_filtered['data_time'] < cutoff_date ]
df = df_filtered.sort_values('data_time', ascending=False)
df.to_csv('weather_long_power_several_month.csv', index=False)
