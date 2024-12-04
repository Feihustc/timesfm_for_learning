# %%

import pandas as pd
import numpy as np

data_df = pd.read_csv("/root/autodl-tmp/weather_long_power_several_month.csv")

# %%
datetime_col = "data_time"
target_col = "real_power"
cov_cols = [col for col in data_df.columns if col != datetime_col and col.startswith('predict') and col != 'predicted_power']
# %%
for col in cov_cols:
    correlation = data_df[target_col].corr(data_df[col])
    print(f"{col}：{correlation}")

# %%
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=3)

X = data_df[cov_cols]
y = data_df[target_col]
selector.fit(X, y)

selected_features_mask = selector.get_support()  # 获取布尔掩码
selected_features = X.columns[selected_features_mask].tolist()

scores = selector.scores_
pvalues = selector.pvalues_

feature_scores = pd.DataFrame({
    'Feature': cov_cols,
    'Score': scores,
    'P Value': pvalues,
    'Selected': selected_features_mask
})

feature_scores = feature_scores.sort_values('Score', ascending=False)

print("\nSelected features:")
print(selected_features)

print("\nDetailed feature scores:")
print(feature_scores)

selected_covs = X[selected_features]
# %%
