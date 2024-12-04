# import timesfm
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
import pandas as pd
import numpy as np
import pdb

horizon_len=96
context_len=96
backend='gpu'
batch_size=32
# checkpoint_path="/root/autodl-tmp/timesfm-1.0-200m/checkpoints" #raw
# checkpoint_path="/root/autodl-tmp/ckpt/16/run_20241129_145520" # 16
checkpoint_path="/root/autodl-tmp/ckpt/96/run_20241129_135722" #96
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
NUM_LAYERS = 20
MODEL_DIMS = 1280

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7
RANDOM_SEED = 1234
tfm = TimesFm(
    hparams=TimesFmHparams(
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=INPUT_PATCH_LEN,
        output_patch_len=OUTPUT_PATCH_LEN,
        num_layers=NUM_LAYERS,
        model_dims=MODEL_DIMS,
        backend=backend,
        per_core_batch_size=batch_size,
        quantiles=QUANTILES
    )
    ,
    checkpoint=TimesFmCheckpoint(
                    path = checkpoint_path
                    )
)

from collections import defaultdict
datetime_col='data_time'
# df = pd.read_csv("/root/autodl-tmp/timesfm/datasets/weather_long_power_dropnull.csv")
csv_file = "/root/autodl-tmp/weather_long_power_several_month.csv"
df = pd.read_csv(csv_file)
df[datetime_col] = pd.to_datetime(df[datetime_col])
data_df = df.sort_values(datetime_col)

cov_cols = [col for col in data_df.columns if col != datetime_col and col.startswith('predict') and col != 'predicted_power']
ts_cols = 'real_power'

'''
这里加了个选择特征的脚本，从cov_cols中选出与ts_cols相关性最高的3个变量
'''
# from select_feature import feature_selection
# num_cov_cols = feature_selection(csv_file, ts_cols, cov_cols)

'''
    需要修改一下这里的数据结构
    inputs：real_power的历史序列；
    outputs：predict_power；
    num_cov_cols：需要predict点的预测天气，否则用不上预测天气的数据了；      
    基于以上
    context_len（输入长度）和horizon_len（预测长度）需要保持一致；
        
    但是在代码中，如果直接将
    for cols in num_cov_cols:
        examples[cols].append(
            df[cols][start:(context_end := start + context_len)].tolist()
        )
    中 df[cols][start:(context_end := start + context_len)]修改为
    df[cols][context_end:(context_end + horizon_len)]会报错

    Traceback (most recent call last):
    File "/root/autodl-tmp/timesfm/tests/feihu_test.py", line 112, in <module>
        cov_forecast, ols_forecast = tfm.forecast_with_covariates(
    File "/root/miniconda3/lib/python3.10/site-packages/timesfm/timesfm_base.py", line 522, in forecast_with_covariates
        xregs, xregs_on_context, _, _, _ = xreg_lib.BatchedInContextXRegLinear(
    File "/root/miniconda3/lib/python3.10/site-packages/timesfm/xreg_lib.py", line 459, in fit
        x_test = _to_padded_jax_array(x_test)
    File "/root/miniconda3/lib/python3.10/site-packages/timesfm/xreg_lib.py", line 49, in _to_padded_jax_array
        di = 2**math.ceil(math.log2(i)) - i
    ValueError: math domain error
        
    此问题在github中有提到：https://github.com/google-research/timesfm/issues/113，作者的回复为：
    Likely because the dynamic covariates you provide to the function call are not available for the forecasting horizon. 
    The current implementation does not support past-only dynamic covariates.
    There are two hacky ways in the notebook to address this limitation. Please take another look.
        
    排查一下
        
    如果不行，考虑将预测天气往前挪一天，比如2024-09-20的predict_weather值为2024-09-21的predict_weather（即20240920拿到20240921的天气预报数据，用来预测20240921发电功率）
    
    针对这个问题，训练可能也需要调整
'''
def get_batched_data_fn(
    batch_size: int = batch_size,
    context_len: int = context_len,
    horizon_len: int = horizon_len,
    ts_cols: str = ts_cols,
    num_cov_cols: list = num_cov_cols
):
    examples = defaultdict(list)
    num_example = 0
    for start in range(0, len(data_df) - (context_len + horizon_len), horizon_len):
        num_example += 1
        examples["inputs"].append(
            df[ts_cols][start:(context_end := start + context_len)].tolist()
        )
        for cols in num_cov_cols:
            examples[cols].append(
                df[cols][start:context_end + horizon_len].tolist()
            )
        examples["outputs"].append(
            df[ts_cols][context_end:(context_end + horizon_len)].tolist()
        )
        examples['date'].append(
            df[datetime_col][context_end:(context_end + horizon_len)].tolist()
        )
        
    def data_fn():
        for i in range(1 + (num_example - 1) // batch_size):
            yield {k : v[(i * batch_size):((i + 1) * batch_size)] for k, v in examples.items()}
    
    return data_fn

def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

input_data = get_batched_data_fn(batch_size=batch_size)
metrics = defaultdict(list)
import time

predictions = {
    'raw_forecast': [],
    'cov_forecast': [],
    'ols_forecast': [],
    'date':[]
}

for i, example in enumerate(input_data()):
    raw_forecast, _ = tfm.forecast(
        inputs=example['inputs'], freq=[0] * len(example["inputs"])
    )
    start_time = time.time()
    cov_forecast, ols_forecast = tfm.forecast_with_covariates(
        inputs=example["inputs"],
        # dynamic_numerical_covariates={
        #     "radiation": example["predicted_radiation"],
        #     "direction": example["predicted_direction"],
        #     "speed": example["predicted_speed"],
        #     "temperature": example["predicted_temperature"],
        #     "humidity": example["predicted_humidity"],
        #     "pressure": example["predicted_pressure"]
        # },
        dynamic_numerical_covariates={
            col:example[col]
            for col in num_cov_cols
        },
    dynamic_categorical_covariates={},
    static_numerical_covariates={},
    static_categorical_covariates={},
    freq=[0] * len(example["inputs"]),
    xreg_mode="xreg + timesfm",              # default
    ridge=0.0,
    force_on_cpu=False,
    normalize_xreg_target_per_input=True,    # default
    )
    
    raw_forecast_list = raw_forecast.tolist() if hasattr(raw_forecast, 'tolist') else raw_forecast
    cov_forecast_list = cov_forecast.tolist() if hasattr(cov_forecast, 'tolist') else cov_forecast
    ols_forecast_list = ols_forecast.tolist() if hasattr(ols_forecast, 'tolist') else ols_forecast
    predictions['raw_forecast'].extend(raw_forecast_list)
    predictions['cov_forecast'].extend(cov_forecast_list )
    predictions['ols_forecast'].extend(ols_forecast_list)
    predictions['date'].extend(example['date'])

predictions_df = pd.DataFrame(predictions)

flattened_data = []

for _, row in predictions_df.iterrows():
    
    raw_forecasts = row['raw_forecast']
    cov_forecasts = row['cov_forecast']
    ols_forecasts = row['ols_forecast']
    dates=row['date']
    for row, cov, ols, date in zip(raw_forecasts, cov_forecasts, ols_forecasts, dates):
        flattened_data.append(
            {
                'date':str(date),
                'row_forecast':row,
                'cov_forecast':cov,
                'ols_forecast':ols
            }
        )
        
predictions_data=pd.DataFrame(flattened_data)
csv_filename = '/root/autodl-tmp/timesfm/tests/result/96/my96_predictions_1202.csv'
predictions_data.to_csv(csv_filename, index=False)
print(predictions_data.describe())
    
print(
    f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
    end="",
  )
metrics["eval_mae_timesfm"].extend(
    mae(raw_forecast[:, :horizon_len], example["outputs"])
)
metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))
metrics["eval_mse_timesfm"].extend(
    mse(raw_forecast[:, :horizon_len], example["outputs"])
)
metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

# print()

for k, v in metrics.items():
  print(f"{k}: {np.mean(v)}")