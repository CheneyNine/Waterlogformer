import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# —— 指标函数 —— 
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def kge(y_true, y_pred):
    r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

# —— ARIMA 预测函数 —— 
def arima_forecast(series, order=(1,1,1), window=24, horizon=24, verbose=False):
    """
    输入：
      series: 一维 numpy array
      order: ARIMA(p,d,q)
      window: 用于训练的历史步数
      horizon: 要预测的步数
    返回：
      forecast: 预测值 np.ndarray, 长度 = horizon
      metrics: dict 包含 mse, mae, nse, kge
    """
    if len(series) < window + horizon:
        raise ValueError(f"序列长度需 >= {window + horizon}，当前 {len(series)}")
    
    train = series[-(window + horizon):-horizon]
    test  = series[-horizon:]
    
    model = ARIMA(train, order=order)
    fit = model.fit()
    if verbose:
        print(fit.summary())
    
    pred = fit.forecast(steps=horizon)
    
    mse_v = mean_squared_error(test, pred)
    mae_v = mean_absolute_error(test, pred)
    nse_v = nse(test, pred)
    kge_v = kge(test, pred)
    
    return pred, {'mse': mse_v, 'mae': mae_v, 'nse': nse_v, 'kge': kge_v}


if __name__ == "__main__":
    # —— 1. 读数据 —— 
    df = pd.read_csv(
        "/root/Time-Series-Library-main/dataset/Flood/rain7_flood.csv",
        parse_dates=["date"],
        index_col="date"
    )
    
    # —— 2. 筛选包含 flood 的列 —— 
    flood_cols = [c for c in df.columns if "flood" in c]
    print(f"共找到 {len(flood_cols)} 个 flood 列。")
    
    # —— 3. 循环预测并收集结果 —— 
    results = []
    for col in flood_cols:
        series = df[col].dropna().values
        try:
            pred, metrics = arima_forecast(series, order=(2,1,2), window=24, horizon=24)
        except Exception as e:
            print(f"列 {col} 预测失败：{e}")
            continue
        
        # 把预测和指标合并到一行结果里
        row = {
            'column': col,
            **{f'pred_{i+1}': pred[i] for i in range(len(pred))},
            **metrics
        }
        results.append(row)
    
    # —— 4. 转成 DataFrame 方便查看 —— 
    res_df = pd.DataFrame(results)
    print(res_df)
    
    # 如果要保存：
    res_df.to_csv("flood_arima_forecasts_and_metrics.csv", index=False)