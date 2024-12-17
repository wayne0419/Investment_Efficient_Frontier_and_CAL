import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Portfolio daily returns data
data = pd.DataFrame({
    'date': [
        '2024-09-20', '2024-09-23', '2024-09-24', '2024-09-25', '2024-09-26', '2024-09-27', '2024-09-30',
        '2024-10-01', '2024-10-04', '2024-10-07', '2024-10-08', '2024-10-09', '2024-10-11', '2024-10-14',
        '2024-10-15', '2024-10-16', '2024-10-17', '2024-10-18', '2024-10-21', '2024-10-22', '2024-10-23',
        '2024-10-24', '2024-10-25', '2024-10-28', '2024-10-29', '2024-10-30', '2024-11-01', '2024-11-04',
        '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08', '2024-11-11'
    ],
    'daily_return': [
        0.0000, 0.0018, 0.0000, 0.0052, 0.0087, 0.0010, -0.0067, 0.0003, 0.0003, 0.0040, -0.0036, -0.0007,
        0.0035, 0.0022, 0.0046, -0.0031, 0.0036, 0.0059, 0.0001, 0.0012, -0.0027, -0.0051, 0.0027, -0.0018,
        -0.0053, -0.0006, -0.0003, 0.0050, 0.0016, 0.0004, 0.0025, 0.0009, -0.0006
    ],
    'market_daily_return': [
        0.0000, 0.0057, 0.0066, 0.0147, 0.0043, -0.0016, -0.0262, 0.0075, -0.0039, 0.0179, -0.0040, 0.0021,
        0.0107, 0.0032, 0.0138, -0.0121, 0.0019, 0.0188, 0.0024, -0.0003, -0.0085, -0.0061, 0.0067, -0.0064,
        -0.0117, -0.0046, -0.0018, 0.0081, 0.0062, 0.0048, 0.0082, 0.0062, -0.0010
    ]
})

# 無風險報酬率（台銀一年期定存）
annual_risk_free_rate = 0.017
risk_free_rate = annual_risk_free_rate / 365

# 模擬天數
simulation_days = len(data)
risk_free_rate_simulated = risk_free_rate * simulation_days

# (1) Return to Risk Ratio
daily_return_mean = data['daily_return'].mean()
daily_return_std = data['daily_return'].std()
return_to_risk_ratio = daily_return_mean / daily_return_std

# (2) Sharpe Ratio
sharpe_ratio = (daily_return_mean - risk_free_rate) / daily_return_std

# (3) Treynor Ratio
# 計算 Beta
X = data['market_daily_return'].values.reshape(-1, 1)
y = data['daily_return'].values.reshape(-1, 1)
reg = LinearRegression().fit(X, y)
beta = reg.coef_[0][0]

treynor_ratio = (daily_return_mean - risk_free_rate) / beta

# (4) Jensen Ratio
jensen_ratio = daily_return_mean - (risk_free_rate + beta * (data['market_daily_return'].mean() - risk_free_rate))

# (5) Beta
# Beta 已經在 Treynor Ratio 的計算中得出

# 打印結果
print("Return to Risk Ratio:", return_to_risk_ratio)
print("Sharpe Ratio:", sharpe_ratio)
print("Treynor Ratio:", treynor_ratio)
print("Jensen Ratio:", jensen_ratio)
print("Beta:", beta)
