import yfinance as yf
import pandas as pd
from datetime import datetime

# 定義抓取多個標的每周最後價格的函數
def get_weekly_last_price(symbols, start_date, end_date):
    result = {}

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        # 下載股價數據
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

        if data.empty:
            print(f"No data found for {symbol}")
            continue

        # 確保日期格式正確
        data.index = pd.to_datetime(data.index)
        
        # 新增一列用於標記周數
        data['Week'] = data.index.to_period('W')

        # 找出每周的最後一個價格
        weekly_last_price = data.groupby('Week').tail(1)['Close']

        # 存入結果字典
        result[symbol] = weekly_last_price

    # 將結果轉換為 DataFrame
    result_df = pd.concat(result, axis=1)
    result_df.columns = symbols

    return result_df

# 配置參數
symbols = ["^TWII", "^TELI"]  # 指定標的，例如 TAIEX (^TWII) 和台積電 (2330.TW)
start_date = "2024-09-20"  # 開始日期
end_date = "2024-11-12"    # 結束日期

# 抓取數據
weekly_prices = get_weekly_last_price(symbols, start_date, end_date)

# 保存結果到檔案
weekly_prices.to_csv("weekly_last_prices.csv")

# 顯示結果
print(weekly_prices)
