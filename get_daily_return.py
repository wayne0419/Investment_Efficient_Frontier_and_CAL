import yfinance as yf
import pandas as pd

def fetch_and_calculate(ticker_symbols, start_date, end_date):
    # Fetch last prices for all tickers
    all_data = pd.DataFrame()
    for ticker in ticker_symbols:
        print(f"Fetching data for {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker} from {start_date} to {end_date}.")
            continue
        # Extract the 'Close' column as last prices
        last_prices = data[['Close']].rename(columns={'Close': ticker})
        last_prices.index.name = 'Date'
        if all_data.empty:
            all_data = last_prices
        else:
            all_data = all_data.join(last_prices, how='outer')
    
    # Calculate daily return rates
    daily_returns = all_data.pct_change().dropna()
    
    # Calculate expected daily return rates (mean of daily returns)
    expected_daily_return = daily_returns.mean()
    
    # Calculate daily return rate standard deviations
    std_daily_return = daily_returns.std()
    
    # Calculate covariance matrix of daily return rates
    cov_matrix = daily_returns.cov()
    
    # Package results
    results = {
        "last_prices": all_data,
        "daily_returns": daily_returns,
        "expected_daily_return": expected_daily_return,
        "std_daily_return": std_daily_return,
        "cov_matrix": cov_matrix
    }
    return results

# Example usage
if __name__ == "__main__":
    # Define the list of ticker symbols and the date range
    tickers = ['3081.TWO', '3357.TWO', '3491.TWO', '3711.TW', '4958.TW', '6279.TWO', '6290.TWO', '8069.TWO']  # 使用 yahoo finance 代號
    start = "2023-09-20" # 起始日期
    end = "2024-09-20" # 結束日期

    # Fetch last prices and calculate metrics
    results = fetch_and_calculate(tickers, start, end)
    
    # Display results
    if results["last_prices"] is not None:
        print("\nLast Prices:")
        print(results["last_prices"].tail())
        
        print("\nDaily Return Rates:")
        print(results["daily_returns"].tail())
        
        print("\nExpected Daily Return Rates:")
        print(results["expected_daily_return"])
        
        print("\nDaily Return Rate Standard Deviations:")
        print(results["std_daily_return"])
        
        print("\nCovariance Matrix of Daily Return Rates:")
        print(results["cov_matrix"])
        
        # Save to CSV (optional)
        results["last_prices"].to_csv("last_prices.csv")
        results["daily_returns"].to_csv("daily_returns.csv")
        # results["cov_matrix"].to_csv("covariance_matrix.csv")
