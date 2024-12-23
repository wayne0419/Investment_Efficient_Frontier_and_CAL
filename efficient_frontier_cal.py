import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

def get_asset_data(tickers, start_date, end_date):
    """
    Fetch historical price data for specified assets.

    Parameters:
    - tickers: List of asset ticker symbols.
    - start_date: Start date for the data.
    - end_date: End date for the data.

    Returns:
    - A DataFrame of daily adjusted close prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_daily_returns(price_data):
    """
    Calculate daily returns from adjusted close prices.

    Parameters:
    - price_data: A DataFrame of asset prices.

    Returns:
    - A DataFrame of daily returns.
    """
    daily_returns = price_data.pct_change().dropna()
    return daily_returns

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and risk (standard deviation).

    Parameters:
    - weights: A list of portfolio weights.
    - mean_returns: Mean returns of the assets.
    - cov_matrix: Covariance matrix of asset returns.

    Returns:
    - Portfolio return and portfolio risk (std dev).
    """
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk

def min_variance(mean_returns, cov_matrix):
    """
    Calculate the minimum variance portfolio with no short selling.

    Parameters:
    - mean_returns: Mean returns of the assets.
    - cov_matrix: Covariance matrix of asset returns.

    Returns:
    - Weights of the minimum variance portfolio.
    """
    num_assets = len(mean_returns)
    args = (cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[1],
                      num_assets * [1. / num_assets], args=args, bounds=bounds, constraints=constraints)
    return result.x

def efficient_frontier(mean_returns, cov_matrix, target_returns):
    """
    Generate the efficient frontier portfolios with no short selling.

    Parameters:
    - mean_returns: Mean returns of the assets.
    - cov_matrix: Covariance matrix of asset returns.
    - target_returns: Array of target returns for efficient portfolios.

    Returns:
    - A tuple of risks and returns for efficient portfolios.
    """
    num_assets = len(mean_returns)
    risks = []
    returns = []

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        result = minimize(lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[1],
                          num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
        if result.success:
            risks.append(calculate_portfolio_performance(result.x, mean_returns, cov_matrix)[1])
            returns.append(target)
    return risks, returns

def calculate_tangent_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """
    Calculate the tangent portfolio with no short selling.

    Parameters:
    - mean_returns: Mean returns of the assets.
    - cov_matrix: Covariance matrix of asset returns.
    - risk_free_rate: Daily risk-free rate.

    Returns:
    - Tangent portfolio weights, return, and risk.
    """
    excess_returns = mean_returns - risk_free_rate
    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(lambda x: -np.dot(x, excess_returns) / calculate_portfolio_performance(x, mean_returns, cov_matrix)[1],
                      num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
    weights = result.x
    portfolio_return, portfolio_risk = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return weights, portfolio_return, portfolio_risk

def plot_efficient_frontier_and_cal(frontier_risks, frontier_returns, tangent_risk, tangent_return, risk_free_rate, asset_risks, asset_returns, tickers):
    """
    Plot the efficient frontier with the Capital Allocation Line (CAL).

    Parameters:
    - frontier_risks: Risks of the efficient frontier portfolios.
    - frontier_returns: Returns of the efficient frontier portfolios.
    - tangent_risk: Risk of the tangent portfolio.
    - tangent_return: Return of the tangent portfolio.
    - risk_free_rate: Risk-free rate for the CAL.
    - asset_risks: Risks (std deviations) of individual assets.
    - asset_returns: Mean returns of individual assets.
    - tickers: List of asset ticker symbols.
    """
    # CAL line
    cal_x = np.linspace(0, max(frontier_risks), 100)
    cal_y = risk_free_rate + (tangent_return - risk_free_rate) / tangent_risk * cal_x

    plt.figure(figsize=(10, 6))
    plt.plot(frontier_risks, frontier_returns, label='Efficient Frontier', color='blue', linewidth=2)
    plt.plot(cal_x, cal_y, label='Capital Allocation Line (CAL)', linestyle='--', color='green')
    plt.scatter(tangent_risk, tangent_return, color='red', label='Tangent Portfolio', zorder=5)
    plt.scatter(asset_risks, asset_returns, color='orange', label='Assets', zorder=5)

    # Label assets
    for i, ticker in enumerate(tickers):
        plt.text(asset_risks[i], asset_returns[i], ticker, fontsize=9, ha='right', va='bottom')

    plt.title('Efficient Frontier and Capital Allocation Line (CAL) - No Short Selling')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return (Daily)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # User input: Tickers and Date range
    tickers = ['3081.TWO', '3357.TWO', '3491.TWO', '3711.TW', '4958.TW', '6279.TWO', '6290.TWO', '8069.TWO']  # Example assets
    start_date = '2024-06-20'
    end_date = '2024-09-20'
    annual_risk_free_rate = 0.017  # Annualized risk-free rate
    daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1  # Convert to daily risk-free rate

    # Step 1: Fetch price data
    price_data = get_asset_data(tickers, start_date, end_date)

    # Step 2: Calculate daily returns
    daily_returns = calculate_daily_returns(price_data)

    # Step 3: Calculate mean returns and covariance matrix
    mean_returns = daily_returns.mean()  # Daily mean returns
    cov_matrix = daily_returns.cov()     # Daily covariance matrix
    asset_risks = daily_returns.std()    # Daily standard deviations

    # Step 4: Calculate Efficient Frontier (No Short Selling)
    target_returns = np.linspace(min(mean_returns), max(mean_returns), 50)
    frontier_risks, frontier_returns = efficient_frontier(mean_returns, cov_matrix, target_returns)

    # Step 5: Calculate Tangent Portfolio (CAL) - No Short Selling
    tangent_weights, tangent_return, tangent_risk = calculate_tangent_portfolio(mean_returns, cov_matrix, daily_risk_free_rate)
    print(f"Tangent Portfolio Weights: {tangent_weights}")
    print(f"Tangent Portfolio Return: {tangent_return:.5f}")
    print(f"Tangent Portfolio Risk: {tangent_risk:.5f}")

    # Step 6: Plot Efficient Frontier and CAL (No Short Selling)
    plot_efficient_frontier_and_cal(frontier_risks, frontier_returns, tangent_risk, tangent_return, daily_risk_free_rate, asset_risks, mean_returns, tickers)
