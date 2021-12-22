import numpy as np
import pandas as pd
from pypfopt import CovarianceShrinkage, EfficientFrontier
from pypfopt.expected_returns import mean_historical_return

from deep_portfolio_optimization import sequence_length, daily_returns
from evaluation import get_sharpe


def generate_optimal_allocation(closing_prices):
    # generate target allocation to match
    weights_list = []
    # compute covariance matrix an all dataset for better approximation
    S = CovarianceShrinkage(closing_prices).ledoit_wolf()
    # lower lookback yield target greater cumulative returns but makes allocation more volatile (not portfolio)
    # and harder to learn
    price_lookback = 20
    for index in range(sequence_length, closing_prices.shape[0] - 1):
        sliced_prices = closing_prices.iloc[(index - sequence_length):(index)]
        mu = mean_historical_return(closing_prices.iloc[index - price_lookback:index + 1])
        ef = EfficientFrontier(mu, S)
        try:
            weights = ef.max_sharpe()
            weights_list.append(weights)
        except:
            # equally weighted when the problem is infeasible
            weights_list.append({"N225": 0.25, "STOXX50E": 0.25, "NDX": 0.25, "GSPC": 0.25})
    weightsA = np.array([list(weights.values()) for weights in weights_list]).round(2)
    best_alloc = pd.DataFrame(weightsA, index=daily_returns.index[sequence_length:-1], columns=daily_returns.columns)
    dates_intersection = daily_returns.index.intersection(best_alloc.index)
    daily_returns_int = daily_returns.loc[dates_intersection]
    best_alloc_int = best_alloc.loc[dates_intersection]

    returns_target = (daily_returns_int * best_alloc_int).sum(axis=1)
    print("{} cumulated return target ".format((returns_target + 1).prod() - 1))
    print("{} target std ".format(returns_target.std()))
    print("{} target sharpe ".format(get_sharpe(daily_returns_int * best_alloc_int)))
    return best_alloc, dates_intersection
