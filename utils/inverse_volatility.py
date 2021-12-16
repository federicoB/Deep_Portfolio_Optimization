import pandas as pd

# load strategy data
daily_returns = pd.read_csv("../daily_returns.csv", index_col=0, parse_dates=True)


def get_weights(returns):
    vola = returns.std(axis=0)
    percent = vola / vola.sum()
    inv = 1 / percent
    inv = inv / inv.sum()
    return inv


window = 10

max_size = daily_returns.shape[0]

alloc = pd.DataFrame([get_weights(daily_returns.iloc[i:i + window]) for i in range(0, max_size - window)])

alloc.index = daily_returns.index[window:]

alloc.plot()

alloc.to_csv('../inverse_volatility.csv')
