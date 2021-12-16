import pandas as pd

# load prices data
sp500 = pd.read_csv("../prices/GSPC.csv", index_col=0, parse_dates=True)
nasdaq = pd.read_csv("../prices/NDX.csv", index_col=0, parse_dates=True)
eurostoxx = pd.read_csv("../prices/STOXX50E.csv", index_col=0, parse_dates=True)
nikkei = pd.read_csv("../prices/N225.csv", index_col=0, parse_dates=True)
# remove duplicates keep last one
sp500 = sp500[~sp500.index.duplicated(keep='last')]
nasdaq = nasdaq[~nasdaq.index.duplicated(keep='last')]
eurostoxx = eurostoxx[~eurostoxx.index.duplicated(keep='last')]
nikkei = nikkei[~nikkei.index.duplicated(keep='last')]

# make union between indexes
index_union = sp500.index.union(eurostoxx.index.union(nikkei.index))
open_close_prices = pd.DataFrame(index=index_union)
open_close_prices.loc[sp500.index, 'sp500_open'] = sp500.loc[:, 'Open']
open_close_prices.loc[sp500.index, 'sp500_close'] = sp500.loc[:, 'Close']
open_close_prices.loc[sp500.index, 'nasdaq_open'] = nasdaq.loc[:, 'Open']
open_close_prices.loc[sp500.index, 'nasdaq_close'] = nasdaq.loc[:, 'Close']
# intersection between american and european + japanese opening days
open_close_prices.loc[eurostoxx.index, 'euro_open'] = eurostoxx.loc[:, 'Open']
open_close_prices.loc[eurostoxx.index, 'euro_close'] = eurostoxx.loc[:, 'Close']
open_close_prices.loc[nikkei.index, 'nikkei_open'] = nikkei.loc[:, 'Open']
open_close_prices.loc[nikkei.index, 'nikkei_close'] = nikkei.loc[:, 'Close']

# fill nan values with open=close (no return)
open_close_prices['sp500_close'].fillna(method='ffill', inplace=True)
open_close_prices['nasdaq_close'].fillna(method='ffill', inplace=True)
open_close_prices['nikkei_close'].fillna(method='ffill', inplace=True)
open_close_prices['euro_close'].fillna(method='ffill', inplace=True)
open_close_prices['sp500_open'].fillna(open_close_prices['sp500_close'], inplace=True)
open_close_prices['nasdaq_open'].fillna(open_close_prices['nasdaq_close'], inplace=True)
open_close_prices['nikkei_open'].fillna(open_close_prices['nikkei_close'], inplace=True)
open_close_prices['euro_open'].fillna(open_close_prices['euro_close'], inplace=True)

close_prices = pd.DataFrame(index=open_close_prices.index)
close_prices['N225'] = open_close_prices['nikkei_close']
close_prices['STOXX50E'] = open_close_prices['euro_close']
close_prices['NDX'] = open_close_prices['nasdaq_close']
close_prices['GSPC'] = open_close_prices['sp500_close']
close_prices.fillna(method='bfill', inplace=True)
close_prices.to_csv('../closing_prices.csv')

daily_returns = pd.DataFrame(index=open_close_prices.index)
daily_returns['N225'] = \
    (open_close_prices['nikkei_close'] - open_close_prices['nikkei_open']) / open_close_prices['nikkei_open']
daily_returns['STOXX50E'] = \
    (open_close_prices['euro_close'] - open_close_prices['euro_open']) / open_close_prices['euro_open']
daily_returns['NDX'] = \
    (open_close_prices['nasdaq_close'] - open_close_prices['nasdaq_open']) / open_close_prices['nasdaq_open']
daily_returns['GSPC'] = \
    (open_close_prices['sp500_close'] - open_close_prices['sp500_open']) / open_close_prices['sp500_open']
daily_returns = daily_returns.fillna(0)

daily_returns.to_csv('../daily_returns.csv')
