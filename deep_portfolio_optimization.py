import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tcn import TCN
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import layers

from evaluation import get_sharpe
from evaluation import multiple_period_evaluation, quick_train_eval
from utils import sequentialize

# set the same seed to improve reproducibility
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

daily_returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
closing_prices = pd.read_csv("closing_prices.csv", index_col=0, parse_dates=True)

scaler = preprocessing.MinMaxScaler()
cum_returns = (1 + daily_returns).cumprod(axis=0) - 1
cum_returns = pd.DataFrame(scaler.fit_transform(cum_returns), index=cum_returns.index, columns=cum_returns.columns)
daily_return_scaled = pd.DataFrame(scaler.fit_transform(daily_returns), index=daily_returns.index,
                                   columns=daily_returns.columns)

df = pd.concat([daily_return_scaled, cum_returns], axis=1)

# hyperparams
batch_size = 5

train_test_split = 0.8

sequence_length = 125

epochs = 5

# generate target allocation to match
weights_list = []

from pypfopt import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

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
df = df.loc[dates_intersection]
returns_target = (daily_returns_int * best_alloc_int).sum(axis=1)
print("{} cumulated return target ".format((returns_target + 1).prod() - 1))
print("{} target std ".format(returns_target.std()))
print("{} target sharpe ".format(get_sharpe(daily_returns_int * best_alloc_int)))

best_alloc.plot()


def get_tcn_yes_bias(sequence_length, output_size, num_features, tcn_dimension=32, kernel_size=3,
                     skip_connection=False):
    inputs = keras.Input(shape=(sequence_length, num_features))
    x = TCN(tcn_dimension, kernel_size=kernel_size, use_skip_connections=skip_connection, activation='relu',
            kernel_initializer=initializers.RandomNormal(seed=1))(inputs)
    x = layers.Dense(output_size, kernel_initializer=initializers.RandomNormal(seed=1))(x)
    x = tf.nn.softmax(x)
    return keras.Model(inputs=inputs, outputs=x, name="allocationN")


model = get_tcn_yes_bias(sequence_length, 4, df.shape[-1])
multiple_period_evaluation(model, df, best_alloc, daily_returns, plots=True)

model = get_tcn_yes_bias(sequence_length, 4, df.shape[-1])
data = df.iloc[df.index >= best_alloc.index[0]]
# train on 2014-2018 and evaluate in 2019:

train_data = data.iloc[data.index < '2019-01-01'].values
val_data = data.iloc[data.index >= '2019-01-01']
val_dates = val_data.index
val_data = val_data.values
y_train = best_alloc.iloc[best_alloc.index < '2019-01-01'].values
y_val = best_alloc.iloc[best_alloc.index >= '2019-01-01'].values
model1 = quick_train_eval(model, train_data, val_data, y_train, y_val, batch_size, epochs, sequence_length)
x_val, _ = sequentialize(val_data, sequence_length)
alloc = model1.predict(x_val)
alloc = pd.DataFrame(alloc, columns=daily_returns.columns, index=val_dates[sequence_length:])

fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

for i, name in enumerate(alloc.columns):
    ax.plot(alloc.iloc[:, i], label=name)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('allocation_2020.png')
plt.show()
