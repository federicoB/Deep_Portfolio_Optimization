import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from evaluation import multiple_period_evaluation, quick_train_eval
from generate_optimal_allocation import generate_optimal_allocation
from models.tcn import get_tcn_yes_bias
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

best_alloc, dates_intersection = generate_optimal_allocation(closing_prices)
df = df.loc[dates_intersection]

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
