import numpy as np

def sequentialize(arr, seq_lenght):
    early_stopping = arr.shape[0] - seq_lenght
    x_dataset = np.zeros((early_stopping, seq_lenght, arr.shape[1]))
    y_dataset = np.zeros((early_stopping,arr.shape[1]))
    for i in range(early_stopping):
        x_dataset[i] = arr[i:i + seq_lenght]
        y_dataset[i] = arr[i+seq_lenght]
    return x_dataset, y_dataset



def sequantialize_and_split(series_dataset,seq_length, train_test_split):
    seq_x, seq_y = sequentialize(series_dataset.copy().values,seq_length)
    dates = series_dataset.index[seq_length:]
    train_dataset_limit = int(seq_x.shape[0] * train_test_split)
    x_train = seq_x[:train_dataset_limit]
    y_train = seq_y[:train_dataset_limit]
    x_val = seq_x[train_dataset_limit:]
    y_val = seq_y[train_dataset_limit:]
    dates_val = dates[train_dataset_limit:]
    return x_train, y_train, x_val, y_val, dates_val
    