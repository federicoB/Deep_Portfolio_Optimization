import tensorflow as tf
from tcn import TCN
from tensorflow import keras
from tensorflow.keras import initializers, layers


def get_tcn_yes_bias(sequence_length, output_size, num_features, tcn_dimension=32, kernel_size=3,
                     skip_connection=False):
    inputs = keras.Input(shape=(sequence_length, num_features))
    x = TCN(tcn_dimension, kernel_size=kernel_size, use_skip_connections=skip_connection, activation='relu',
            kernel_initializer=initializers.RandomNormal(seed=1))(inputs)
    x = layers.Dense(output_size, kernel_initializer=initializers.RandomNormal(seed=1))(x)
    x = tf.nn.softmax(x)
    return keras.Model(inputs=inputs, outputs=x, name="allocationN")
