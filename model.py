from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import initializers


def cnn_lstm(input_dim, input_length, n_filter):
    model = Sequential()
    model.add(Conv1D(input_shape=(input_length, input_dim),
                     filters=n_filter,
                     kernel_size=3,
                     padding="same",
                     # activation="relu",
                     strides=1,
                     kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
    model.add(Conv1D(filters=n_filter,
                     kernel_size=3,
                     padding="same",
                     # activation="relu",
                     strides=1,
                     kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Bidirectional(LSTM(2 * n_filter)))
    model.add(Dense(n_filter * 2, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(n_filter, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


if __name__ == '__main__':
    print(cnn_lstm(4, 21, 64).summary())

