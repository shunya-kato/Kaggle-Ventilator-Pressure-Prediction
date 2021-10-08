import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *


def create_model(input_size, num_feats):
    gpus = tf.config.experimental.list_logical_devices("GPU")

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    else:
        strategy = tf.distribute.get_strategy()
    
    with strategy.scope():
        model = Sequential([
            Input(shape=(input_size, num_feats)),
            Bidirectional(LSTM(700, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            Bidirectional(LSTM(256, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dense(256, activation='selu'),
            #Dropout(0.1)
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mae')
    return model