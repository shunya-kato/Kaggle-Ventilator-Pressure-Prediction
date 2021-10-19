import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *

def create_model(input_size, num_feats):
    # if single gpu
    # strategy = tf.distribute.get_strategy()
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = Sequential([
            Input(shape=(input_size, num_feats)),
            Bidirectional(LSTM(2048, dropout=0.05, return_sequences=True)),
            Bidirectional(LSTM(1024, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            Bidirectional(LSTM(256, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dense(256, activation='selu'),
            #Dropout(0.1),
            Dense(1)
        ])
        model = model
        model.compile(optimizer='adam', loss='mae')
    return model