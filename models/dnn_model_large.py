import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *

def create_model(input_size, num_feats):
    strategy = tf.distribute.get_strategy()

    with strategy.scope():
        x_input = Input(shape=(input_size, num_feats))
    
        x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
        x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
        x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
        x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
        x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
        
        z2 = Bidirectional(GRU(units=384, return_sequences=True))(x2)
        
        z31 = Multiply()([x3, z2])
        z31 = BatchNormalization()(z31)
        z3 = Bidirectional(GRU(units=256, return_sequences=True))(z31)
        
        z41 = Multiply()([x4, z3])
        z41 = BatchNormalization()(z41)
        z4 = Bidirectional(GRU(units=128, return_sequences=True))(z41)
        
        z51 = Multiply()([x5, z4])
        z51 = BatchNormalization()(z51)
        z5 = Bidirectional(GRU(units=64, return_sequences=True))(z51)
        
        x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
        
        x = Dense(units=128, activation='selu')(x)
        
        x_output = Dense(units=1)(x)


        model = Model(inputs=x_input, outputs=x_output, name='DNN_Model')
        model.compile(optimizer='adam', loss='mae')
    return model