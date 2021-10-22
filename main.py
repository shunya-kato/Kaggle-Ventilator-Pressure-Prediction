from datetime import datetime
import pandas as pd
import json
import sys
import os
import gc
import datetime
sys.path.append(os.path.abspath(".."))
from scripts.load_features import load_features
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from models.dnn_model_large import create_model
import numpy as np

def add_features(df):
    df['cross']= df['u_in'] * df['u_out']
    df['cross2']= df['time_step'] * df['u_out']
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['time_step_cumsum'] = df.groupby(['breath_id'])['time_step'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    print("Step-1...Completed")
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    print("Step-3...Completed")
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    print("Step-4...Completed")
    
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    
    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['breath_id__u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['breath_id__u_in_lag'] = df['breath_id__u_in_lag'] * df['breath_id_lagsame']
    df['breath_id__u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['breath_id__u_in_lag2'] = df['breath_id__u_in_lag2'] * df['breath_id_lag2same']
    print("Step-5...Completed")
    
    df['time_step_diff'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['ewm_u_in_mean'] = (df\
                           .groupby('breath_id')['u_in']\
                           .ewm(halflife=9)\
                           .mean()\
                           .reset_index(level=0,drop=True))
    df[["15_in_sum","15_in_min","15_in_max","15_in_mean"]] = (df\
                                                              .groupby('breath_id')['u_in']\
                                                              .rolling(window=15,min_periods=1)\
                                                              .agg({"15_in_sum":"sum",
                                                                    "15_in_min":"min",
                                                                    "15_in_max":"max",
                                                                    "15_in_mean":"mean"})\
                                                               .reset_index(level=0,drop=True))
    print("Step-6...Completed")
    
    df['u_in_lagback_diff1'] = df['u_in'] - df['u_in_lag_back1']
    df['u_out_lagback_diff1'] = df['u_out'] - df['u_out_lag_back1']
    df['u_in_lagback_diff2'] = df['u_in'] - df['u_in_lag_back2']
    df['u_out_lagback_diff2'] = df['u_out'] - df['u_out_lag_back2']
    print("Step-7...Completed")
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")
    
    return df


def main():
    print("GPUs Available: ", tf.test.is_gpu_available())
    path = Path(__file__).parent
    with open(path / 'configs' / 'default.json') as f:
        config = json.load(f)
    
    seed = config['seed']
    batch_size = config['batch_size']
    input_size = config['input_size']
    epochs = config['epochs']
    n_splits = config['n_splits']
    #lr = config['lr']
    decay_late = config['decay_late']
    decay_epochs = config['decay_epochs']

    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(path / 'data' / 'input' / 'train_no_ileagl_pressure.csv')
    test_df = pd.read_csv(path / 'data' / 'input' / 'test.csv')

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    print(train_df.columns)

    y = train_df['pressure'].to_numpy().reshape(-1, input_size)

    pressure = y.squeeze().reshape(-1,1).astype('float32')

    P_MIN = np.min(pressure)
    P_MAX = np.max(pressure)
    P_STEP = (pressure[1] - pressure[0])[0]
    print('Min pressure: {}'.format(P_MIN))
    print('Max pressure: {}'.format(P_MAX))
    print('Pressure step: {}'.format(P_STEP))
    print('Unique values:  {}'.format(np.unique(pressure).shape[0]))

    del pressure
    gc.collect()

    train_df.drop(['pressure', 'id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame', 'breath_id_lag2same'], axis=1, inplace=True)
    test_df.drop(['id', 'breath_id', 'one', 'count','breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame', 'breath_id_lag2same'], axis=1, inplace=True)

    num_feats = len(train_df.columns)

    print(f"train: {train_df.shape} \ntest: {test_df.shape}")

    scaler = StandardScaler()
    scaler.fit(train_df)
    train = scaler.transform(train_df)
    test = scaler.transform(test_df)

    train = train.reshape(-1, input_size, num_feats)
    test = test.reshape(-1, input_size, num_feats)
    Fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    test_preds = []
    val_score = 0

    for fold, (train_idx, test_idx) in enumerate(Fold.split(train, y)):
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]

        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.75, patience=10, verbose=0)
        #scheduler = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_epochs*((len(train)*0.8)/batch_size), decay_late)
        es = EarlyStopping(monitor='val_loss',mode='min', patience=50, verbose=0,restore_best_weights=True)
        tb = callbacks.TensorBoard(log_dir=path/'logs'/f'fold{fold}', histogram_freq=1)

        model = create_model(input_size, num_feats)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, callbacks=[es, lr, tb])
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

        results = model.evaluate(X_valid, y_valid, batch_size=batch_size)
        val_score += results

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

    dt = datetime.datetime.now()
    val_score /= n_splits
    filename = f'sub_({dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute})_({val_score}).csv'
    submission = pd.read_csv(path / 'data' / 'output' / 'sample_submission.csv')
    submission["pressure"] = np.median(np.vstack(test_preds),axis=0)
    submission["pressure"] = np.round((submission.pressure - P_MIN)/P_STEP) * P_STEP + P_MIN
    submission["pressure"] = np.clip(submission.pressure, P_MIN, P_MAX)
    submission.to_csv(path / 'data' / 'output' / filename, index=False)

if __name__ == '__main__':
    main()