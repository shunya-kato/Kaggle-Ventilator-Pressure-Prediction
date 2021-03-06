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
from models.dnn_model import create_model
import numpy as np

def main():
    # print("GPUs Available: ", tf.test.is_gpu_available())
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

    train_df = pd.read_csv(path / 'data' / 'input' / 'train.csv')
    test_df = pd.read_csv(path / 'data' / 'input' / 'test.csv')

    train_feats, test_feats = load_features(config['features'])

    train_df = pd.concat([train_df, train_feats], axis=1)
    test_df = pd.concat([test_df, test_feats], axis=1)

    train_df['R'] = train_df['R'].astype(str)
    train_df['C'] = train_df['C'].astype(str)
    test_df['R'] = test_df['R'].astype(str)
    test_df['C'] = test_df['C'].astype(str)

    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

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

    train_df.drop(['pressure', 'id', 'breath_id', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame', 'breath_id_lag2same'], axis=1, inplace=True)
    test_df.drop(['id', 'breath_id', 'count','breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame', 'breath_id_lag2same'], axis=1, inplace=True)

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