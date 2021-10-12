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
from models.bi_lstm import create_model

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
    lr = config['lr']
    decay_late = config['decay_late']
    decay_epochs = config['decay_epochs']

    train_df = pd.read_csv(path / 'data' / 'input' / 'train.csv')
    test_df = pd.read_csv(path / 'data' / 'input' / 'test.csv')

    train_feats, test_feats = load_features(config['features'])

    train_df = pd.concat([train_df, train_feats], axis=1)
    test_df = pd.concat([test_df, test_feats], axis=1)

    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    y = train_df['pressure'].to_numpy().reshape(-1, input_size)

    train_df.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
    test_df.drop(['id', 'breath_id'], axis=1, inplace=True)

    num_feats = len(train_df.columns)

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

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_epochs*((len(train)*0.8)/batch_size), decay_late)
        es = EarlyStopping(monitor='val_loss',mode='min', patience=35, verbose=1,restore_best_weights=True)
        tb = callbacks.TensorBoard(log_dir=path/'logs'/f'fold{fold}', histogram_freq=1)

        model = create_model(input_size, num_feats)

        history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_size, callbacks=[es, callbacks.LearningRateScheduler(scheduler), tb])
        test_preds.append(model.predict(test).squeeze().reshape(-1, 1).squeeze())

        results = model.evaluate(X_valid, y_valid, batch_size=batch_size)
        val_score += results

        del X_train, X_valid, y_train, y_valid, model
        gc.collect()

    dt = datetime.datetime.now()
    val_score /= n_splits
    filename = f'sub_({dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute})_({val_score}).csv'
    submission = pd.read_csv(path / 'data' / 'output' / 'sample_submission.csv')
    submission['pressure'] = sum(test_preds)/n_splits
    submission.to_csv(path / 'data' / 'output' / filename, index=False)

if __name__ == '__main__':
    main()