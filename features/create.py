from re import T
import pandas as pd
from base import Feature, get_arguments, generate_features
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
from scripts.load_features import load_features
import json
import numpy as np

class u_out_lagback_diff1(Feature):
    def create_features(self):
        self.train['u_out_lagback_diff1'] = train['u_out'] - train['u_out_lag_back1']
        self.test['u_out_lagback_diff1'] = test['u_out'] - test['u_out_lag_back1']
        
if __name__ == '__main__':
    args = get_arguments()

    path = Path(__file__).parent.parent
    with open(path / 'configs' / 'default.json') as f:
        config = json.load(f)
    train = pd.read_csv(path / 'data' / 'input' / 'train.csv')
    test = pd.read_csv(path / 'data' / 'input' / 'test.csv')
    train_feats, test_feats = load_features(config['features'])
    train = pd.concat([train, train_feats], axis=1)
    test = pd.concat([test, test_feats], axis=1)
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    Feature.dir = path / 'features'
    print(train.head())
    print(test.head())

    generate_features(globals(), args.force)