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

class sum_in_15(Feature):
    def create_features(self):
        self.train["sum_in_15"] = (train.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .sum().reset_index(level=0,drop=True))
        self.test["sum_in_15"] = (test.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .sum().reset_index(level=0,drop=True))

class min_in_15(Feature):
    def create_features(self):
        self.train["min_in_15"] = (train.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .min().reset_index(level=0,drop=True))
        self.test["min_in_15"] = (test.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .min().reset_index(level=0,drop=True))

class max_in_15(Feature):
    def create_features(self):
        self.train["max_in_15"] = (train.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .max().reset_index(level=0,drop=True))
        self.test["max_in_15"] = (test.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .max().reset_index(level=0,drop=True))

class mean_in_15(Feature):
    def create_features(self):
        self.train["mean_in_15"] = (train.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .mean().reset_index(level=0,drop=True))
        self.test["mean_in_15"] = (test.groupby('breath_id')['u_in'].rolling(window=15,min_periods=1)\
                                    .mean().reset_index(level=0,drop=True))




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