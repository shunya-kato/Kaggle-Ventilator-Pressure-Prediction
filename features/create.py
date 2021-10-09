from re import T
import pandas as pd
from base import Feature, get_arguments, generate_features
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath("."))
from scripts.load_features import load_features
import json

class R__C(Feature):
    def create_features(self):
        train['R'] = train['R'].astype(str)
        train['C'] = train['C'].astype(str)
        test['R'] = test['R'].astype(str)
        test['C'] = test['C'].astype(str)
        self.train['R__C'] = train["R"].astype(str) + '__' + train["C"].astype(str)
        self.test['R__C'] = test["R"].astype(str) + '__' + test["C"].astype(str)

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