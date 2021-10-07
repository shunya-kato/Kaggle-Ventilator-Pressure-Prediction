import pandas as pd
from pathlib import Path

path = Path(__file__).parent.parent
path = path / 'features'
path = str(path)

def load_features(feats):
    dfs = [pd.read_feather(path+f'/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(path+f'/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test