from re import T
import pandas as pd
from base import Feature, get_arguments, generate_features
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
from scripts.load_features import load_features
import json

class time_end(Feature):
    def create_features(self):
        # All entries are first point of each breath_id
        first_df = train.loc[0::80,:]
        # All entries are first point of each breath_id
        last_df = train.loc[79::80,:]
    
        time_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))     
        self.train['time_end'] = train['breath_id'].map(time_end_dict)

        # All entries are first point of each breath_id
        first_df = test.loc[0::80,:]
        # All entries are first point of each breath_id
        last_df = test.loc[79::80,:]
        
        # The Main mode DataFrame and flag
        time_end_dict = dict(zip(last_df['breath_id'], last_df['time_step']))     
        self.test['time_end'] = test['breath_id'].map(time_end_dict)
        

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