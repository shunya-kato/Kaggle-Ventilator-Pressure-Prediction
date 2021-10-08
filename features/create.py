import pandas as pd
from base import Feature, get_arguments, generate_features
from pathlib import Path

class area(Feature):
    def create_features(self):
        train['area'] = train['time_step'] * train['u_in']
        self.train['area'] = train.groupby('breath_id')['area'].cumsum()
        test['area'] = test['time_step'] * test['u_in']
        self.test['area'] = test.groupby('breath_id')['area'].cumsum()

if __name__ == '__main__':
    args = get_arguments()

    path = Path(__file__).parent.parent
    train = pd.read_csv(path / 'data' / 'input' / 'train.csv')
    test = pd.read_csv(path / 'data' / 'input' / 'test.csv')
    Feature.dir = path / 'features'

    generate_features(globals(), args.force)