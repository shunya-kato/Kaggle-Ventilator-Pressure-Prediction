import pandas as pd
from pathlib import Path
import datetime

def main():
    path = Path(__file__).parent.parent
    files = ["sub_(2021-10-10-20-11)_(0.17449791431427003).csv", "sub_(2021-10-12-10-43)_(0.17985812127590178).csv"]
    submission = pd.read_csv(path / 'data' / 'output' / 'sample_submission.csv')
    for file in files:
        df = pd.read_csv(path/'data'/'output'/file)
        submission['pressure'] += df['pressure']

    submission['pressure'] /= len(files)
    dt = datetime.datetime.now()
    submission.to_csv(path / 'data' / 'output' / f"ensemble_({dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute}).csv", index=False)

if __name__ == '__main__':
    main()