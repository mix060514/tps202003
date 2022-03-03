import pandas as pd

def read_data():
    train = pd.read_csv('../data/train.csv',index_col='row_id', parse_dates=['time'])
    pred = pd.read_csv('../data/test.csv',index_col='row_id', parse_dates=['time'])
    return train, pred

