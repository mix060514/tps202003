import pandas as pd
from utils import read_data
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

train, test = read_data()

# Feature Engineering
for df in [train, test]:
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    df['minute'] = df.time.dt.minute
    df['yearmonth'] = df.time.dt.year * 12 + df.time.dt.month
    df['yearmonthday'] = df.time.dt.year * 10000 + df.time.dt.month * 100 + df.time.dt.day
    df['days_in_month'] = df.time.dt.days_in_month
    df['dayofyear'] = df.time.dt.dayofyear
    df['monthday'] = df.time.dt.month * 100 + df.time.dt.day

cat_cols = ['x', 'y', 'direction','weekday','hour','minute']
for c in cat_cols:
    print(f'category: {c}')
    data_tmp = pd.DataFrame({c: train[c], 'target': train['congestion']})
    target_median = data_tmp.groupby(c)['target'].median()
    print(f'target_median:\n{target_median}\n----')

    test[c] = test[c].map(target_median)
    print(f'test[c][:3]:\n{test[c][:3]}\n----')

    tmp = np.repeat(np.nan, train.shape[0])

    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train):
        target_median = data_tmp.iloc[idx_1].groupby(c)['target'].median()
        tmp[idx_2] = train[c].iloc[idx_2].map(target_median)
    
    train[c] = tmp


train.drop('time', axis=1, inplace=True)
test.drop('time', axis=1, inplace=True)

x = train.drop('congestion', axis=1)
y = train['congestion']
tr_x, va_x, tr_y, va_y = train_test_split(x, y)

dtrain = xgb.DMatrix(tr_x, label=tr_y, enable_categorical=True)
dvalid = xgb.DMatrix(va_x, label=va_y, enable_categorical=True)

# params = dict(objective='reg:squarederror', random_state=71, verbosity=1, eval_metric='mae')
params = dict(objective='reg:pseudohubererror', random_state=71, verbosity=1, eval_metric='mae')
num_round = 100000

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20)

te_x = test
dpred = xgb.DMatrix(te_x, enable_categorical=True)
pd.DataFrame(
    dict(row_id=test.index, congestion=model.predict(dpred, ntree_limit=model.best_ntree_limit))
).to_csv('../submissions/result.csv', index=False, )





