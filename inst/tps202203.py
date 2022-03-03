import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
train = pd.read_csv('../data/train.csv')
train.shape
pred = pd.read_csv('../data/test.csv')
pred
xx = pd.concat([train, pred])
xx = xx.astype(dict(time='category', direction='category'), )
train = xx.iloc[:848835, :]
pred = xx.iloc[848835:, :]

x = train.drop(['row_id', 'congestion'], axis = 1)
x = x.astype(dict(time='category', direction='category'), )
y = train['congestion']
tr_x, va_x, tr_y, va_y = train_test_split(x, y)

dtrain = xgb.DMatrix(tr_x, label=tr_y, enable_categorical=True)
dvalid = xgb.DMatrix(va_x, label=va_y, enable_categorical=True)

params = dict(objective='reg:squarederror', random_state=71, verbosity=1, 
eval_metric='mae')
num_round = 100000

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=20)

pr_x = pred.drop(['row_id', 'congestion'], axis = 1)
pr_x = pr_x.astype(dict(time='category', direction='category'), )
dpred = xgb.DMatrix(pr_x, enable_categorical=True)
pd.DataFrame(
    dict(row_id=pred['row_id'], congestion=model.predict(dpred, ntree_limit=model.best_ntree_limit))
).to_csv('../submissions/result.csv', index=False, )
