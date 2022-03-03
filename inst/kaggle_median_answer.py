import pandas as pd
from utils import read_data


train, test = read_data()

# Feature Engineering
for df in [train, test]:
    df['weekday'] = df.time.dt.weekday
    df['hour'] = df.time.dt.hour
    df['minute'] = df.time.dt.minute

train
# Compute the median congestion for every place and time of week
medians = train.groupby(['x', 'y', 'direction', 'weekday', 'hour', 'minute']).congestion.median().astype(int)
medians


# Write the submission file
sub = test.merge(medians, 
                 left_on=['x', 'y', 'direction', 'weekday', 'hour', 'minute'],
                 right_index=True)[['congestion']]
sub.reset_index(inplace=True)
sub.to_csv('../submissions/submission_no_machine_learning.csv', index=False)
sub
