import pandas as pd
from utils import read_data
from pycaret.time_series import *


train, test = read_data()
train.head()

setup(train, fold=5)
