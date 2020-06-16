import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import sys

import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
# y = np.array(temp.label.tolist())
# print(y)
# lb = LabelEncoder()

# y = np_utils.to_categorical(lb.fit_transform(y))

# print(y)
def getLabels(row):
   # function to load files and extract features
   label = row.NAME
   
   return  pd.Series(label)
test = pd.read_csv(os.path.join('.', 'test.csv'))
temp = test.apply(getLabels, axis=1)
y = np.array(temp)
print(y)
lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))

print(y)