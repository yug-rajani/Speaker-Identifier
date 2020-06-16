import os
import tensorflow
import pandas as pd
import librosa
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
def parser(row):
   # function to load files and extract features
   file_name = row.FILE
   print(file_name)
   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print(e)
      print("Error encountered while parsing file: ", file_name)
      return None, None
 
   feature = mfccs
   label = row.NAME
   print([feature, label])
   return  pd.Series([feature, label])

model = load_model('test3.model')
test = pd.read_csv(os.path.join('.', 'test.csv'))
temp = test.apply(parser, axis=1)
temp.columns = ['feature', 'label']


X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))


predictions = model.predict(X)

print(predictions)
print(y)
# print(accuracy_score(y, predictions.round()))
rounded_y=np.argmax(y, axis=1)
print(confusion_matrix(rounded_y, predictions.argmax(axis=1)))
print(classification_report(rounded_y, predictions.argmax(axis=1)))