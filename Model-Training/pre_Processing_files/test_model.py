import os
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

model = load_model('test1.model')
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
# Evaluate predictions
print(accuracy_score(y, predictions.round()))
rounded_y=np.argmax(y, axis=1)
print(confusion_matrix(rounded_y, predictions.argmax(axis=1)))
print(classification_report(rounded_y, predictions.argmax(axis=1)))