#model-new

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
from keras.optimizers import Adam,RMSprop
from keras.utils import np_utils
# from sklearn import metrics 
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


data_dir = '.'
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
# print(temp)


valdation = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
temp1 = train.apply(parser, axis=1)
temp1.columns = ['feature', 'label']
# print(temp1)



#np.random.shuffle(temp)
X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))



val_x = np.array(temp1.feature.tolist())
val_y = np.array(temp1.label.tolist())

lb = LabelEncoder()

val_y = np_utils.to_categorical(lb.fit_transform(val_y))



print("#######################learning model started######################")
num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(40, input_shape=(40,)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(25))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=RMSprop())
print("#######################learning model compile over ######################")
print("#######################learning model training######################")

history = model.fit(X, y, batch_size=32, epochs=1000, verbose=2, validation_data=(val_x, val_y))
score = model.evaluate(	val_x,val_y,verbose=0)

print('Test Loss:',score[0])
print('Test accuracy:',score[1])

model.save(os.path.join('.', 'test2.model'))


