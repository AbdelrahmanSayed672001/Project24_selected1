import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

# Organize data into train, valid, test dirs
os.chdir('E:\\Hard_Drive_E\\Faculty\\Level3\\1st semester\\Selected-1\\Selected_Project\\data\\data\\dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')      
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')        
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')      
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

#ann
#Building and Training the ANN
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3) , activation = "relu" , padding = "same",input_shape =(224,224,3)))
model.add(MaxPool2D(pool_size=(2,2) , strides = 2))
model.add(Conv2D(filters=64 , kernel_size=(3,3) , activation = "relu",padding = "same"))
model.add(MaxPool2D(pool_size=(2,2) , strides = 2))
model.add(Flatten())
model.add(Dense(units=2 , activation = "softmax"))


#model.summary()
model.compile(optimizer = Adam(learning_rate=0.0001) , loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(x=train_batches , validation_data=valid_batches , epochs=10 , verbose=2)

#Making the predictions

predictions = model.predict(x=test_batches , verbose=0)
np.round(predictions)

#Confusion matrix
cm = confusion_matrix(y_true=test_batches.classes , y_pred = np.argmax(predictions , axis = -1))
