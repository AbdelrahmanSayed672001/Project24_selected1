import ann as ann
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
######################
# image classification using svm
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# ROC
import pandas as pd
import matplotlib.pyplot as plt

#############################
# ROC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc

# matrix

from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#############################

warnings.simplefilter(action='ignore', category=FutureWarning)

# Organize data into train, valid, test dirs

os.chdir('dogs-vs-cats')
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
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10,
                         shuffle=False)

# imgs, labels = next(train_batches)
#############################################
dir = train_path

categories = ['Cat', 'Dog']

data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)

        try:
            pet_img = cv2.resize(pet_img, (50, 50))
            image = np.array(pet_img).flatten()

            data.append([image, label])
        except Exception as e:
            pass

pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.03, random_state=0)
# #
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

# pick = open('medel.sav', 'wb')
# pickle.dump(model, pick)
# pick.close()

pick = open('medel.sav', 'rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['Cat', 'Dog']

print('Accuracy: ', accuracy)
print('prediction is : ', categories[prediction[0]])

mypet = xtest[0].reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.show()

# ROC

y_pred = model.predict(xtest).ravel()

nn_fpr_keras, nn_tpr_keras, nn_tresholds_keras = roc_curve(ytest, y_pred)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)
plt.show()

#matrix

X, y = make_classification(random_state=0)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(xtrain, ytrain)
SVC(random_state=0)
plot_confusion_matrix(clf, xtest, ytest)
plt.show()

