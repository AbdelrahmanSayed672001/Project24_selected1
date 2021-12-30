import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
sns.set_style('darkgrid')
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read the dataset
dataset = pd.read_csv(r'diabetes.csv')
dataset.head()

print("Shape of Data is ==> ",dataset.shape)
dataset.info()
dataset.describe().T

# from the describtion we note that the dataset has 768 observations with 9 variable. 
#  Target variable is Outcome.
# Well descriptive analysis shows that variable Glucose, BoodPressure,SckinThickness, Insulin and BMI have minimum value 0
#  which does not make any sense, these values are either missing or outliers
# I can see in Pregnancies column, minimum is 0 (May be this is sign for no pregnancy) which is considerable,
#  But maximum month of pregnancy is 17 which does not make any sense. 


# Data Cleaning
# rename the column DiabetesPedigreeFunction to DPF
dataset.rename({'DiabetesPedigreeFunction':'DPF'},inplace = True,axis =1)
dataset.head()

dataset.dtypes

#  function to handle outliers
def std_based(col_name,dataset):
    mean = dataset[col_name].mean()
    std = dataset[col_name].std()
    cut_off = std * 3
    lower, upper = mean - cut_off, mean + cut_off
    new_dataset = dataset[(dataset[col_name] < upper) & (dataset[col_name] > lower)]
    return new_dataset

# check null values
dataset.isnull().any()

#describe Pregnancies column
dataset['Pregnancies'].describe()

# We can see that minimum is 0 which may be considered as no Pregnancy,
#  But maximum is 17 which is not making sense.

#distribution and also boxplot for outliers in Pregnancies feature to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Pregnancies'],ax=axes[0],color='m')
axes[0].set_title('Distribution of Pregnancy',fontdict={'fontsize':8})
axes[0].set_xlabel('Pregnancy Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()


plot01=sns.boxplot('Pregnancies',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

#Treating Outlier and then verifying it
dataset = std_based('Pregnancies',dataset)

#distribution and also boxplot for outliers in Pregnancies feature after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Pregnancies'],ax=axes[0],color='red')
axes[0].set_title('Distribution of Pregnancy',fontdict={'fontsize':8})
axes[0].set_xlabel('Pregnancy Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('Pregnancies',data=dataset,ax=axes[1],orient = 'v',color='yellow')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# describe Glucose column
dataset['Glucose'].describe()

# we will see that Glucose = 0 does not make any sense

# show the distribution and also boxplot for Glucose column to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Glucose'],ax=axes[0],color='b')
axes[0].set_title('Distribution of Glucose',fontdict={'fontsize':8})
axes[0].set_xlabel('Glucose Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('Glucose',data=dataset,ax=axes[1],orient = 'v',color='m')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# we saw that there is no outlier and also distribution is normal , So we will treat 0 with mean value.
dataset.Glucose = dataset.Glucose.replace(0,dataset.Glucose.mean())
dataset.head()

# the distribution and also boxplot for Glucose column after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Glucose'],ax=axes[0],color='r')
axes[0].set_title('Distribution of Glucose',fontdict={'fontsize':8})
axes[0].set_xlabel('Glucose Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('Glucose',data=dataset,ax=axes[1],orient = 'v',color='y')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# show the describtion of the BloodPressure column
dataset.BloodPressure.describe()

# We need to look at BP=0 show we will show the distribution and also boxplot for it to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['BloodPressure'],ax=axes[0],color='m')
axes[0].set_title('Distribution of BloodPressure',fontdict={'fontsize':8})
axes[0].set_xlabel('BloodPressure Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('BloodPressure',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# It looks like there are few Outliers at both higher end and lower end.
# But at higher end maximum BP is 122, So it is considerable.
# Now at lower end BP near 25 is not making sense. 
# So we will treat missing value with medium and then we will also treat outliers.

dataset.BloodPressure = dataset.BloodPressure.replace(0,dataset.BloodPressure.median())
dataset.head()

# handling the outliers
dataset = std_based('BloodPressure',dataset)

# show the distribution and also boxplot for BloodPressure to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['BloodPressure'],ax=axes[0],color='b')
axes[0].set_title('Distribution of BloodPressure',fontdict={'fontsize':8})
axes[0].set_xlabel('BloodPressure Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('BloodPressure',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# show the describtion of SkinThickness column
dataset.SkinThickness.describe()

# Let us look at 0 SkinThickness by showing the distribution and also boxplot for it to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['SkinThickness'],ax=axes[0],color='b')
axes[0].set_title('Distribution of SkinThickness',fontdict={'fontsize':8})
axes[0].set_xlabel('SkinThickness Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('SkinThickness',data=dataset,ax=axes[1],orient = 'v',color='m')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# we will replace the 0 value with the mean
dataset.SkinThickness = dataset.SkinThickness.replace(0,dataset.SkinThickness.mean())
dataset.head()

# handling the outliers
dataset = std_based("SkinThickness",dataset)

# show the distribution and also boxplot for SkinThickness column to show the outliersto show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['SkinThickness'],ax=axes[0],color='green')
axes[0].set_title('Distribution of SkinThickness',fontdict={'fontsize':8})
axes[0].set_xlabel('SkinThickness Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('SkinThickness',data=dataset,ax=axes[1],orient = 'v',color='m')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# describtion of the Insulin column
dataset.Insulin.describe()

# show the distribution and also boxplot for Insulin column to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Insulin'],ax=axes[0],color='b')
axes[0].set_title('Distribution of Insulin',fontdict={'fontsize':8})
axes[0].set_xlabel('Insulin Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('Insulin',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# We can see there are many outliers. So we will fill 0 with Median of Insulin.
# we will also treat Outliers after removing zero.

dataset.Insulin = dataset.Insulin.replace(0,dataset.Insulin.median())
dataset.head()

# handling outliers
dataset = std_based('Insulin',dataset)

# showing the distribution and also boxplot for Insulin column after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Insulin'],ax=axes[0],color='r')
axes[0].set_title('Distribution of Insulin',fontdict={'fontsize':8})
axes[0].set_xlabel('Insulin Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()


plot01=sns.boxplot('Insulin',data=dataset,ax=axes[1],orient = 'v',color='m')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# describtion of the BMI column
dataset.BMI.describe()

# show the distribution and also boxplot for BMI column to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['BMI'],ax=axes[0],color='b')
axes[0].set_title('Distribution of BMI',fontdict={'fontsize':8})
axes[0].set_xlabel('BMI Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('BMI',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# Outliers are considerable, So we will replace zero with mean
dataset.BMI = dataset.BMI.replace(0,dataset.BMI.mean())
dataset.head()

# show the distribution and also boxplot for BMI column after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['BMI'],ax=axes[0],color='m')
axes[0].set_title('Distribution of BMI',fontdict={'fontsize':8})
axes[0].set_xlabel('BMI Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('BMI',data=dataset,ax=axes[1],orient = 'v',color='c')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# describtion od DPF column
dataset.DPF.describe()

# show the distribution and also boxplot for DPF column to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['DPF'],ax=axes[0],color='green')
axes[0].set_title('Distribution of DPF',fontdict={'fontsize':8})
axes[0].set_xlabel('DPF Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('DPF',data=dataset,ax=axes[1],orient = 'v',color='m')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# Outliers are present at higher end , so we will treat them
# handling outliers
dataset = std_based('DPF',dataset)

# show the distribution and also boxplot for DPF column after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['DPF'],ax=axes[0],color='green')
axes[0].set_title('Distribution of DPF',fontdict={'fontsize':8})
axes[0].set_xlabel('DPF Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('DPF',data=dataset,ax=axes[1],orient = 'v')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# describtion of age column
dataset.Age.describe()

# show the distribution and also boxplot for Age column to show the outliers
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Age'],ax=axes[0],color='green')
axes[0].set_title('Distribution of Age',fontdict={'fontsize':8})
axes[0].set_xlabel('Age Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()

plot01=sns.boxplot('Age',data=dataset,ax=axes[1],orient = 'v')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# handle the outliers
dataset = std_based('Age',dataset)

# show the distribution and also boxplot for Age column after modifying it
fig,axes = plt.subplots(nrows=1,ncols=2,dpi=120,figsize = (8,4))

plot00=sns.distplot(dataset['Age'],ax=axes[0],color='green')
axes[0].set_title('Distribution of Age',fontdict={'fontsize':8})
axes[0].set_xlabel('Age Class',fontdict={'fontsize':7})
axes[0].set_ylabel('Frequency/Distrubtion',fontdict={'fontsize':7})
plt.tight_layout()


plot01=sns.boxplot('Age',data=dataset,ax=axes[1],orient = 'v')
axes[1].set_title('Five Point Summary',fontdict={'fontsize':8})
plt.tight_layout()

# Now we are done with missing value and Outliers
# Let us take a look at data and then move ahead with other steps
dataset.head()

# show the shape of the dataset
dataset.shape

# show the info of the dataset 
dataset.info()

# show the variance of the dataset
dataset.var()

# Variance is varying to a greater extent, So we will standardize
# we are removing DPF because variance is very low
dataset.drop('DPF',axis = 1,inplace=True)

# spliting Data
dataset.Outcome.value_counts()

# show the outcome
sns.countplot(dataset['Outcome']).set_title('Distribution of Outcome')
plt.show()

# We can see that Outcome is balance so we need not to Stratify data
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-------------------------------- ANN Implementation -----------------------------------
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu' ))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
history = ann.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size = 32, epochs = 100)

import matplotlib.pyplot as plt
 
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
 
epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
#
# Plot the model accuracy vs Epochs
#
ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax[0].set_title('Training & Validation Accuracy', fontsize=16)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Accuracy', fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
ax[1].set_title('Training & Validation Loss', fontsize=16)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Loss', fontsize=16)
ax[1].legend()


# roc is a useful tool when predicting the probability of a binary outcome
# It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) 
# for a number of different candidate threshold values between 0.0 and 1.0
# Smaller values on the x-axis of the plot indicate lower false positives and higher true negatives
# Larger values on the y-axis of the plot indicate higher true positives and lower false negatives
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# predict probabilities
lr_probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# The confusion matrix shows the ways in which your classification model is confused when it makes predictions
# Example of a confusion matrix in Python
from sklearn.metrics import confusion_matrix

# make predictions
predicted = model.predict(X_test)
expected = y_test

results = confusion_matrix(expected, predicted)

print(results)

