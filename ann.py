# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Theano
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras based on Theano and Theano
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # [3, 13)
y = dataset.iloc[:, 13].values

# Encoding categorical data # Country and Female/Male to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # column [1] text into numbers
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2]) # column [2] text into numbers
# только для 1 индекса
onehotencoder = OneHotEncoder(categorical_features = [1]) # 1 - index for encoding
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # 1 индекс уберем to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # warning: solve cross_validation => model_selection

# Feature Scaling # its really compulsory to do it =)
# http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#training-a-naive-bayes-classifier
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

































