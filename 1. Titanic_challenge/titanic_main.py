#!/usr/bin/env python


# Importing all needed libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings

warnings.filterwarnings(action="ignore")

# Analyzing the dataset

testing_data = pd.read_csv('test.csv')
training_data = pd.read_csv('train.csv')

# Converting the predictive values to numerical

training_data['Survived'] = training_data['Survived'].astype('int')

train_labels = training_data['Survived']

train_data = training_data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

# Removing null variables

train_data['Sex'] = train_data.Sex.map({"male": 1, "female": 0})
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data.Embarked.map({"C": 0, "Q": 1, "S": 2})
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].median())

# Converting the training data to numpy array

X = np.asarray(train_data[['Pclass', 'Sex', 'Age', 'SibSp',
                           'Parch', 'Fare', 'Embarked']])
y = np.asarray(training_data['Survived'])

# Data preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

# Train test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Building the Model

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)

# Model Evaluation

yhat = LR.predict(X_test)
jaccard_score(y_test, yhat)
print(f'The Jaccard_score is {jaccard_score}')

test_data = testing_data
test_data = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Removing null variables

test_data['Sex'] = test_data.Sex.map({"male": 0, "female": 1})
test_data['Embarked'] = test_data.Embarked.map({"C": 0, "Q": 1, "S": 2})
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

test_x = np.asarray(test_data[['Pclass', 'Sex', 'Age', 'SibSp',
                               'Parch', 'Fare', 'Embarked']])

test_x = preprocessing.StandardScaler().fit(test_x).transform(test_x)

# Testing predictions
pred_t = LR.predict(test_x)
pred_t = pred_t.astype(int)

submission = pd.read_csv('gender_submission.csv')
submission['Survived'] = pred_t
submission.to_csv('gender_submission.csv', index=False)
