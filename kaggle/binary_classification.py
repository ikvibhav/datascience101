import numpy as np
import pandas as pd
import os
from kaggleUtilsIkv import perform_explore, perform_correlation_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

TRAIN_PATH = 'datasets/titanic/train.csv'
TEST_PATH = 'datasets/titanic/test.csv'


train_df = pd.read_csv(TRAIN_PATH).fillna(0)

# perform_explore(train_df)
# 
# feature_list = ["Survived","Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Embarked"]
# print(perform_correlation_matrix(train_df, feature_list))

Y = train_df['Survived']
feature_list = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Embarked"]

X = pd.get_dummies(train_df[feature_list])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = SVC(C=100, kernel='rbf', gamma='auto')
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print(f"Validation Accuracy: {accuracy:.2f}")

# print(f"Prediction")
# test_df = pd.read_csv(TEST_PATH).fillna(0)
# abcd = pd.get_dummies(test_df[feature_list])
# predictions = model.predict(abcd)
# output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
# output.to_csv('datasets/titanic/my_submission.csv', index=False)