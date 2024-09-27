import numpy as np
import pandas as pd
import os
from kaggleUtilsIkv import perform_explore, perform_correlation_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = 'datasets/titanic/train.csv'
TEST_PATH = 'datasets/titanic/test.csv'

#Obtain the data
train_df = pd.read_csv(TRAIN_PATH).fillna(0)
Y = train_df['Survived']
feature_list = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = pd.get_dummies(train_df[feature_list], drop_first=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=0)

# Support Vector Classifier
model = SVC(C=100, kernel='rbf', gamma='auto')
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print(f"Validation Accuracy: {accuracy:.2f}")

# K Nearest Neighbors Classifier
clf = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3, weights="uniform"))])
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))

# Prediction
print(f"Prediction")
test_df = pd.read_csv(TEST_PATH).fillna(0)
test_df_vals = pd.get_dummies(test_df[feature_list], drop_first=True)
predictions = clf.predict(test_df_vals)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)