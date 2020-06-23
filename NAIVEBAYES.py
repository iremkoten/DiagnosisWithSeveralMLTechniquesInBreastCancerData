import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, dataset.columns != 'diagnosis'].values
Y = dataset.iloc[:, 1].values

dataset.isnull().sum()
dataset.isna().sum()


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)

print(cm)
print('Accuracy' + str(accuracy_score(Y_test, Y_pred)))





