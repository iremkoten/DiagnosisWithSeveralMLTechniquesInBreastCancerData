#LDA
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, dataset.columns != 'diagnosis'].values
Y = dataset.iloc[:, 1].values

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# lda = LDA(n_components=2)
# X_r2 = lda.fit(X, Y).transform(X)

lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, Y_train)
X_test = lda.transform(X_test)

colors = ['red', 'green']
target_names = ['M','B']
plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_train[Y_train == i, 0], X_train[Y_train == i, 0],alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of  dataset')

plt.show()

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(Y_test, y_pred)))
