#first PCA then LDA
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

imputer = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=0)
imputer = imputer.fit(X[:, 1:]) 
X[:, 1:] = imputer.transform(X[:, 1:])

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
#print(principalComponents)


lda = LDA(n_components=2)
X_r2 = lda.fit(principalComponents, Y).transform(principalComponents)

colors = ['red', 'green']
target_names = ['M','B']
plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 0],alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of  dataset')
plt.show()