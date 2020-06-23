#Preprocessing-PCA
import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset[['diagnosis']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['M', 'B']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['diagnosis'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
