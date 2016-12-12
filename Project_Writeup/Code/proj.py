import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn.decomposition import PCA

#Loading the data set from competition
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

print train_set.head()

#printing the dataset 
print("Training set has {0[0]} rows and {0[1]} columns".format(train_set.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test_set.shape))

train_set = train_set.as_matrix()
test_set = test_set.as_matrix()

X = train_set[:,1:]
Y = train_set[:,0]

# to display first 9 images
X = X[0:9]
Y = Y[0:9]
for i in range(0, 9):
        plt.subplot(3,3,i+1)
        plt.imshow(X[i].reshape((28, 28)),cmap=cm.binary)
plt.show()

X_test = test_set

#Cross validation on data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

print("train data shape: %r, train target shape: %r"
      % (X_train.shape, y_train.shape))
print("test data shape: %r, test target shape: %r"
      % (X_test.shape, y_test.shape))

#Random Forest algorithm
print("\n Random Forest Classifier")
rfc = RandomForestClassifier(n_estimators = 2, random_state=0)
rfc.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(rfc.score(X_train, y_train)))
result = rfc.predict(X_test)
print(result)

#To save results into .csv file

df = pd.DataFrame(result)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('result.csv', header=True)

#Principal component analysis
#print("Principal component analysis")
#pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X_train)
#print(pca)

#Support vector classifier
print('Accuracy for Support vector classifier:')
classifier = SVC(kernel='rbf').fit(X_train,y_train).score(X_train,y_train)
print(classifier)
