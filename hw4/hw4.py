"""
================================
Nearest Neighbors Classification
================================
__author__ = 'akshara boppidi'
"""
import hw4a
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.model_selection import KFold
import time

# Creating a empty array for storing best values of k-neighbors
result = np.array([])
# To repeat the experiment 100 times
for i in range(100):
# read digits data & split it into X (training input) and y (target output)
    X,y,ytrue = hw4a.genDataSet(1000)
    X= X.reshape(len(X),1)
    bestk=[]
    kc=0
	# Varying the k values in given range
    for n_neighbors in range(1,900,2):
        kf = KFold(n_splits=10)
        kscore=[]
        k=0
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            # we create an instance of Neighbors Regressor and fit the data.
            clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
            clf.fit(X_train, y_train)
   
            kscore.append(clf.score(X_test,y_test))
    #print kscore[k]
            k=k+1

        bestk.append(sum(kscore)/len(kscore))
 # print bestk[kc]
        kc+=1

#given this array of E_outs in CV, we are finding the max, its 
# corresponding index, and its corresponding value of n_neighbors

##print bestk

   #Sorting the bestk values
   # newbestk=sorted(bestk,reverse=True)

    #To obtain the index of the bestk values
    inp = sorted(range(len(bestk)),key=bestk.__getitem__)
	#To get the three max values of the n_neighbors indexes.
    bestvalues = (inp[-1]*2+1,inp[-2]*2+1,inp[-3]*2+1)
    #print bestvalues
	#appending the bestvalues into result array.
    result = np.append(result,bestvalues)
    #print result

# To plot an histogram for the result array	
n, bins, patches = plt.hist(result, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

# To print three best index and n_neighbors values
#print newbestk[0]
#print(i[-1]*2)+1
#print newbestk[1]
#print(i[-2]*2)+1
#print newbestk[2]
#print(i[-3]*2)+1

#clf = neighbors.KNeighborsRegressor(304, weights='distance')
#clf.fit(X, y)

print "Eout:"
print clf.score(X,y)
print "Eouttrue"
print clf.score(X,ytrue)

#yhat = clf.predict(X)
#plt.plot(x,y,'.')
#plt.plot(x,ytrue,'rx')
#plt.plot(x,yhat,'g+')
#plt.show()