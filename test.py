import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from PNN import *

data = pd.read_csv('dataset/sonar.all-data')
rows = data.shape[0]  # gives number of row count
cols = data.shape[1]  # gives number of col count
cols = cols - 1

X = data.values[:, 0:cols] 
labels = data.values[:, cols]
Y = []
for i in range(len(labels)):
    if labels[i] == 'R':
        Y.append(0.0)
    else:
        Y.append(1.0)
Y = np.asarray(Y)

print(len(X))

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.1, random_state = 42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size = 0.2, random_state = 42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print(X_train1.shape,X_test1.shape)
print(X_train2.shape,X_test2.shape)
print(X_train3.shape,X_test3.shape)

clf = SGDClassifier(learning_rate = 'constant', eta0 = 0.1, shuffle = False)
clf.partial_fit(X_train1, y_train1,classes=np.unique(Y))
y_pred = clf.predict(X_test1)
accuracy = accuracy_score(y_test1,y_pred)*100
print(accuracy)

clf.partial_fit(X_train2, y_train2)
y_pred = clf.predict(X_test2)
accuracy = accuracy_score(y_test2,y_pred)*100
print(accuracy)
clf.partial_fit(X_train3, y_train3)
y_pred = clf.predict(X_test3)
accuracy = accuracy_score(y_test3,y_pred)*100
print(accuracy)







clf = MultinomialNB()
clf.partial_fit(X_train1, y_train1,classes=np.unique(Y))
y_pred = clf.predict(X_test1)
accuracy = accuracy_score(y_test1,y_pred)*100
print(accuracy)

clf.partial_fit(X_train2, y_train2)
y_pred = clf.predict(X_test2)
accuracy = accuracy_score(y_test2,y_pred)*100
print(accuracy)
clf.partial_fit(X_train3, y_train3)
y_pred = clf.predict(X_test3)
accuracy = accuracy_score(y_test3,y_pred)*100
print(accuracy)




bls = broadnet_enhmap(maptimes = 10, 
                       enhencetimes = 10,
                       traintimes = 10,
                       map_function = 'tanh',
                       enhence_function = 'sigmoid',
                       batchsize = 'auto', 
                       acc = 1,
                       mapstep = 10,
                       enhencestep = 10,
                       reg = 0.001)
#training bls with data
bls.fit(X_train1, y_train1)
predictlabel = bls.predict(X_test1)
accuracy = show_accuracy(predictlabel,y_test1)
print(accuracy)


bls.incremental_input(X_train1, X_train2, y_train2)
predictlabel = bls.predict(X_test2)
accuracy = show_accuracy(predictlabel,y_test2)
print(accuracy)

bls.incremental_input(np.row_stack((X_train1, X_train2)), X_train3, y_train3)
predictlabel = bls.predict(X_test3)
accuracy = show_accuracy(predictlabel,y_test3)
print(accuracy)







