import preprocess_mpg as mpg
import preprocess_papers as papers

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

def normalize( ms ):
	for i in range(0,len(ms[0])):
		c = ms[:,i]
		max_ = np.max(c)
		min_ = np.min(c)
		for j in range(0,len(c)):
			c[j] = (c[j] - min_) / (max_ - min_)

def manhattan_distance(x, y):
	return (sum([ abs(a-b) for a, b in zip(x, y) ]))

def mod_standard_score(x):
	card = len(x)
	median = np.median(x)
	asd = sum([ abs(i-median) for i in x]) / card
	return [(i - median)/asd for i in x], mean, deviation
	
def KNN(x_train, y_train, k_neighbors, n_classes, x_test):
	predictions = []
	for ms in x_test:
		knns = []
		for i in range(0, len(x_train)):
			knns.append([manhattan_distance(ms, x_train[i]), i])
		knns = sorted(knns, key=lambda ind: ind[0])
		neighbors = [0]*n_classes
		for j in range(0, k_neighbors):
			neighbors[y_train[knns[j][1]]] += 1
		predictions.append(neighbors.index(max(neighbors)))
	return predictions

def metrics(X, Y, n_classes, norm):
	k = 10

	# Ten fold cross
	sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
	sss.get_n_splits(X, Y)

	ms = np.zeros((n_classes, n_classes))
	
	for train_index, test_index in sss.split(X, Y):
		print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]

		if(norm):
			normalize(x_train)
			normalize(x_test)

		predictions = KNN(x_train, y_train, k, n_classes, x_test)

		ms += confusion_matrix(y_test, predictions)

	print("10 Foldcross Validation \n", ms)
	print("Accuracy: ", sum(ms.diagonal()) / np.sum(ms))
	
	# Confusion matrix

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

	if(norm):
		normalize(X_train)
		normalize(X_test)
		
	predictions = KNN(X_train, Y_train, k, n_classes, X_test)
	
	cm = confusion_matrix(Y_test, predictions)

	print("\nConfusion Matrix \n", cm)
	print("Accuracy: ", sum(cm.diagonal()) / np.sum(cm))


	print("\nKappa Score\n", cohen_kappa_score(Y_test, predictions))


## Iris Dataset

#iris = datasets.load_iris()

#X = iris['data']
#Y = iris['target']
#n_classes = len(iris['target_names'])

## Cars Dataset
#X, Y, c = mpg.preprocess_mpg()
#n_classes = len(c)

#Papers Dataset
X, Y, c = papers.preprocess_papers()
n_classes = len(c)

metrics(X, Y, n_classes, False)


