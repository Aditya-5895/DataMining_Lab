import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
  # print(x1,x2)
  # print(type(x1),type(x2))
  return np.sqrt(np.sum((x1 - x2) ** 2))

class Disease_Classifier():
    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        l = [ ]
        for i in y_pred:
          l.append([i[0][0],i[1][0],i[0][1],i[1][1]])
        return pd.DataFrame(np.array(l),columns=['Most Probable disease','Dist','Second most probable disease','Dist'])

    def _predict(self, x):

        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        dist = np.sort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        return [k_neighbor_labels,dist]

    def accuracy(self,x,y_true):
        y_pred = self.predict(x)
        y_pred = y_pred['Most Probable disease']
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return f'The model have accuracy of {accuracy*100} % on this data.'


# Model Implementation

#X = np.array(df.drop('Disease',axis=1))
#y = np.array(df['Disease'])
#model = Disease_Classifier(2)
#model.fit(X,y)
#y_pred = model.predict(x,y)
#model.sccuracy(x,y)