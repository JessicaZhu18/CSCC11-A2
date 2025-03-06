"""
Created on Sat Feb  3 12:54:33 2024

@author: mehdy
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.asarray(X)
Y = np.asarray(y)
print('X shape', X.shape)
print('Y shape', y.shape)

def show_digit(x_, y_):
    X_reshape = x_.reshape(28, 28) # reshape it to have 28*28
    plt.imshow(X_reshape, 'gray')
    plt.title('Label is: ' + y_)
    plt.show()

show_digit(X[0],y[0])

test_percentage = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size= test_percentage, random_state=12) # split the data to have 20% test set and 80% training set
x_max = np.max(X_train)
X_train = (X_train)/(x_max)  # Normalize X_train based on the maximum value of X_train
X_test = (X_test)/(x_max) #  Normalize X_test based on the maximum value of X_train

from sklearn.neighbors import KNeighborsClassifier
R_train_KNN = []
R_test_KNN = []
Neighbours = np.arange(1,100,5)
i = 0
for k in Neighbours:
  model = KNeighborsClassifier(n_neighbors=k, weights="distance")  # define KNN model
  model.fit(X_train, y_train)  # fit the data
  y_res_train = model.predict(X_train) # Output for training set
  y_res_test = model.predict(X_test) # Output for test set
  R_train_KNN.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_KNN.append(sklearn.metrics.accuracy_score(y_test, y_res_test))
  print("Donee #" + str(i+1) + ", k="+str(k)+", "+ str(R_train_KNN[i]) +", "+ str(R_test_KNN[i]))
  i = i + 1

from sklearn.ensemble import RandomForestClassifier
R_train_RF = []
R_test_RF = []
Min_sample = np.arange(5,51,5)
i = 0
for s in Min_sample:
  model_RF = RandomForestClassifier(min_samples_leaf=s) # define Random forest model
  model_RF.fit(X_train, y_train) # fit the data
  y_res_train = model_RF.predict(X_train) # Output for train
  y_res_test = model_RF.predict(X_test) 
  R_train_RF.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_RF.append(sklearn.metrics.accuracy_score(y_test, y_res_test))
  print("Donee #" + str(i+1) + ", s="+str(s)+", " + str(R_train_RF[i]) +", "+ str(R_test_RF[i]))
  i = i + 1



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Neighbours, R_train_KNN, label="Training Accuracy", marker='o', linestyle='-')
plt.plot(Neighbours, R_test_KNN, label="Testing Accuracy", marker='o', linestyle='-')

# Adding labels and title
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy Score")
plt.title("Training and Testing Accuracy for Different k values in KNN")
plt.legend()
plt.grid(True)
plt.show()


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Min_sample, R_train_RF, label="Training Accuracy", marker='o', linestyle='-')
plt.plot(Min_sample, R_test_RF, label="Testing Accuracy", marker='o', linestyle='-')

# Adding labels and title
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy Score")
plt.title("Training and Testing Accuracy for Different min_samples_leaf values in Random Forest")
plt.legend()
plt.grid(True)
plt.show()