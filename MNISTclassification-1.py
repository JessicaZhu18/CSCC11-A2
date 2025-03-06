"""
CSCC11 A2 Programming KNN and RF
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# At most one of the Q2 and Q3 can be set to True, when both False, we test Q1
Q2 = False
Q3 = False
PR_NOISE = (0.20, 0.10, 0.03, 0.07, 0.06, 0.19, 0.10, 0.1, 0.00, 0.15)


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

show_digit(X[5],y[5])

test_percentage = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size= test_percentage, random_state=12) # split the data to have 20% test set and 80% training set

if Q2 or Q3: # Q2 and Q3 have noisy test set
    X_test =  X_test + np.random.uniform(0,255,size = X_test.shape)

X_train = X_train / np.max(X_train) # TODO Normalize X_train based on the maximum value of X_train
X_test = X_test / np.max(X_test)  # TODO Normalize X_test  based on the maximum value of X_train

if Q3: # Q3 has noisy training set
    Noise = [str(index) for index in np.random.choice(10,20000, p = PR_NOISE)]
    y_train[:20000] = Noise

R_train_KNN = []
R_test_KNN = []
Neighbours = np.arange(1,100,5)
for k in Neighbours:
  model = KNeighborsClassifier(k)  # TODO define KNN model
  model.fit(X_train, y_train)                     # TODO fit the data
  y_res_train = model.predict(X_train)   # TODO Output for training set
  y_res_test = model.predict(X_test)    # TODO Output for test set
  R_train_KNN.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_KNN.append(sklearn.metrics.accuracy_score(y_test, y_res_test))

def show_multiple_digits(X, y, num_samples = 5):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        X_reshape = X[i].reshape(28, 28)
        plt.imshow(X_reshape, 'gray')
        plt.title(f'Label is: {y[i]}')
    plt.show()

show_multiple_digits(X[:5], y[:5])

R_train_RF = []
R_test_RF = []
Min_sample = np.arange(5,51,5)
print('2')
for s in Min_sample:
  print('1')
  model_RF = RandomForestClassifier(min_samples_leaf=s) # TODO define Random forest model
  model_RF.fit(X_train, y_train)                      # TODO fit the data
  y_res_train = model_RF.predict(X_train)    # TODO Output for train set
  y_res_test = model_RF.predict(X_test)    # TODO Output for test set
  R_train_RF.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_RF.append(sklearn.metrics.accuracy_score(y_test, y_res_test))

# Plotting KNN
plt.figure(figsize=(10, 6))
plt.plot(Neighbours, R_train_KNN, label="Training Accuracy")
plt.plot(Neighbours, R_test_KNN, label="Testing Accuracy")

# Adding labels and title
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy Score")
plt.title("Training and Testing Accuracy for Different k values in KNN")
plt.show()


# Plotting Random Forest
plt.figure(figsize=(10, 6))
plt.plot(Min_sample, R_train_RF, label="Training Accuracy")
plt.plot(Min_sample, R_test_RF, label="Testing Accuracy")

# Adding labels and title
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy Score")
plt.title("Training and Testing Accuracy for Different min_samples_leaf in RF")
plt.show()