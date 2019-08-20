import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from matplotlib.colors import ListedColormap
from SupervisedLearning.SimplePerceptron import Perceptron as Perceptron
from SupervisedLearning.SimplePerceptron import plot_decision_regions as plot_decision_regions
from SupervisedLearning.AdalineGD import AdalineGD as AdalineGD
from SupervisedLearning.AdalineSGD import AdalineSGD as AdalineSGD


df = pd.read_csv(sys.argv[1],
            header = None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100, [0,2]].values

#Ppn
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()

#GD
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
adal = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_) + 1, marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squarred-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

#SGD
df = pd.read_csv('C:/Users/mimasaka/Documents/myOwnTraining/'
                'Python ML/Python-Machine-Learning-Second-Edition/Chapter02/iris.data',
                header = None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100, [0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Decent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()