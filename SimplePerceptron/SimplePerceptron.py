
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron classifier

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes ove the training dataset
    random_state: int
        Random number generator seed for random weight initialization

    Attributes
    ----------
    w_: 1d-array
        weights after fiiting
    errors_: list
        number of misclassifications (updates) in each epoch

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_featues]
            Training vectors, where n_samples is the number aof samples and
            n_features is the number of features.
        y: array-like, shape = [n_samples]
            Target values

        Returns
        -------
        self:object

        """
        rgen = np.random.RandomState(self.random_state)
        #bell shape centered at loc, std dev = scale. output is 1d array of size element
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        
        #iterate from 0 to n_iter (not included)
        for _ in range(self.n_iter):
            errors = 0
            #zip() does a one to one map as tuple
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] = np.add(self.w_[1:], (update * xi)).tolist()
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)        


    
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """ To visualize  the decisions boundaries
    """
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    #just find the limit and/reduce 1
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #np.arange(start, stop, step): create list of tupple from start to stop with step of step
    #np.meshgrid convert: create accessible arrays from list of tupple
    #(-1,-2) (-1,0) (-1,1)    xx1 = [-1 -1 -1][0 0 0 ][1 1 1]
    #(0,-2)(0,0)(0,1)      ==> 
    #(1,-2)(1,0)(1,1)         xx2 = [-2 -2 -2][0 0 0 ][1 1 1]
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    #ravel() xx1 = [-1 -1 -1 0 0 0 1 1 1]
    #        xx2 = [-2 -2 -2 0 0 0 1 1 1]
    #array() [[-1 -1 -1 0 0 0 1 1 1]
    #         [-2 -2 -2 0 0 0 1 1 1]] concatenation... sort of
    #.T , transpose from in this case a 2x9 to 9x2

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')



df = pd.read_csv('C:/Users/mimasaka/Documents/myOwnTraining/'
            'Python ML/Python-Machine-Learning-Second-Edition/Chapter02/iris.data',
            header = None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100, [0,2]].values


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length[cm]')
plt.ylabel('petal length[cm]')
plt.legend(loc='upper left')
plt.show()

