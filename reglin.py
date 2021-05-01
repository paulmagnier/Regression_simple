import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

#Constantes
M = 90
alpha = 0.01
N = 400

############
# LINEAIRE #
############

#DataSet
X, MatY = make_regression(n_samples=M, n_features=1, noise=15)
MatY = MatY.reshape(M, 1)

#Representation matricielle
un = np.ones((M, 1))
MatX = np.hstack((X, un))
MatTheta = np.random.randn(2, 1)

#Affichage
def affiche():
    plt.scatter(X, MatY, marker='.')
    a, b = MatTheta[0, 0], MatTheta[1, 0]
    plt.plot(X, a*X + b, c='r')
    plt.show()
affiche()

#Fonction cout
def J(Theta):
    return (1/(2*M)) * np.sum((MatX.dot(Theta) - MatY)**2)

#Gradient
def grad(Theta):
    return (1/M) * (MatX.T).dot(MatX.dot(Theta) - MatY)

def descente(Theta):
    return Theta - alpha*grad(Theta)

#Regression lineaire
l = []
for i in range(N):
    l.append(J(MatTheta))
    MatTheta = descente(MatTheta)

#Historique
def historique():
    plt.plot([i for i in range(N)], l)
    plt.show()

#affiche()
#historique()

###############
# POLYNOMIALE #
###############

#DataSet
X, MatY = make_regression(n_samples=M, n_features=1, noise=15)
MatY = MatY.reshape(M, 1)

#Representation matricielle
un = np.ones((M, 1))
MatX = np.hstack((X**2, X, un))
MatTheta = np.random.randn(3, 1)

#Affichage
def affiche():
    plt.scatter(X, MatY, marker='.')
    a, b, c = MatTheta[0, 0], MatTheta[1, 0], MatTheta[2, 0]
    min = X.min()
    max = X.max()
    X2 = np.linspace(min, max, 100)
    res = a*(X2**2) + b*X2 + c
    plt.plot(X2, res, c='r')
    plt.show()
affiche()

#Fonction cout
def J(Theta):
    return (1/(2*M)) * np.sum((MatX.dot(Theta) - MatY)**2)

#Gradient
def grad(Theta):
    return (1/M) * (MatX.T).dot(MatX.dot(Theta) - MatY)

def descente(Theta):
    return Theta - alpha*grad(Theta)

#Regression lineaire
l = []
for i in range(N):
    l.append(J(MatTheta))
    MatTheta = descente(MatTheta)

#Historique
def historique():
    plt.plot([i for i in range(N)], l)
    plt.show()

#affiche()
historique()