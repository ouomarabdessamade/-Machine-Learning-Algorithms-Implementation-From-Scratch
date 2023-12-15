#--------la premier parter de TP-----------#
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import numpy as np 
import pandas as pd
import sympy as sp

#-------------l'importation de data----------#
data = pd.read_csv("dataset.csv")
#---------la selection de feuteure-----------#
X = np.array(data["temperature"])
X = X.reshape(X)

#-------on ajoute 1 pour le bais-----------------#
X = np.array(np.hstack((X,np.ones(X.shape))))

#------------la target : lebel-------------------#
Y = np.array(data["pressure"])
Y = Y.reshape(Y.shape[0],1)

#1- Visualize the graph relating the pressure to the temperature. 
plt.scatter(data['temperature'],Y)

#2- What can you notice from the graph?
''' We can notice from the graph that the relationship between the inputs and outputs is polynomial'''

#3- Use linear regression model to capture the relationship between the inputs and the 
# outputs of the data. Then, return the empirical error 𝐿s(ℎreg).
 
np.random.seed(21)
w = np.random.randn(2,1)

def model(X,w):
    return X.dot(w)

def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)

def grad(X,y,w):
    return (2/len(y)) * X.T.dot(model(X,w)-y)

def alphaSearch(X,y,w):
  grd = grad(X,y,w)
  a = sp.Symbol("a")
  wk = w - a*grd
  f = str(cost_func(X,y,wk))
  def fun(a):
    fu = eval(f)
    return fu
  alpha = minimize_scalar(fun)
  return alpha.x

cost_history=np.zeros(1000)
#-----------le gradient de descente-------------------#
def gradient_descent(X,y,w,ep=0.01):
    global j
    j = 0
    learning_rate=alphaSearch(X,y,w)
    while np.linalg.norm(grad(X,y,w)) > ep:
        w = w - learning_rate * grad(X,y,w)
        learning_rate=alphaSearch(X,y,w)
        cost_history[j]=cost_func(X,y,w)
        j+=1
        # print(w)
    return w,cost_history
w_final,cost_history=gradient_descent(X,Y,w)

# empirical error 𝐿s(ℎreg)
print('Ls(hreg)= ',cost_func(X,Y,w_final))

#4- Visualize linear model in the constructed graph.

cost_history = cost_history[0:j]
plt.grid()
plt.plot(data['temperature'],model(X,w_final),c='r')

#5- Comment your result.

''' Le model lineaire est inconvenat pour capturer 
la relaion entre les données de cette data puisque l'erreur emperique est trop grand'''



plt.show()
