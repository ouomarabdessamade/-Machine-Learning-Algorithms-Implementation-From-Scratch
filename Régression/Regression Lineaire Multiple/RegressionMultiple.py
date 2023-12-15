from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

#---------------importation de data-------#
data_f = pd.read_excel('pop.xls',usecols='A:D',engine='xlrd')
data_c = pd.read_excel('pop.xls',usecols='E',engine='xlrd')
data_f['one'] = np.ones((len(data_f),1))
# print(data_f)
X = np.array(data_f)
Y = np.array(data_c)
# X = np.hstack((X,np.ones((X.shape[0],1))))
# X = np.concatenate((X,np.ones((X.shape[0],1))))

# mise à l'échelle des données
for i in range(4):
    data_f[f'X{i+1}']=(data_f[f'X{i+1}']-data_f[f'X{i+1}'].mean())/data_f[f'X{i+1}'].std()
    # print('***********',data_f[f'X{i+1}'])
Y = (Y-Y.mean())/Y.std()
# print(X)
# print(Y)
np.random.seed(3)
w = np.random.randn(5,1)
# w = np.ones((5,1))
def model(X,w):
    return X.dot(w)
# print(model(data_f,w))
 
def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)
# print(cost_func(data_f,Y,w))
def grad(X,y,w):
    return np.array((2/len(y)) * X.T.dot(model(X,w)-y))

# print(type(grad(data_f,Y,w)))
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
#----------le gradient de desc--------------------------------------#
def gradient_descent(X, y, w, delta=0.001):
    global j
    j = 0
    learning_rate=0.01  #alphaSearch(X,y,w)
    while np.linalg.norm(grad(X,y,w)) > delta :
        w = w - learning_rate*grad(X,y,w)
        # learning_rate=alphaSearch(X,y,w)
        cost_history[j]=cost_func(X,y,w)
        j+=1
        if j%100==0:
            print('iter = ',j,'----',cost_func(X,y,w),'----',np.linalg.norm(grad(X,y,w)))
    return w,cost_history

w_final,cost_history=gradient_descent(data_f,Y,w)
print('wf',w_final)
cost_history = cost_history[0:j]
plt.grid()
plt.plot(range(j),cost_history)
plt.title('Comportement de fonction de perd')
plt.show()