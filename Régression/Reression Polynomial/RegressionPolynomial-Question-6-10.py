#--------la partie 2 de TP02--------------------#
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import sympy as sp

#-----importation de data------------------#
data = pd.read_csv("dataset.csv")
#----------le feautures--------------------#
x = np.array(data["temperature"])
x = x.reshape((x.shape[0],1))#redimention 

X = np.array(np.hstack((x,np.ones(x.shape))))

#---targets function------------------------#
Y = np.array(data["pressure"])
Y = Y.reshape(Y.shape[0], 1)

#6- Now, fit the data with polynomial regression model. Try different polynomial orders 
# (𝑄 = 2, 3, 4), and in each time compute the empirical error Ls(ℎpoly). 
# Pour Q = 2
X_Q2 = np.array(np.hstack((x**2,X)),dtype=float)
X_Q3 = np.array(np.hstack((x**3,X_Q2)),dtype=float)
X_Q4 = np.array(np.hstack((x**4,X_Q3)),dtype=float)

#--------------Mise à l'échelle des données-----------------------------------#
for i in range(X_Q2.shape[1]-1):
    X_Q2[:,i]=(X_Q2[:,i]-X_Q2[:,i].mean()/X_Q2[:,i].std())

for i in range(X_Q3.shape[1]-1):
    X_Q3[:,i]=(X_Q3[:,i]-X_Q3[:,i].mean()/X_Q3[:,i].std())

for i in range(X_Q4.shape[1]-1):
    X_Q4[:,i]=(X_Q4[:,i]-X_Q4[:,i].mean()/X_Q4[:,i].std())

Y = (Y-Y.mean())/Y.std()

np.random.seed(9)
w_2 = np.random.randn(3,1)
#print(w_2)
w_3 = np.random.randn(4,1)
w_4 = np.random.randn(5,1)
# w_3 = np.zeros((4,1))

def model(X,w):
    return X.dot(w)

#------------------loss function-------------------#
def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)

# print(cost_func(X_Q4,Y,w_4))
def grad(X,y,w):
    return (2/len(y)) * X.T.dot(model(X,w)-y)

#------fonction pour chrcher le pas optimale------------------#
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
#---------le gradient de descente----------------------------#
def gradient_descent(X, y, w, delta=0.3):
    compteur = 0
    #---initialisation de learning rate par l'optimale-------#
    learning_rate = alphaSearch(X,y,w)
    while np.linalg.norm(grad(X,y,w)) > delta :
        w = w - learning_rate * grad(X,y,w)
        learning_rate=alphaSearch(X,y,w)
        cost_history[compteur]=cost_func(X,y,w)
        compteur+=1
        print(np.linalg.norm(grad(X,y,w)))
        
    print('le nombre des itiration : ', compteur)
    return w,cost_history

#--------la visualisation des resultats---------------#
w_Q2_f,cost_history_2 = gradient_descent(X_Q2, Y, w_2)
# w_Q3_f,cost_history_3=gradient_descent(X_Q3,Y,w_3)
# w_Q4_f,cost_history_4=gradient_descent(X_Q4,Y,w_4)
plt.title('Beste sdeltaarteur')
plt.plot(x,model(X_Q2,w_Q2_f),c='green')
plt.scatter(x,Y)
plt.show()

#the best model is : model(X_Q2,w_Q2_f) #
#the best model between linear regression model and the best polynomial regression model is this last one #

# w_0_1 ,ch= gradient_descent(X_Q2, Y, w_2, delta=0.1)
# w_0_2 ,ch= gradient_descent(X,Y,w_2,delta=0.2)
# w_0_3 ,ch= gradient_descent(X,Y,w_2,delta=0.3)
# w_0_4 ,ch= gradient_descent(X,Y,w_2,delta=0.4)
# plt.plot(x,model(X_Q2,w_0_1),c='r')
# plt.scatter(x,Y)
# plt.show()


''' the best model for this data is the model with Q=2, when we choose learning rate constant, the algorithm may diverge,
so the best solution is with adaptave learning rate '''
