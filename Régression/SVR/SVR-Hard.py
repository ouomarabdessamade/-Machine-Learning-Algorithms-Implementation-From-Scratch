import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

'''Let us take same data used in linear regession but now will use the concept of marge ---> SVR'''

data = pd.read_csv('dataset-cars.csv')
X = np.array(data['speed'])
Y = np.array(data['dist'])
ep = 36
#----------- objectif function --------------------------------------------------------------------#

fct = lambda w : 0.5*(w[0]**2)

#------------------------------------------constraints----------------------------------------------#
cons = ()
for i,j in zip(X,Y):
    cons = cons + ({'type': 'ineq', 'fun': lambda w,i=i,j=j: ep - abs(j - (w[0]*i + w[1])) },)

#--------------------- maximisation de la marge <===> minization using SLSQP méthode ------------------------------------------------#
res = minimize(fct, np.array([1, 1]), method='SLSQP',jac='2-point',constraints=cons)
theta=res.x

#------------------ ploting --------------------------------------------------------------------------#
x1 = np.linspace(X.min()-1,X.max()+1)
x2 = theta[0]*x1 + theta[1]
x3 = theta[0]*x1 + theta[1] + ep
x4 = theta[0]*x1 + theta[1] - ep
plt.plot(x1,x2,c='r')
plt.plot(x1,x3,'--',c='b')
plt.plot(x1,x4,'--',c='b')
plt.scatter(X,Y)
plt.show()