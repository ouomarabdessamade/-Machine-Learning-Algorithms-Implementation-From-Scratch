import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
plt.style.use('ggplot')


#_________________________________________________DATA__________________________________________________________________#
fig = plt.figure()
df=pd.read_csv('dataset-cars.csv')
x=np.array(df['speed'])
y=np.array(df['dist'])
ep=30
C=10
n=len(x)

#________________________________________________maximisation de la marge_______________________________________________#
fun = lambda w: 0.5 * (w[0] ** 2) + C * sum([w[m] for m in range(2, n + 2)]) + C * sum([w[m] for m in range(n + 2, n + n + 2)])
cons = list()
for i, j, k, l in zip(x, y, range(2, n + 2), range(n + 2, 2 * n + 2)):
    t = {'type': 'ineq', 'fun': lambda w, i=i, j=j, k=k: -j + (i * w[0] + w[1]) + ep + w[k]}
    cons.append(t)
    t={'type': 'ineq', 'fun': lambda w, k=k:     w[k] }
    cons.append(t)
    t = {'type': 'ineq', 'fun': lambda w, i=i, j=j, l=l: j - (i * w[0] + w[1]) + ep + w[l]}
    cons.append(t)
    t = {'type': 'ineq', 'fun': lambda w, l=l: w[l]}
    cons.append(t)

res = minimize(fun, np.random.randn(1,2*n+2), method='SLSQP',jac='2-point',constraints=cons)
theta=res.x

#________________________________________________ploting________________________________________________________________#
x1 = np.arange(np.min(x),np.max(x))
x2 = (theta[0]*x1+theta[1])
x3 = np.arange(np.min(x),np.max(x))
x4 = (theta[0]*x3+theta[1]-ep)
x5 = np.arange(np.min(x),np.max(x))
x6 = (theta[0]*x5+theta[1]+ep)
plt.scatter(x,y)
#plt.axis('scaled')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Hard_SVC')
plt.plot(x1, x2,c = 'green')
plt.plot(x3, x4,c = 'blue')
plt.plot(x5, x6,c = 'blue')
plt.show()
