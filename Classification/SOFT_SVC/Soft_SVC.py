import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#____________________________________________________DATA_______________________________________________________________
n=100
np.random.seed(21)
d=np.random.normal(0.3, 0.2, size=(n, 2))
l2=np.random.uniform(0, 1,n)>0.5
# y is in {-1, 1}
l = 2. * l2 - 1
d *= l[:, np.newaxis]
d -= d.mean(axis=0)
#________________________________________________Constarints____________________________________________________________
C=10
fun = lambda w: 0.5*(w[0]**2+w[1]**2)+C*sum([ w[m] for m in range(3,n+3)])
cons=list()
for i,j,k in zip(d,l,range(3,n+3)):
    f= {'type': 'ineq', 'fun': lambda w,i=i,j=j,k=k:    i[0]*j *w[0]+ i[1]*j *w[1]+ j *w[2] -1+w[k]}
    cons.append(f)
for i in range(3,n+3):
    f={'type': 'ineq', 'fun': lambda w,i=i:     w[i] }
    cons.append(f)
cons = tuple(cons)

res = minimize(fun, np.random.randn(1,n+3), method='SLSQP',jac='2-point',constraints=cons)
w=res.x
#__________________________________________________Ploting______________________________________________________________
x1 = np.arange(-1,8)
x2 = (-w[1]*x1-w[2])/w[0]
x3 = np.arange(-1,8)
x4 = (-w[1]*x3-w[2]-1)/w[0]
x5 = np.arange(-1,8)
x6 = (-w[1]*x5-w[2]+1)/w[0]
plt.scatter(d[:,0],d[:,1], c = l, s = 100)
plt.axis('scaled')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('C_SVC')
plt.grid(linestyle = '--',c = 'red')
plt.plot(x1, x2,c = 'green')
plt.plot(x3, x4,c = 'red')
plt.plot(x5, x6,c = 'red')
plt.show()