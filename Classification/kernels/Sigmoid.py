import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import datasets
plt.style.use('ggplot')
#____________________________________________________DATA_______________________________________________________________
np.random.seed(20)
n=20


d,l = datasets.make_classification( n_samples=n,n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
l[l==0]=-1
s=1
def kernel(x,y,s):
    return np.tanh(s*(x[0]*y[0]+x[1]*y[1]))


#________________________________________________maximisation de la marge_______________________________________________
C=10
fun = lambda w: 1/(sum([ w[i] for i in range(0,n)])-1/2*sum([ sum([ w[i]*w[j]*l[i]*l[j]*kernel(d[j],d[i],s) for j in range(0,n)]) for i in range(0,n)]))
cons=list()
f={'type': 'eq', 'fun': lambda w:     sum([w[i]*l[i] for i in range(0,n)]) }
cons.append(f)
for i in range(0,n):
    f={'type': 'ineq', 'fun': lambda w,i=i:     w[i] }
    cons.append(f)

cons = tuple(cons)

res = minimize(fun, np.random.randn(1,n), method='SLSQP',jac='2-point',constraints=cons)
w=res.x

B=l[0]-sum([w[i]*l[i]*kernel(d[0],d[i],s) for i in range(0,n)])
#__________________________________________________Ploting______________________________________________________________

x=np.arange(np.min(d[:,0])-1,np.max(d[:,0])+1,0.01)
y=np.arange(np.min(d[:,1])-1,np.max(d[:,1])+1,0.01)
x,y = np.meshgrid(x,y)
z= np.sign(sum([w[i]*l[i]*kernel([x,y],d[i],s) for i  in range(0,n)])+B)
plt.scatter(d[:,0],d[:,1], c = l, s = 100)
plt.contourf(x, y, z,1, colors = ['darkblue','yellow'], alpha = .1)
plt.contour(x, y, z, cmap = 'viridis')
plt.show()


