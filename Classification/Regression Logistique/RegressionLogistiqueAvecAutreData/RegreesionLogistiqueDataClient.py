import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


d=pd.read_csv('clients.csv')


X=list()
for i,j in zip(np.array(d['Age']),np.array(d['EstimatedSalary'])) :
	X.append([1,i,j])
X = np.array(X)
y = np.array(d['Purchased'])

# Feature Scaling la standarisation
sc = StandardScaler()
x = sc.fit_transform(X)



#initialisation de w
w = [0,0,0]



def lossfunction(x, y, w):
    m = len(x)
    som =0
    for i in range(m):
         som += np.log(1 + np.exp(-y[i]*(np.dot(w, x[i]))))
        
    return som/m


    

def gradient(x, y, w):
    som = 0
    m = len(x)
    for i in range(m):
        t1 = (-y[i] * np.exp(-y[i] * np.dot(w, x[i])))
        t2 = 1/(1 + np.exp(-y[i] * np.dot(w, x[i])))
        
        som +=  (t1/t2) * x[i]
    return som/m
    


#l'algorithme
def regressionLogistic(x, y, w, learning_rate= 0.7, delta = 0.0001):
    cptr = 0
    gradloss = gradient(x,y, w)
    while np.linalg.norm(gradloss) > delta :
        w = w - learning_rate * gradloss
        #print(w)
        gradloss = gradient( x, y, w)
        cptr += 1
        print(np.linalg.norm(gradloss))
        
    loss = lossfunction(x, y, w)
    return w, loss, cptr
    
    
#regressionLogistic(data, y, w) 
theta, loss, compteur = regressionLogistic(x, y, w) 
print('lossfunction = ', loss, 'nbr d iteration = ', compteur)

#plot_decision_boundary(x, w)


plt.title('Regression Logistique')
plt.scatter(x[:,1], x[:,2], s=40, c = y)
if ( theta[1]!= 0 ):
    t = np.linspace(-2, 2, 2)
    y = (-theta[1]/theta[2])*t - theta[0]/theta[2]
plt.plot(t, y, color='green')
plt.show()





