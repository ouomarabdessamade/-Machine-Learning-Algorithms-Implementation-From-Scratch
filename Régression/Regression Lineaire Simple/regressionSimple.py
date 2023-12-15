import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#---importation de data------#
S = pd.read_csv('cars.csv')


#--------fonction cout-------------------#
def cost_function(x, y, w):
    m = len(y)
    som = 0
    for i in range(m):
        wx = np.dot(w, np.array([x[i][0], 1]))
        som += (wx - y[i])**2
        
    return(1/m)*som

#print(cost_function(X, y_data, theta))


#--gradient et descente de Gradient
def gradient(x, y, w):
    m = len(y)
    som = 0
    for i in range(len(y)):
        wx = np.dot(w , np.array([x[i][0], x[i][1]]))
        yx = y[i]* (np.array([x[i][0], x[i][1]]))
        som += (np.dot(wx, np.array([x[i][0], x[i][1]]))- yx)
    result = (2/m)* som
    #print(np.linalg.norm(result))
    return result

#----------la fonction de regression--------------#
def regressionLineaire(data, w, delta =0.5, alpha = 0.003):
    x = data.iloc[:,:-1]
    x = np.array(x)
    c = 0
    #------Ajouter un col des 1-------------------#
    X = np.hstack((x, np.ones(x.shape)))
    #print(X)
    Y = data.iloc[:,-1:]
    Y = np.array(Y)
    #print(Y)
    #---------la calcule de grad(w)---------------#
    gradLw = gradient(X, Y, w)
    wk = w
    while np.linalg.norm(gradLw) > delta :
        wk_plus1 = wk - alpha * gradient(X, Y, wk)
        wk = wk_plus1
        gradLw = gradient(X, Y, wk)
        print(np.linalg.norm(gradLw))
        c+=1
        #print(c)

    #end while
    print('fin ',wk)
    print('c = ', c)
    return wk
    
 

#initialisation de w
#w = np.random.randn(2, 1)
w = [1, 1]
#----teste de la fonction------#
theta = regressionLineaire(S, w)

#----------la visualisation de data-----------------------------#
plt.title('la regression lineaire ')
plt.grid(False)#plot a grid
plt.xlim( 0, 40)
plt.ylim(-20, 130)
plt.xlabel('speed')
plt.ylabel('dist')

for i in range(len(S)):
    plt.scatter(S['speed'], S['dist'], alpha=1, color='blue', marker='o')


t = np.linspace(0, 140, 10)
y = (theta[0] * t + theta[1])
plt.plot(t, y, color="green")
 
plt.show()






















