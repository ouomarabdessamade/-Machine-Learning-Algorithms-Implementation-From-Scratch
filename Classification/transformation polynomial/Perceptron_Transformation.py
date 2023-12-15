import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import math as mpt


#----------------data :
data = pd.read_csv('DataTransformationPly.csv')

#---------(1)------------------------------------------------------------------#
data = data.sample(n=600) 

#------------------------------------2-----------------------------------------#
m = len(data) #le nbr des exemple dans la dataset apres la generation 

#la taille minimale de l'échantillon égale 80% de la taille du dataset initiale.
tail_min_data = 0.8 * m

#-------on Fixe epsilon à 0.01 et on calcule δ qui donne tail_min_data---------#
#on a la relation qui relier la taile min de data,delta et epsilon est donnees par : 
#  tail_min_data ≃ Ln(2/delta)/epsilon

epsilon = 0.01 #on fixe epsilon
#la calcule de delta
delta = 2/ mpt.exp(epsilon * tail_min_data)
print("delta:", delta)

#-----3)Diviser les donn´ees en 80% pour train-set et 20% pour test-set--------#
test_data = data.sample(int(len(data)*0.2), random_state=0) # 20%
train_data = data.drop(test_data.index, axis=0) # 80%
  
#----definir data de traing----------------------------------------------------#
X_train = list()
for i,j in zip(np.array(train_data['X1']),np.array(train_data['X2'])) :
    X_train.append([i,j])
X_train = np.array(X_train)
y_train = np.array(train_data['y'])
#en remplace les 0 par -1 dans y_train
for i in range (len(y_train)):
    if y_train[i]==0:
        y_train[i]=-1
#print(y_train)


#----definir data de testing
X_test = list()
for i,j in zip(np.array(test_data['X1']),np.array(test_data['X2'])) :
    X_test.append([i,j])
X_test = np.array(X_test)
y_test = np.array(test_data['y'])
#en remplace les 0 par -1 dans y_test
for i in range (len(y_test)):
    if y_test[i]==0:
        y_test[i]=-1

#--4)la calcule de VC dimension et la Borne de généralisation------------------#
VC_dim = len(X_train[0])+1 #vc_dim est le nbr des features + 1
print("le vc dimension:", VC_dim)

def born_gene(epcilon, delta, vc):
    return (1/epcilon)*(4*mpt.log2(2/delta)+8*vc*mpt.log2(13/epcilon))
#----la borne de generalisation-----------------------------------------------#
born_gen = born_gene(0.03, delta, VC_dim)
print("la borne de generalisation : ",born_gen)
        
#---la visualisation des donnees d'entrainnement
plt.title('la visualisation de data ')
plt.scatter(X_train[:,[0]], X_train[:,[1]], s=40, c = y_train)
plt.show()


#_______________________________________________________________Plotting and generating data___________________________________________________________________________________
def plt_show():
    plt.draw()
    plt.pause(0.5)
    fig.clear()
    
fig = plt.figure()
X,y=datasets.make_circles(n_samples=300,  shuffle=True, noise=0.05, random_state=0, factor=0.5)
y = y > 0
x_i = X[:,0]
y_i = X[:,1]
plt.scatter(x_i,y_i,c = y, s = 100,cmap = 'spring')
plt.grid(linestyle = '--',c = 'red')
plt_show()
X = X**2
x_i = X[:,0]
y_i = X[:,1]
plt.scatter(x_i,y_i,c = y, s = 100,cmap = 'spring')
plt.grid(linestyle = '--',c = 'red')
plt_show()
roor = np.array([min(X[:,0]), max(X[:,0]),min(X[:,1]), max(X[:,1])])

def plot_decision_boundary(X, theta):
    x1 = np.array([min(X[:,0]), max(X[:,0])])
    x2 = (-theta[1]*x1-theta[0])/theta[2]
    plt.scatter(X[:,0],X[:,1], c = y, s = 100,cmap = 'spring')
    plt.axis((roor[0],roor[1],roor[2],roor[3]))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Perceptron\'s world')
    plt.grid(linestyle = '--',c = 'red')
    plt.plot(x1, x2,c = 'green')
    plt_show()
#_______________________________________________________________PERCEPTRON___________________________________________________________________________________
def L_s(X,w):
    global data_and_labels
    global data
    data_and_labels = list()
    data = list()
    x_i = X[:,0]
    y_i = X[:,1]
    L_s = 0
    for i in range(len(X)):
        if y[i]:
            data_and_labels.append([np.array([1,x_i[i],y_i[i]]),1])
        else:
            data_and_labels.append([np.array([1,x_i[i],y_i[i]]),-1])
        data.append([x_i[i],y_i[i]])
    for i in range(len(X)):
        L_s += int(np.where(np.dot(w,data_and_labels[i][0].T)*data_and_labels[i][1]<0,1,0))
    return float(L_s)/len(X)

w = np.array([0.8,0.5,0.6])
def perceptron(X,w):
    j=0
    while(L_s(X,w)!=0):
        for i in range(len(X)):
            if np.dot(w,data_and_labels[i][0].T)*data_and_labels[i][1]<0:
                #print(data_and_labels[i][0]*data_and_labels[i][1])
                w+=data_and_labels[i][0]*data_and_labels[i][1]
                #plot_decision_boundary(np.array(data),w)
                j+=1
    return w
#__________________________________________________________________________________________________________________________________________________
w = perceptron(X,w)
X,y = datasets.make_circles(n_samples=300,  shuffle=True, noise=0.05, random_state=0, factor=0.5)
print('w* = ',w)
y = y > 0
x_i = X[:,0]
y_i = X[:,1]
plt.scatter(x_i,y_i,c = y, s = 100,cmap = 'spring')
plt.grid(linestyle = '--',c = 'green')
circle = plt.Circle((0,0),w[0],color = 'green',fill = False)
fig = plt.gcf()
ax = fig.gca()
ax.add_patch(circle)
plt.show()


