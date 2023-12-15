import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import pandas as pd
import math as mpt


plt.style.use('ggplot')
#------------------------------------------------------data-------------------------------------------------------------
data = pd.read_csv('data_crcle.csv')

#---------(1)------------------------------------------------------------------#
data = data.sample(n=150) 

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
VC_dim = len(X_train[0])+1 
print("le vc dimension:", VC_dim)

def born_gene(epcilon, delta, vc):
    return (1/epcilon)*(4*mpt.log2(2/delta)+8*vc*mpt.log2(13/epcilon))
#----la borne de generalisation-----------------------------------------------#
born_gen = born_gene(0.03, delta, VC_dim)
print("la borne de generalisation : ",born_gen)
        
#---la visualisation des donnees d'entrainnement-------------------------------#
#---------------------data plot before transformation--------------------------#
fig = plt.figure()

plt.title("data plot before transformation")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, linewidth=1)
plt.draw()
plt.pause(2)
fig.clear()

#---------------------------------------data plot after transformation-------------------------------------------------
plt.title("data plot after transformation")
d =np.asarray((X_train[:,0]**2,X_train[:,1]**2)).T
plt.axis([np.min(d[:,0])-0.1, np.max(d[:,0])+0.1, np.min(d[:,1])-0.1, np.max(d[:,1])+0.1])
plt.scatter(d[:,0], d[:,1], c=y_train, linewidth=1)
plt.draw()
plt.pause(1)
fig.clear()
w=np.array([10,-13,18])

#-----------------------------------------------------loss function ----------------------------------------------------
def loss(w,d,l):
    S=0
    m = len(d)
    for i,j in zip(d,l):
        if np.dot(w, np.array([1,i[0], i[1]]))*j < 0:
            S = S + 1
    return S/m

def gradloss(w,d,l):
    S=0
    m = len(d)
    for i,j in zip(d,l):
        S = S - 2*np.dot(j-np.dot(w, np.array([1,i[0], i[1]])),np.array([1,i[0], i[1]]))

    return S/m
#-----------------------------------------------------plot--------- ----------------------------------------------------
def plt_show(w,d,l,ti):
    x = np.array([min(d[:,0])-1,max(d[:,0])+1])
    y = (-w[1] * x -w[0]) / w[2]
    plt.title("Transformation")
    plt.axis([np.min(d[:,0])-0.1, np.max(d[:,0])+0.1, np.min(d[:,1])-0.1, np.max(d[:,1])+0.1])
    plt.scatter(d[:,0], d[:,1], c =l, linewidth=1)
    plt.plot(x, y, color="green")
    plt.draw()
    plt.pause(ti)
    fig.clear()
plt_show(w,d,y_train,0.1)

#-------------------------------------------------Perceptron on transformed data----------------------------------------
ls=loss(w,d, y_train)
gls=gradloss(w,d, y_train)
while la.norm(gls)>0.7:
    for i,j in zip(d, y_train):
        ei = (j - np.dot(w, np.array([1, i[0], i[1]])))
        if ei!= 0:
            w = w + 0.2 * ei * np.array([1, i[0], i[1]])
            plt_show(w,d, y_train, 0.1)
    ls=loss(w,d, y_train)
    gls=gradloss(w,d, y_train)
    print(la.norm(gls))
plt_show(w,d, y_train, 2)


#-----------------------------------------------------Final plot--------------------------------------------------------
plt.title("Resultat final : Adalin trasformation")
x=np.arange(np.min(X_train[:,0])-1,np.max(X_train[:,0])+1,0.01)
y=np.arange(np.min(X_train[:,1])-1,np.max(X_train[:,1])+1,0.01)
x,y = np.meshgrid(x,y)
z= np.sign(w[0]+w[1]*x**2+w[2]*y**2)
plt.scatter(X_train[:,0],X_train[:,1], c = y_train, linewidth=1)
plt.contourf(x, y, z,1, colors = ['darkblue','yellow'], alpha = .1)
plt.contour(x, y, z, cmap = 'viridis')
plt.show()






