import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import math as mpt

#----------------data :
data = pd.read_csv('clients.csv')

#---------(1)------------------------------------------------------------------#
data = data.sample(n=200)#selection dun echantion de taille 200

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
test_data = data.sample(int(len(data)*0.2), random_state=0) # 20% pour testing
train_data = data.drop(test_data.index, axis=0) # 80% pour training

#----definir data de traing--------------------
X_train = list()
for i,j in zip(np.array(train_data['Age']),np.array(train_data['EstimatedSalary'])) :
    X_train.append([i,j])
X_train = np.array(X_train)
y_train = np.array(train_data['Purchased'])
#en remplace les 0 par -1 dans y_train
for i in range (len(y_train)):
    if y_train[i]==0:
        y_train[i]=-1
        
#----definir data de testing
X_test = list()
for i,j in zip(np.array(test_data['Age']),np.array(test_data['EstimatedSalary'])) :
    X_test.append([i,j])
X_test = np.array(X_test)
y_test = np.array(test_data['Purchased'])
#en remplace les 0 par -1 dans y_test
for i in range (len(y_test)):
    if y_test[i] == 0:
        y_test[i]=-1

#--4)la calcule de VC dimension et la Borne de généralisation------------------#
VC_dim = len(X_train[0])+1 #vc_dim est le nbr des features + 1
print("le vc dimension:", VC_dim)

def born_gene(epcilon, delta, vc):
    return (1/epcilon)*(4*mpt.log2(2/delta)+8*vc*mpt.log2(13/epcilon))
#----la borne de generalisation-----------------------------------------------#
born_gen = born_gene(0.03, delta, VC_dim)
print("la borne de generalisation : ",born_gen)

# la standarisation de X_train
sc1 = StandardScaler()
X_train = sc1.fit_transform(X_train)

# la standarisation de  X_test
sc2 = StandardScaler()
X_test = sc2.fit_transform(X_test)
#print(X_train)

#---la visualisation des donnees d'entrainnement
plt.title('la visualisation de data de training')
plt.scatter(X_train[:,[0]], X_train[:,[1]], s=40, c = y_train)
plt.show()



#-----------------------------loss function-----------------------------------#
def calculdelsw(w, x, y):
    som = 0
    for i in range(len(x)):
        som+= (y[i] - (np.dot(np.transpose(w), np.array([x[i][0], x[i][1], 1]))))**2
    return som/len(x)

#-----------------------------le gradient-------------------------------------#
def gradient(w , x, y):
    som = 0
    for i in range(len(x)):
        ei = (y[i] - np.dot(np.transpose(w), np.array([x[i][0], x[i][1], 1])))
        #print("ei = ", ei)
        som += ei * (np.array([x[i][0], x[i][1], 1]))
    result = (-2*som)/len(x)
    return np.linalg.norm(result)   



#-----------------------Algoritme de Adalin-----------------------------------#
def Adalin2D(X, y):
    plt.ion()
    w = [0, 0, 0]
    z = np.linspace(X.min(), X.max(), 2)
    delta = 0.01
    Alpha = 0.0001
    t = 0
    gradLosw = 1
    grad =  gradient(w , X, y)
    #print("gradient = ", grad)
    while grad > delta :
        for i in range (len(X)):
            gradLosw = (y[i] - np.dot(np.transpose(w), np.array([X[i][0], X[i][1], 1])))
            if(gradLosw != 0):
                w = w +  Alpha * gradLosw * (np.array([X[i][0], X[i][1], 1]))
                t+=1
                #--------------------plot result chaque 10 itiration----------------------#
                if t % 1000 == 0 :
                    #-------Plot de graphe------------------------------------------------#
                    plt.clf() #--------clear figure--------#
                    plt.grid(False)#--------plot a grid--------#
                    plt.xlim(-2.5, 2.5)
                    plt.ylim(-2, 3)
                    
                    plt.title('iteration numero :'+str(t))
                    plt.scatter(X[:,[0]], X[:,[1]], s=40, c = y)
            
                    if ( w[1]!= 0 ):
                        k = (-w[0]/w[1])* z - w[2]/w[1]
                        plt.plot(z , k, color='green')
                        
                    plt.show()
                    plt.pause(0.05)
                #end if 
            #end if
        #end for
        grad =  gradient(w , X, y) 
        #print(grad)
    return w, t   

#-----------------Visualiser le séeparateur-----------------------------------#

#---------------------------resultat de l'lgorithme---------------------------#
w, t = Adalin2D(X_train, y_train)

#-------Calculer l’erreur d’approximation et l’erreur de généralisation-------#
erreur_d_approximation = calculdelsw(w, X_train, y_train)             
print("w = ", w,"\nl'erreur d'approximation : " , erreur_d_approximation,"\nNbr d'itiration t = ", t)

#----------la phase de teste---------------------------------------------------#
erreur_d_generalisation = calculdelsw(w, X_test, y_test)             
print("l'erreur de généralisation : " , erreur_d_generalisation)

#-------------------------------------------------9----------------------------------------------------------
print('\n\nQustion 9 : Appliquer la validation croisée K-Fold\n')
def KFold(X, K, shuffle = True, seed = 4321):
    # shuffle modifies indices inplace
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(indices)
    def test( n_samples, indices):
        fold_sizes = (n_samples // K) * np.ones(K, dtype = int)
        fold_sizes[:n_samples%K] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test = np.zeros(n_samples, dtype = bool)
            test[test_indices] = True
            yield test
            current = stop
    for test in test(n_samples, indices):
        i = indices[np.logical_not(test)]
        # print(i)
        j = indices[test]
        yield i, j

KF = KFold(X_train, K=5, shuffle=True, seed=4312)#return 5 echantillon

erreur_approx = 0
for i, j in KF:
    #print("TRAIN:", i, "TEST:", j)
    X_tr, X_tst = X_train[i], X_train[j]
    y_tr, y_tst = y_train[i], y_train[j]
    loss = calculdelsw(w, X_tr, y_tr)
    erreur_approx += loss
    erreur_generalisation = calculdelsw(w, X_tst, y_tst)
print("l'erreur d'aproximation is :", loss)
print("l'erreur de generalisation is :", erreur_generalisation)
print("l'erreur de k-fold",erreur_approx/5)


#------------------------------------------------10------------------------------------------------------------------
#for classification we use sign(wtx) for regression we use mean(wtx)
print(" Question 10 : estimation de Bias-variance ")
bias_var = np.sum((np.sign(np.dot(w,np.array([np.ones(len(X_test)),X_test[:,0],X_test[:,1]]))) - y_test) ** 2) / y_test.size
print('Average bias: %.3f' % bias_var)

#--------plot de resultat final-----------------------------------------------#
plt.clf() #--------clear figure--------#
plt.grid(False)#--------plot a grid--------#
plt.xlim(-2.5, 2.5)
plt.ylim(-2, 3)
plt.title('Result of Adaline')
z = np.linspace(X_train.min(), X_train.max(), 2)
plt.scatter(X_train[:,[0]], X_train[:,[1]], s=40, c = y_train)
k = (-w[0]/w[1])* z - w[2]/w[1]
plt.plot(z , k, color='green')  
plt.show()
