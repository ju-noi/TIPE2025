# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:35:54 2024

@author: Jula
"""
import numpy as np
import matplotlib.pyplot as plt
import random 
import pandas as pd

##method
#.tolist transforme un tableau (ici 1 ligne, n colonne) en liste
#le np.random.seed(graine) permet de faire la même simulation jusqu'à changer la graine


#-----------------------------------------------------------------------------------------
# simulation du nuages de points pour l'entrainement
#-----------------------------------------------------------------------------------------

a=3 # coeff directeur
o=2 #ordonnée à l'origine

n=100 #nbr de point ie données entrantes
np.random.seed(0) #pour avoir la même simulation à chaque exécution du programme
x = np.random.rand(n).tolist() #génère une liste de n flottant compris entre 0 et 1
    
y =[ o + a * x[i] + np.random.rand(len(x)).tolist()[i]-0.5 for i in range(len(x))]
#on retranche 0.5 car en moyenne le np.random.rand() nous décale en ordonnée y +0.5

##visualisation des listes
#print (x,y)

#générer le nuage de point
plt.figure(1)
plt.scatter(x,y,s=40) 
plt.xlabel('x')
plt.ylabel('y')
plt.title("nuage d'apprentissage selon la droite "+ str(a)+"x+"+ str(o))
plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------
# initialiser l'apprentissage
#-----------------------------------------------------------------------------------------

#modèle non entrainé (poids aléatoires)
np.random.seed(0)
w=np.random.random()
b=np.random.random()
print('paramètres générés (w,b)=(',w,',',b,')')

#-----------------------------------------------------------------------------------------
# apprentissage par descente du gradient
#-----------------------------------------------------------------------------------------

N=1000 #nbr de fois que le programme est entrainé
µ_0=0.02  #taux d'apprentissage fixé

def grad(N,n, µ_0,x,y,w,b):
    for e in range(N):
        dw=0
        db=0
        µ=µ_0
        for i in range(n): #somme
            y_i=w*x[i]+b #y prédit
            dw= dw+(y_i-y[i])*x[i]
            db=db+(y_i-y[i])
        dw=2*dw/n
        db=2*db/n
        
        #mise à jour des paramètres par descente du gradient
        w=w-µ*dw
        b=b-µ*db
    return(w,b)

w,b = grad(N,n, µ_0,x,y,w,b)
print('paramètres calculés (w,b)=(',w,',',b,')')



plt.figure(2)
X=np.linspace(0,1) #points d'abscisse répartis sur [O,1]
Y=w*X+b
plt.scatter(x,y,s=100)
plt.plot(X,Y, color='r',label="apprentissage: "+str(f"{w:.2f}")+'x+'+str(f"{b:.2f}"))
plt.plot(X, a*X+o, color='g', label="modèle théorique: "+str(f"{a:.2f}")+'x+'+str(f"{o:.2f}"))
plt.xlabel('x' )
plt.ylabel('y' )
plt.xticks( )
plt.yticks( )
plt.legend()
plt.title("Superposition du modèle à la droite théorique")
plt.show()

# #-----------------------------------------------------------------------------------------
# # évolution des paramètres en fonction du nombre d'itérations
# #-----------------------------------------------------------------------------------------

N=1000 #nombre d'itérations du programme
µ_0=0.75 #taux d'apprentissage modifiable
L_w=[]
L_b=[]

for e in range(1,N+1):
    dw=0
    db=0
    µ= µ_0
    for i in range(n): #somme
        y_i=w*x[i]+b   #y prédit
        dw= dw+(y_i-y[i])*x[i]
        db=db+(y_i-y[i])
    dw=2*dw/n #on obtient la dérivée de l'erreur par rapport à w
    db=2*db/n
    
    #mise à jour des paramètres par descente de gradient
    w=w-µ*dw
    b=b-µ*db
    L_w.append(w)
    L_b.append(b)

##tracé de l'évolution des poids w et b
    
plt.figure(3)
X=[i for i in range(1,N+1)]
Y=L_w
Z=L_b
plt.title('évolution des poids en fonction du nombre d itérations' )
plt.plot(X,Y, color='r', label='évolution w')
plt.plot(X,Z, color='b', label='évolution b')
plt.xticks( )
plt.yticks( )
plt.xlabel('nombre itérations' )
plt.legend( )
plt.show()

# #-----------------------------------------------------------------------------------------
# # utiliser le modèle sur d'autres données d'apprentissages
# #-----------------------------------------------------------------------------------------
np.random.seed(1)
a=random.randint(1, 5) # coeff directeur
o= random.randint(1, 5)#ordonnée à l'origine
print ('générer des paramètres théoriques: ', a,', ',o)

n=100 #nbr de point ie données entrantes
np.random.seed(1) #pour avoir la même simulation à chaque exécution du programme
x = np.random.rand(n).tolist() #génère une liste de n flottant compris entre 0 et 1
    
y =[ o + a * x[i] + np.random.rand(len(x)).tolist()[i]-0.5 for i in range(len(x))]

#modèle non entrainé
np.random.seed(1)
w=np.random.random()
b=np.random.random()
#print('modèle non entrainé (aléatoire):', w,b)

N=100 #nbr de fois que le programme est entrainé
µ_0=0.5  #taux d'apprentissage modifiable

w,b = grad(N,n, µ_0,x,y,w,b)

plt.figure(4)
X=np.linspace(0,1)
Y=w*X+b
plt.scatter(x,y,s=100)
plt.plot(X,Y, color='r',label="apprentissage: "+str(f"{w:.2f}")+'x+'+str(f"{b:.2f}"))
plt.plot(X, a*X+o, color='g', label="modèle théorique: "+str(f"{a:.2f}")+'x+'+str(f"{o:.2f}"))
plt.xlabel('x' )
plt.ylabel('y' )
plt.xticks( )
plt.yticks( )
plt.legend()
plt.title("Superposition du modèle à la droite théorique")
plt.show()

# #-----------------------------------------------------------------------------------------
# # Comparaison des taux d'apprentissage
# #-----------------------------------------------------------------------------------------

# algorithmes des perceptrons (taux d'apprentissage simple ou linéaire)
def perceptron_simple(x,y,w,b,N):
# liste de données d'apprentissage + liste de sorties théorique, N nbre de passage
    L=[]
    µ_0=0.5  #taux d'apprentissage
    for e in range(1,N+1):
        dw=0
        db=0
        µ= µ_0
        for i in range(n):
            y_i=w*x[i]+b #y prédit
            dw= dw+(y_i-y[i])*x[i]
            db=db+(y_i-y[i])
        dw=2*dw/n 
        db=2*db/n
        
#mise à jour des paramètres par descente de gradient
        w=w-µ*dw
        b=b-µ*db
    L.append(w)
    L.append(b)
    return L
    
def perceptron_lin(x,y,w,b,N):
    L=[]
    µ_0=0.5  #taux d'apprentissage
    for e in range(1,N+1):
        dw=0
        db=0
        µ= µ_0/(e+1)
        for i in range(n): 
            y_i=w*x[i]+b #y prédit
            dw= dw+(y_i-y[i])*x[i]
            db=db+(y_i-y[i])
        dw=2*dw/n 
        db=2*db/n
        
#mise à jour des paramètres par descente de gradient
        w=w-µ*dw
        b=b-µ*db
    L.append(w)
    L.append(b)
    return L

# pour le calcul du coefficient de corrélation
def moyenne(Y):
    m=0
    for i in range(len(Y)):
        m+=Y[i]
    return m/len(Y)

def coeff_Pearson(Y,X,w,b):
    S1=0
    S2=0
    for i in range(len(Y)):
        S1+= (Y[i]-(w*x[i]+b))**2
        S2+=(Y[i]-moyenne(Y))**2
    return 1-S1/S2

# tracer des graphiques
X=np.linspace(0,1)
Y_th= a*X+o

N=[5, 10, 100,1000]

p1, p2=100,1000 #nombre d'itérations

plt.figure(5)
for i in range(len(N)):
    n=N[i] #nbr de points
   
    plt.subplot((len(N)+1)//2, 2, i+1)
    x = np.random.rand(n).tolist() #génère une liste de n flottant compris entre 0 et 1
    y =[ o + a * x[i] + np.random.rand(len(x)).tolist()[i]-0.5 for i in range(len(x))]
    plt.scatter(x,y,s=10) #(prend en argument deux listes)

    plt.plot(X,Y_th, label="modèle théorique: "+str(f"{a:.2f}")+'x+'+str(f"{o:.2f}"), color='r')
    Y_simple= perceptron_simple(x,y,w,b,p1)[0]*X+perceptron_simple(x,y,w,b,p1)[1]
    plt.plot(X,Y_simple, label='taux simple, a='+str("{:.2f}".format(perceptron_simple(x,y,w,b,p1)[0]))
             + ' b='+ str("{:.2f}".format(perceptron_simple(x,y,w,b,p1)[1])))
    Y_lin= perceptron_lin(x,y,w,b,p2)[0]*X+perceptron_lin(x,y,w,b,p2)[1]
    plt.plot(X,Y_lin, label='taux itératif, a='+str("{:.2f}".format(perceptron_lin(x,y,w,b,p2)[0]))
             + ' b='+ str("{:.2f}".format(perceptron_lin(x,y,w,b,p2)[1])))
    plt.title("application du modèle pour "+ str(n)+ " données")
    print('taux simple',N[i], ':', str("{:.2f}".format( coeff_Pearson(y,x,perceptron_simple(x,y,w,b,p1)[0]
                                                                      ,perceptron_simple(x,y,w,b,p1)[1]))))
    print('taux itératif',N[i],':' ,str("{:.2f}".format(coeff_Pearson(y,x,perceptron_lin(x,y,w,b,p2)[0]
                                                                      ,perceptron_lin(x,y,w,b,p2)[1]))))
    plt.xticks()
    plt.yticks()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------
# application à une base de données concrète
#-----------------------------------------------------------------------------------------

chemin = './dataset/Salary_dataset.csv'
dossier= pd.read_csv(chemin, decimal="." )
print(dossier)
 

L_salaire=dossier['Salary'].tolist()
L_exp=dossier['YearsExperience'].tolist()


x = dossier['YearsExperience'].tolist()   
y = dossier['Salary'].tolist()


# #générer le nuage de point de la base de donnée
# plt.scatter(x,y,s=30) 
# plt.xlabel('expérience',)
# plt.ylabel('salaire',)
# plt.xticks()
# plt.yticks()
# plt.show()

#modèle non entrainé

w=np.random.random()
b=np.random.random()
#print(w,b)

n=len(x)
N=10000 #nbr de fois que le programme est entrainé
µ_0=0.02  #taux d'apprentissage

w,b = grad(N,n, µ_0,x,y,w,b) 
print('calculés',w,b)  
 
##tracé du graphique
plt.figure(6)
X=np.linspace(0,x[len(x)-1])
Y=w*X+b

plt.scatter(x,y,s=30)
plt.plot(X,Y, color='r', label=str(f"{w:.2f}")+'x +'+str(f"{b:.2f}"))
plt.title('prévision du salaire annuel en fonction de l expérience professionnelle')
plt.xticks()
plt.yticks()
plt.xlabel('expérience')
plt.ylabel('salaire')
plt.legend()
plt.show()


