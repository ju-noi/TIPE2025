# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:39:15 2025

@author: Jula
"""

'''
Application à une base de données concrètes
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chemin = "./dataset/diabetes.csv"
dossier= pd.read_csv(chemin, decimal="." )
dataset_L=np.array(dossier).tolist()
dataset=np.array(dossier)
#print(dataset) #matrices des analyses


x = dossier['Age'].values.astype(float)[dossier['Outcome'] == 1]
y = dossier['BMI'].values.astype(float)[dossier['Outcome'] == 1]
z = dossier['Glucose'].values.astype(float)[dossier['Outcome'] == 1]

n = len(x)

#visualisation des données
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, marker='d', s=30, alpha=0.6) 
plt.title("Age, IMC et taux de glucose des patients atteints de diabète")
ax.set_xlabel('Âge')
ax.set_ylabel('BMI')
ax.set_zlabel('Glucose')
plt.legend()
plt.tight_layout()
plt.show()

# normalisation nécessaire pour la régression car échelles différentes

def norm(x):
    x.tolist()
    x_norm =[]
    min = x.min()
    max = x.max()
    for elt in x:
        x_norm.append((elt-min)/(max-min))
    return (x_norm)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(norm(x), norm(y), norm(z), marker='d', s=30, alpha=0.6) 
plt.title("visualisation des données normalisées")
ax.set_xlabel('Âge')
ax.set_ylabel('BMI')
ax.set_zlabel('Glucose')
plt.legend()
plt.tight_layout()
plt.show()

#générer des poids aléatoires

a= np.random.random()
b= np.random.random()
c = np.random.random()
w=np.array([a,b,c])
w=np.array([0.9, 0.5, 0.1])

print('initialisation: ', w)


e=[1 for i in range(n)]  
v=np.array([[norm(x)[i],norm(y)[i],e[i]]for i in range(n)])


N=1000 #nbr de fois que le programme est entrainé
µ=0.01  #taux d'apprentissage

for e in range(N):
    while np.isnan(w).any():
        z_pr=np.array(v@w)
        dw=-2*np.array((v.T)@(norm(z)-z_pr))
        w=np.array(w-µ*dw)
    e = N
W_list=w.tolist()
print(W_list)
def f(x,y):
    return x*W_list[0]+y*W_list[1]+ W_list[2]

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
x_ax=np.linspace(0,1)
y_ay=np.linspace(0,1)
X,Y=np.meshgrid(x_ax,y_ay)
ax.scatter(norm(x), norm(y), norm(z), marker='d', color='red') 
plt.title("tracé surface")
ax.plot_wireframe(X,Y,f(X,Y),color='grey', label= str(f"{W_list[0]:.2f}")+'x +'+str(f"{W_list[1]:.2f}")+'y +'+str(f"{W_list[2]:.2f}"))
ax.set_zlim(0, 1)
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Glucose')
plt.show()
