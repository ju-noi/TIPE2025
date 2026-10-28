import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd



#-----------------------------------------------------------------------------------------
# simulation du nuages de points dans un plan pour l'entrainement
#-----------------------------------------------------------------------------------------

##générer un plan de points
a_th,b_th,c_th = 2,5,2
n=10 #nbr de point ie données entrantes
np.random.seed(0) 
x =np.random.rand(n).tolist()
y =np.random.rand(n).tolist()

z =[a_th * x[i] + np.random.rand(len(x)).tolist()[i]-0.5 +b_th*y[i]+c_th for i in range(len(x))]

plt.close('all')

#nuage de point pour l'apprentissage
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(x, y, z, label='données entrantes', marker='d',s=40) 
plt.title("simuler un nuage de point dans un plan")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------
# initialiser l'apprentissage
#-----------------------------------------------------------------------------------------

#générer des poids aléatoires
np.random.seed(0)
a= np.random.random()
b= np.random.random()
c = np.random.random()
w=np.array([a,b,c])
print('modèle non entrainé:',w)


e=[1 for i in range(n)]  
v=np.array([[x[i],y[i],e[i]]for i in range(n)])
w_th=np.array([a_th,b_th,c_th])

#-----------------------------------------------------------------------------------------
# apprentissage par descente du gradient
#-----------------------------------------------------------------------------------------

N=1000 #nbr de fois que le programme est entrainé
µ=0.05  #taux d'apprentissage
for e in range(N):
    z_pr=np.array(v@w) #@ produit matriciel
    dw=-2*np.array((v.T)@(z-z_pr))/n #.T transposée
    w=np.array(w-µ*dw)
print('paramètres théoriques:',w_th)    
print('fin exécution:', w)
    
plt.close('all')

ax=Axes3D(plt.figure())
W_list=w.tolist()

def f(x,y):
    return x*W_list[0]+y*W_list[1]+ W_list[2]


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
x_ax=np.linspace(0,1)
y_ay=np.linspace(0,2)
X,Y=np.meshgrid(x_ax,y_ay)
ax.scatter(x, y, z, marker='d', color='red', s=60, label=str(f"{a_th:.2f}")+'x +'+str(f"{b_th:.2f}")+'y +'+str(f"{c_th:.2f}"))
plt.title("plan de régression")
ax.plot_wireframe(X,Y,f(X,Y),color='grey', label=str(f"{W_list[0]:.2f}")+'x +'+str(f"{W_list[1]:.2f}")+'y +'+str(f"{W_list[2]:.2f}"))
ax.set_xlabel('X',)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
plt.show() 

#-----------------------------------------------------------------------------------------
# application à une base de données concrète
#-----------------------------------------------------------------------------------------

chemin = "./dataset/diabetes.csv"
dossier= pd.read_csv(chemin, decimal="." )
dataset_L=np.array(dossier).tolist()
dataset=np.array(dossier)
#print(dataset) #matrices des analyses


x = dossier['Age'][dossier['Outcome'] == 1]
y = dossier['BMI'][dossier['Outcome'] == 1]
z = dossier['Glucose'][dossier['Outcome'] == 1]

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

print(x, np.shape(x))
print(norm(x))

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
print(w)


e=[1 for i in range(n)]  
v=np.array([[norm(x)[i],norm(y)[i],e[i]]for i in range(n)])


N=100 #nbr de fois que le programme est entrainé
µ=0.05  #taux d'apprentissage
for e in range(N):#en imposant un nombre de passages
    z_pr=np.array(v@w)
    dw=-2*np.array((v.T)@(norm(z)-z_pr))
    w=np.array(w-µ*dw)
print(w_th, w)
    
# # Second sous-graphique (surface)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax=Axes3D(plt.figure())
W_list=w.tolist()

def f(x,y):
    return x*W_list[0]+y*W_list[1]+ W_list[2]

fig = plt.figure()
#ax = fig.gca(projection='3d')  # Affichage en 3D
ax = fig.add_subplot(projection = '3d')
x_ax=np.linspace(0,1)
y_ay=np.linspace(0,2)
X,Y=np.meshgrid(x_ax,y_ay)
ax.scatter(x, y, z, label='données entrantes', marker='d', color='red') 
#ax.plot_surface(X,Y,f(X,Y))
plt.title("tracé surface")
ax.plot_wireframe(X,Y,f(X,Y),color='grey')
plt.show()
