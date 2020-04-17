# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:01:14 2020

@author: ALmaMY yOUSsEf Ly
"""


#Loi Binomiale
import numpy as np
import scipy.stats as sps
#Représentation du diagramme en bâtons
import matplotlib.pyplot as plt
n, p, N = 20, 0.3, int(1e4)
B = np.random.binomial(n, p, N)
f = sps.binom.pmf(np.arange(n+1), n, p)
plt.hist(B,bins=n+1,normed=1,range=(0.5,n+.5), color="white",label="loi empirique")
plt.stem(np.arange(n+1),f,"r",label="loi theorique")
plt.title("loi Binomiale")
plt.legend()



#loi uniforme
import numpy as np
#Représentation graphique
import matplotlib.pyplot as plt
s = np.random.uniform(-1,0,1000)
count, bins, ignored = plt.hist(s, 15, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.title("loi Uniforme")
plt.show()



#Loi normale
import numpy as np
"""Affichage de  l'histogramme des échantillons,
 ainsi que la fonction de densité de probabilité"""
import matplotlib.pyplot as plt
mu, sigma = 0, 0.1     # moyenne et écart type
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.title("loi noramle")
plt.show()




#Loi de poisson
#Tirez des échantillons de la distribution
import  numpy  as  np 
#Affichage de l'histogramme de l'échantillon
import matplotlib.pyplot as plt
s = np.random.poisson(5, 10000)
count ,  bins ,  ignored  =  plt . hist ( s ,  14 ,  normed = True ) 
plt.title("loi de Poisson")
plt . show ()



#Loi exponential
import numpy as np 
#affichage des courbes
import matplotlib.pyplot as plt  
in_array = [1, 1.2, 1.4, 1.6, 1.8, 2] 
out_array = np.exp(in_array)  
y = [1, 1.2, 1.4, 1.6, 1.8, 2] 
plt.plot(in_array, y, color = 'blue', marker = "*")   
#graphe en rouge pour numpy.exp() 
plt.plot(out_array, y, color = 'red', marker = "o") 
plt.title("numpy.exp()")
plt.xlabel("X") 
plt.ylabel("Y") 
plt.title("loi exponential")
plt.show()   



#simulation de la loi de bernouillie
def simu_Bernouilli(p):
    import random as r #chargement du module random
    y = r.random()#génére un nbre y aléatoirment entre 0 et 1 avec une loi uniforme continue
    if y < p:     #p appartient à ]0,1[
        X = 1
    else:
        X = 0
    return X

#simulation de la loi binomiale
#1iére méthode: méthode Monté-Carlo
def simu_Binomiale(n,p):
    import random as r #chargement du module random
    for i in range (n):
        X = 0 #on initialise la var à 0
        y = r.random() #génére un nbre y aléatoirment entre 0 et 1 avec une loi uniforme continue
        if y < p:    #puis on trace un segment entre 0 et 1. Si y < p: succés :1 sinon échec: 0
            X = X + 1
    return X
 
#2iéme méthode: module numpy
def simu_Binomiale_bis(n,p):
    import numpy as np
    X = np.random.binomial(n,p)
    return X

#pour une matrice
    #n : nbre de fois que l'on répéte la de bernouillie de paramétre p 
    #l et c: respectivement le nbre de ligne et de colonne de la matrice (shape)
def simu_Binomiale_mat(n,p,l,c): 
    #module random du module numpy et dans ce module on retrouve la ftn de la loi binomiale
    import numpy as np
    X = np.random.binomial(n,p,(l,c))
    return X

#pour une liste
def simu_Binomiale_list(n,p,l): 
    #module random du module numpy et dans ce module on retrouve la ftn de la loi binomiale
    #l: taille de la liste
    import numpy as np
    X = np.random.binomial(n,p,l)
    return X
print(simu_Binomiale_list(30,0.2,10000))

#Simulation loi géométrique
#cas1: Monté Carlos

def simu_geometrique(p): #p est dans 0 et 1
    import random as r
    X = 1 #on initialise la var à 1
    y = r.random() #on génre un nbre aléatoire avec une loi uniforme continue
    while  y < p:
        X += 1
    return X
    
#cas2: module numpy
def simu_geometrique_bis(p):
    import numpy as np
    X = np.random.geometric(p)
    return X

#liste
def simu_geometrique_bis_liste(p,l):
    import numpy as np
    X = np.random.geometric(p,l)
    return X   

#matrice
import numpy as np
X = np.random.geometric(0.5,(3,3))


#simuler la loi uniforme discréte
import random as r
X = r.randint(1, 6)
 

#simuler la loi de poisson
import numpy as np
e = 0.1
X = np.random.poisson(e) 

#liste
import numpy as np
e = 0.1
l = 100
X = np.random.poisson(e, l) 

#matrice 
import numpy as np
e = 0.1
l = 4
c = 4
X = np.random.poisson(e, (l,c))















































