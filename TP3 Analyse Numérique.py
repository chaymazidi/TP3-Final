#!/usr/bin/env python
# coding: utf-8

# # <font color=red> Compte Rendu TP3: Intégration Numérique </font>

# ## <font color=green > I. Objectif: </font>

# L'objectif de ce TP est d'aborder le calcul général de l'intégrale de la fonction f (x) dans un domaine fini défini avec des termes finis a et b de différentes manières.
# ![2.PNG](attachment:2.PNG)

# ## <font color=green > II. Partie Théorique: </font>

# ## <span style="color:	olivedrab ">Motivation: </span>

# Dans certains cas très limités, une telle **`intégrale`** peut être calculée analytiquement (à la main). Cependant, ce n’est que très rarement possible, et le plus souvent un des cas suivants se présente :
# - Le calcul analytique est long, compliqué et rébarbatif
# - Le résultat de l’intégrale est une fonction compliquée qui fait appel à d’autres fonctions elles-même longues à évaluer
# - Cette intégrale n’a pas d’expression analytique
# Dans tous ces cas, on préfèrera calculer **`numériquement la valeur de l’intégrale I`**.
# 

# ## <span style="color:	olivedrab ">Principe: </span>

# L’idée principale est de trouver des méthodes qui permettent de calculer rapidement une valeur approchée I de l’intégrale à calculer tel que:
# ![3.PNG](attachment:3.PNG)
# Les méthode qu'on va utiliser pour calculer l'intergrale I sont :
# - **`Méthode des réctangles`**
# - **`Méthode du point milieu`**
# - **`Méthode des trapézes`**
# - **`Méthodes de Simpson`**

# ## <span style="color:	olivedrab ">Calcul approché des intégrales: </span>

# ## <span style="color:	maroon ">   (1) Méthode des Rectangles </span>

# ![8.PNG](attachment:8.PNG)

# ## <span style="color: 		purple ">Exemple: </span>

# ### Méthode des Rectangles à gauche:

# ![10.PNG](attachment:10.PNG)

# ### Méthode des Rectangles à droite:

# ![11.PNG](attachment:11.PNG)

# ![rect6.gif](attachment:rect6.gif)

# ## <span style="color:	maroon ">   (2) Méthode des trapèzes: </span>

# En analyse numérique, la méthode des trapèzes est une méthode pour le calcul numérique d'une intégrale I s'appuyant sur l'interpolation linéaire par intervalles.
# 
# Pour obtenir de meilleurs résultats, on découpe l'intervalle [a , b] en n intervalles plus petits et on applique la méthode sur chacun d'entre eux. Bien entendu, il suffit d'une seule évaluation de la fonction à chaque nœud :
# ![4.PNG](attachment:4.PNG)

# ## <span style="color: 		purple ">Exemple: </span>

# ![trap%C3%A9ze.png](attachment:trap%C3%A9ze.png)

# ## <span style="color:	maroon ">   (3) Méthode du point milieu: </span>

# En analyse numérique, la méthode du point médian est une méthode permettant de réaliser le calcul numérique d'une intégrale.
# 
# Le principe est d'approcher l'intégrale de la fonction f par l'aire d'un rectangle de base le segment [a,b] et de hauteur f(a+b/2).
# 
# ![6.PNG](attachment:6.PNG)

# ## <span style="color: 		purple ">Exemple: </span>

# ![point%20mileu.png](attachment:point%20mileu.png)

# ![Trapezium2.gif](attachment:Trapezium2.gif)

# ## <span style="color:	maroon ">   (4) Méthode de Simpson: </span>

# En analyse numérique, la méthode de Simpson, du nom de Thomas Simpson, est une technique de calcul numérique d'une intégrale.
# 
# Un polynôme étant une fonction très facile à intégrer, on approche l'intégrale de la fonction f sur l'intervalle [a, b], par l'intégrale de P sur ce même intervalle. On a ainsi, la simple formule :
# ![7.PNG](attachment:7.PNG)
# 

# ## <span style="color: 		purple ">Exemple: </span>

# ![Simpson_rule.png](attachment:Simpson_rule.png)

# ![ps205_982234Simpson.gif](attachment:ps205_982234Simpson.gif)

# # III. Partie pratique:

# ## importation:

# In[13]:


from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad 


# ## Classe de la méthode des Rectangles:

# In[14]:


class Rectangle(object): #class rectangle 
    def __init__(self, a, b, n, f):
        self.a = a
        self.b = b
        self.x = np.linspace(a, b, n+1)
        self.f = f
        self.n = n
    def integrate(self,f):
        x=self.x# contiens les xi
        y=f(x)#les yi 
        h = float(x[1] - x[0])
        s = sum(y[0:-1])
        return h * s
    def Graph(self,f,resolution=1001):
        xl = self.x
        yl = f(xl)
        xlist_fine=np.linspace(self.a, self.b, resolution)
        for i in range(self.n):
            x_rect = [xl[i], xl[i], xl[i+1], xl[i+1], xl[i]] # abscisses des sommets
            y_rect = [0   , yl[i], yl[i]  , 0     , 0   ] # ordonnees des sommets
            plot(x_rect, y_rect,"r")
        yflist_fine = f(xlist_fine)
        plt.plot(xlist_fine, yflist_fine)
        plt.plot(xl, yl,"gd")
        plt.ylabel('f(x)')
        plt.title('Méthode des Rectangles')
        plt.text( 0.5*( self.a+ self.b ) , f(self.b ) , 'I_{} ={:0.8f}'.format(self.n,self.integrate( f ) ) , fontsize =15 )


# ## Classe de la méthode du point milieu:

# In[15]:


class Milieu(object): #class rectange 
    def __init__(self, a, b, n, f):#initialiser les paramètres du classe
        self.a = a
        self.b = b
        self.x = np.linspace(a, b, n+1)
        self.f = f
        self.n = n
    def integrate(self,f):
        x=self.x# contiens les xi
        h = float(x[1] - x[0])
        s=0
        for i in range(self.n):
            s=s+f((x[i]+x[i+1])*0.5)
        return h*s
       
    def Graph(self,f,resolution=1001):
        xl = self.x
        yl=f(xl);
        xlist_fine=np.linspace(self.a, self.b, resolution)
        
        for i in range(self.n):
            
            m=(xl[i]+xl[i+1])/2
            x_rect = [xl[i], xl[i], xl[i+1], xl[i+1], xl[i]] # abscisses des sommets
            y_rect = [0   , f(m), f(m)  , 0     , 0   ] # ordonnees des sommets
            plot(x_rect, y_rect,"g")
            yflist_fine = f(xlist_fine)
            plt.plot(xlist_fine, yflist_fine)
            plt.plot(m,f(m),"r*")
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('Méthode du point milieu')
            plt.text( 0.5*( self.a+ self.b ) , f(self.b ) , 'I_{} ={:0.8f}'.format(self.n,self.integrate( f ) ) , fontsize =15 )


# ## Classe de la méthode des trapézes:

# In[16]:


class Trapezoidal(object):
    def __init__(self, a, b, n, f):
        self.a = a
        self.b = b
        self.x = np.linspace(a, b, n+1)
        self.f = f
        self.n = n
    def integrate(self,f):
        x=self.x
        y=f(x)
        h = float(x[1] - x[0])
        s = y[0] + y[-1] + 2.0*sum(y[1:-1])
        return h * s / 2.0
    def Graph(self,f,resolution=1001):
        xl = self.x
        yl = f(xl)
        xlist_fine=np.linspace(self.a, self.b, resolution)
        for i in range(self.n):
            x_rect = [xl[i], xl[i], xl[i+1], xl[i+1], xl[i]] # abscisses des sommets
            y_rect = [0   , yl[i], yl[i+1]  , 0     , 0   ] # ordonnees des sommets
            plot(x_rect, y_rect,"m")
        yflist_fine = f(xlist_fine)
        plt.plot(xlist_fine, yflist_fine)#plot de f(x)
        plt.plot(xl, yl,"cs")#point support
        plt.ylabel('f(x)')
        plt.title('Méthode des Trapézes')
        plt.text( 0.5*( self.a+ self.b ) , f(self.b ) , 'I_{} ={:0.8f}'.format(self.n,self.integrate( f ) ) , fontsize =15 )


# ## Classe de la méthode de Simpson:

# In[17]:


class Simpson (object) :
    def __init__ ( self ,a, b,n, f ) :
        self.a = a
        self.b = b
        self.x = np.linspace(a,b,n+1)
        self.f = f
        self.n = n
    def integrate ( self, f) :
        x=self.x
        y=f(x)
        h=float(x[1] - x[0])
        n =len(x)-1
        s = y[0] + y[-1] + 4.0*sum(y[1:-1])
        return h * s / 4.0
    def Graph ( self,f, resolution=1001) :
        xl=self.x
        yl= f(xl)
        xlist_fine=np.linspace(self.a , self.b,resolution)
        for i in range (self.n) :
            xx = np.linspace(xl[i] ,xl[i+1], resolution)
            m=(xl[i]+xl[i+1])/2
            a= xl[i]
            b= xl[i+1]
            l0=(xx-m)/(a-m)*(xx-b )/(a-b)
            l1=(xx-a)/(m-a )*(xx-b)/(m-b)
            l2=(xx-a )/(b-a)*(xx-m)/(b-m)
            P= f(a)*l0+f(m)*l1+f(b)*l2
            plt.plot(xx,P,"r")
        yflist_fine=f(xlist_fine)
        plt.plot(xlist_fine,yflist_fine,"b")
        plt.plot(xl,yl,"ro")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Methode de simpson')
        plt.text( 0.5*( self.a+ self.b ) , f(self.b ) , 'I_{} ={:0.8f}'.format(self.n,self.integrate( f ) ) , fontsize =15 )


# ## Réprésentation graphique et affichage:

# In[18]:


from ipywidgets import interact, interactive, fixed, interact_manual, widgets


# In[19]:


get_ipython().run_line_magic('matplotlib', 'widget')
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np


# In[20]:


def sim(a,b,n,f):
    T = Trapezoidal(a, b, n, f)
    S = Simpson(a, b, n, f)
    R = Rectangle(a, b, n, f)
    M = Milieu(a,b,n,f)

    #fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(221) 
    grid()
    T.Graph(f) 

    ax = fig.add_subplot(222)
    grid()
    S.Graph(f)

    ax = fig.add_subplot(223)
    grid()
    R.Graph(f)

    ax = fig.add_subplot(224)
    M.Graph(f)
    grid()

    plt.show()

    
    


# In[21]:


output = widgets.Output() 
with output:
    fig= plt.figure(figsize=(6,6))

fig.canvas.toolbar_position = 'bottom' 


# In[22]:


# create some control elements
int_slider = widgets.IntSlider(value=1, min=1, max=10, step=1, description='N')
color_picker = widgets.ColorPicker(value="red", description='color')
text_a= widgets.IntText(value=-1, description='borne a', continuous_update=False)
text_b = widgets.IntText(value=1, description='borne b', continuous_update=False)
select = widgets.Dropdown(options={'1/(1+x**2)':lambda x:1/(1+x**2),
                                    'sin(x)':lambda x: sin(x),
                                    'cos(x)':lambda x:cos(x),
                                    'x**2':lambda x:x**2},description='fonction f') 
button = widgets.Button(description="Simulation")

# callback functions
def update(change):
    """redraw line (update plot)"""
    fig.clear() 
    sim(text_a.value,text_b.value,int_slider.value,select.value)
   
def line_color(change):
    """set line color"""
    fig.clear()
    sim(text_a.value,text_b.value,int_slider.value,select.value)

def on_button_clicked(b):
    with output:
        fig.clear()
        sim(text_a.value,text_b.value,int_slider.value,select.value)

int_slider.observe(update, 'value')
color_picker.observe(line_color, 'value')


# In[24]:


controls = widgets.VBox([int_slider, color_picker,text_a, text_b,select,button])
button.on_click(on_button_clicked)
widgets.HBox([controls, output])


# ## Ou cas où, mon fichier n'est pas exécuté sur le binder, j'ai simulé mon travail sous forme de GIF.

# ![Simulation_TP3_analyse_num.gif](attachment:Simulation_TP3_analyse_num.gif)

# ## Interprétation:

# La plupart de ces méthodes d'intégration numérique fonctionnent sur le même principe. On commence par couper le gros intervalle [a,b] en N plus petits intervalles [ai,ai+1], avec a1=a et aN+1=b. Puis, pour chaque intervalle [ai,ai+1], on essaie d'approcher 
# ![integ%28ai+1,ai%29.PNG](attachment:integ%28ai+1,ai%29.PNG)
# 

# 
# - #### <span style="color:	red "> la méthode des rectangles à gauche : on approche </span>
# ![integ_rect.PNG](attachment:integ_rect.PNG)

# <div class="alert alert-success">
# Cela signifie qu'on approche l'intégrale de f par l'aire des rectangles hachurés en vert :
# </div>

# ![20.PNG](attachment:20.PNG)

# - #### <span style="color:	red "> la méthode du point milieu : on approche </span>

# ![integ_miliey.PNG](attachment:integ_miliey.PNG)

# <div class="alert alert-success">
# cela signifie qu'on approche l'intégrale de f par l'aire des rectangles hachurés en bleu :
# </div>

# ![21.PNG](attachment:21.PNG)

# - #### <span style="color:	red "> la méthode des trapèzes : on approche </span>

# ![integ_trapeze.PNG](attachment:integ_trapeze.PNG)

# <div class="alert alert-success">
# cela signifie qu'on approche l'intégrale de f par l'aire des trapèzes hachurés en marron :
# </div>

# ![22.PNG](attachment:22.PNG)

# - #### <span style="color:	red "> la méthode de Simpson : on approche </span>

# ![integ_sim.PNG](attachment:integ_sim.PNG)

# <div class="alert alert-success">
# cela signifie qu'on approche l'intégrale de f par l'aire de simpson coloré en jaune :
# </div>

# ![23.PNG](attachment:23.PNG)

# ## IV. Conclusion:

# Aprés la simulation de ces 4 méthodes de calcules de l'intégrale, on constate que:
# 
# - #### <span style="color:	#0000ff "> la méthode de Simpson </span>
# 
# fournit des résultats exacts pour des polynômes de degré inférieur ou égal à 3.
# À la fois à cause de **`sa simplicité`** de mise en œuvre, et de **`sa bonne précision`**, cette méthode est la plus utilisée par les calculatrices pour tous calculs approchés d'intégrales de fonctions explicites.
# 
# - #### <span style="color:	#0000ff "> la méthode des trapézes: </span>
# 
# Est la première des formules de Newton-Cotes, avec deux nœuds par intervalle. **`Sa rapidité`** de mise en œuvre en fait une méthode **`très employée`**. Cependant, la méthode de Simpson permet une estimation plus précise d'un ordre pour un coût souvent raisonnable.
# 
# - #### <span style="color:	#0000ff "> la méthode du point milieu: </span>
# 
# est une amélioration de la méthode d'Euler qui a sur cette dernière l'avantage d'être d'ordre 2 : si le pas de temps est divisé par 10, la **`précision`** augmente d'un facteur 100. 
# 
# - #### <span style="color:	#0000ff "> la méthode des rectangeles: </span>
# 
# Lorsque la fonction est continue et décroissante sur [a;b], les inégalités sont inversées: dans le cas où la fonction est monotone, la méthode des rectangles présente donc le (gros) avantage de donner un encadrement de I, ce qui permet facilement de donner un sens aux résultats renvoyés.
