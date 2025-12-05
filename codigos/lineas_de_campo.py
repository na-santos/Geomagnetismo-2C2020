https://ernestocrespo13.wordpress.com/2018/01/07/graficar-lineas-de-campo-electrico-con-matplotlib-y-python/



#GRAFICA DIPOLO ELECTRICO 3DB

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
	
phi=np.linspace(0, np.pi, 100)
theta=np.arange(0,2*np.pi,0.01)
theta1= (np.pi/2)*np.ones(len(theta))-theta
l=np.linspace(0,1,10)

Color=["#82FA58","#82FA58","#64FE2E","#3ADF00","#31B404","#298A08","#21610B","#21610B","#21610B","#21610B","#21610B"]

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(131, polar=True)
ax1= fig.add_subplot(132, polar=True)
ax2= fig.add_subplot(133, polar=True)
for i in range(0,10):
	r=l[i]*np.sin(theta)**2
	ax1.plot(theta,r,Color[i], linewidth=1.5)
	ax.plot(theta1,r,Color[i], linewidth=1.5)
	r_=l[i]*(abs(np.sin(theta)))*((abs(np.cos(theta)))**0.5)
	r__=l[i]*((4-5*np.sin(theta)**2)*np.sin(theta)**2)**(1/3)
	ax2.plot(theta,r_,Color[i], linewidth=1.5)

ax.set_title('Dipolo magnético axial 2D',fontsize=15,pad=18)
ax1.set_title('Dipolo magnético ecuatorial 2D',fontsize=15,pad=18)
ax2.set_title('Cuadripolo magnético axial 2D',fontsize=15,pad=18)
plt.tight_layout()
plt.savefig('lineas__.png', format='png', dpi=300)
plt.show()

