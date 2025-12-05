import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

#Ejercicio 2 y 3
path= '/home/noelia/Escritorio/Geomagnetismo_curso/igrf13coeffs-OE.xlsx'
df = pd.read_excel(path)

df_array=np.array(df)
igrf_2020=df_array[3:,27]
m=df_array[3:,2]
l=df_array[3:,1]

r =  6371
#r=3840
re = 6371 

energia_l=[]

for j in range(1,14):
	coef=igrf_2020[np.where(l==j)]
	energia = (j+1)*sum(coef**2)*((re/r)**(2*j+4))
	energia_l.append(energia)
	

B_cuadrado=sum(energia_l)

Bd_cuadrado = energia_l[0]

B_cuadri_cuadrado = energia_l[1]

B_octu_cuadrado = energia_l[2]


fraccion_nodip= np.sqrt(sum(energia_l[1:])/energia_l[0])

fraccion_dip_cuadri= np.sqrt(energia_l[1]/energia_l[0])

fraccion_dip_octu= np.sqrt(energia_l[2]/energia_l[0])


print(fraccion_no_dip,fraccion_dip_cuadri,fraccion_dip_octu)
#########################3 Ejercicio 4

l = np.arctan(np.tan(45*np.pi/180))/2
l*180/np.pi
###################################Ejercicio9
path= '/home/noelia/Escritorio/Geomagnetismo_curso/igrf13coeffs-OE.xlsx'
df = pd.read_excel(path)

df_array=np.array(df)
igrf_2005=df_array[3:,24]
m=df_array[3:,2]
l=df_array[3:,1]

r = 6371
r =  3840

re = 6371 

energia_l=[]

for j in range(1,14):
	coef=igrf_2005[np.where(l==j)]
	energia = (j+1)*sum(coef**2)*((re/r)**(2*j+4))
	energia_l.append(energia)
	


total=sum(energia_l)

energia_l[0]

energia_l[1]

cociente = energia_l[0]/total


###################################Ejercicio12

path= '/home/noelia/Escritorio/Geomagnetismo_curso/igrf13coeffs-OE.xlsx'
df = pd.read_excel(path)

df_array=np.array(df)
igrf_dipolar=df_array[2:6,3:28]
igrf_cuadri=df_array[6:11,3:28]
igrf_octo=df_array[11:18,3:28]
m=df_array[3:,2]
l=df_array[3:,1]

r =  6371
re = 6371 

energia_dipolar_año=[]
energia_cuadri_año=[]
energia_octo_año=[]
year=[]
j=1
n=2
p=3
for i in range(0,24):
	energia_d = (j+1)*sum(igrf_dipolar[1:,i]**2)*((re/r)**(2*j+4))
	energia_dipolar_año.append(energia_d)
	energia_c = (n+1)*sum(igrf_cuadri[:,i]**2)*((re/r)**(2*n+4))
	energia_cuadri_año.append(energia_c)
	energia_o = (p+1)*sum(igrf_octo[:,i]**2)*((re/r)**(2*p+4))
	energia_octo_año.append(energia_o)
	date = datetime. strptime(str(igrf_dipolar[0,i]), '%Y')
	year.append(date)


igrf_2020=df_array[3:,3:27]

energia_total = []

for k in range(0,24):
	energia_l=0
	for j in range(1,14):
		igrf = igrf_2020[:,k]
		coef = igrf[np.where(l==j)]
		energia = (j+1)*sum(coef**2)*((re/r)**(2*j+4))
		energia_l=energia_l+energia
	energia_total.append(energia_l)




fig, axs = plt.subplots(1,2, figsize=(15,6))

fig.suptitle('Evolución temporal de la energía en la superficie de la Tierra', fontsize=18)
axs[0].plot(year, energia_dipolar_año,'rx',label='Dipolar')
axs[0].plot(year, energia_total, 'kx',label='Total')
axs[0].set_xlabel('Año',fontsize=18)
axs[0].set_ylabel('Energía',fontsize=18)
axs[0].tick_params(axis='both',labelsize=16)
axs[0].legend(loc='best',fontsize=16)
axs[1].plot(year, energia_cuadri_año,'bx',label='Cuadrupolar')
axs[1].plot(year, energia_octo_año, 'gx',label='Octupolar')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[1].xaxis.set_major_locator(mdates.YearLocator(20,month=1,day=1))
axs[1].set_xlabel('Año',fontsize=18)
axs[1].set_ylabel('Energía',fontsize=18)
axs[1].tick_params(axis='both',labelsize=16)
plt.legend(loc='best',fontsize=16)
plt.savefig('energia_E.png', format='png', dpi=300)
plt.show()


###################################Ejercicio 14

igrf_dipolar=df_array[2:6,3:28]
igrf_cuadri=df_array[6:11,3:28]
igrf_octo=df_array[11:18,3:28]
m=df_array[3:,2]
l=df_array[3:,1]

r =  6371
re = 6371 

energia_dipolar_año=[]
energia_cuadri_año=[]
energia_octo_año=[]
year=[]
j=1
n=2
p=3
for i in range(0,24):
	energia_d = (j+1)*sum(igrf_dipolar[1:,i]**2)*((re/r)**(2*j+4))
	energia_dipolar_año.append(energia_d)
	energia_c = (n+1)*sum(igrf_cuadri[:,i]**2)*((re/r)**(2*n+4))
	energia_cuadri_año.append(energia_c)
	energia_o = (p+1)*sum(igrf_octo[:,i]**2)*((re/r)**(2*p+4))
	energia_octo_año.append(energia_o)
	date = datetime. strptime(str(igrf_dipolar[0,i]), '%Y')
	year.append(date)

varo=np.array(energia_octo_año)/np.array(energia_dipolar_año)
vari=np.array(energia_cuadri_año)/np.array(energia_dipolar_año)

varot=np.array(energia_octo_año)/np.array(energia_total)
varit=np.array(energia_cuadri_año)/np.array(energia_total)

fig, axs = plt.subplots(2,2, figsize=(15,6), sharex=True)
axs[0,0].plot(year, 100*vari,'bx')
axs[0,0].set_ylabel('(%)',fontsize=18)
axs[0,0].set_title('Cuadrupolo-Dipolo',fontsize=16)
axs[0,0].tick_params(axis='both',labelsize=15)
axs[1,0].set_title('Octupolo-Dipolo',fontsize=16)
axs[1,0].plot(year, 100*varo,'gx')
axs[1,0].tick_params(axis='both',labelsize=15)
#axs[1,0].set(xlabel='Año',ylabel='(%)',fontsize=18)
axs[1,0].set_ylabel('(%)',fontsize=18)
axs[1,0].set_xlabel('Año', fontsize = 18)
axs[0,1].plot(year, 100*varit,'bx')
axs[0,1].set_title('Cuadrupolo-Total',fontsize=16)
axs[0,1].set_ylabel('(%)',fontsize=18)
axs[0,1].tick_params(axis='both',labelsize=15)
axs[1,1].plot(year, 100*varot,'gx')
axs[1,1].set_title('Octupolo-Total',fontsize=16)
axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[1,1].xaxis.set_major_locator(mdates.YearLocator(20,month=1,day=1))
axs[1,1].tick_params(axis='both',labelsize=15)
axs[1,1].set_ylabel('(%)',fontsize=18)
axs[1,1].set_xlabel('Año', fontsize = 18)
fig.suptitle('Evolución temporal de la variación relativa de la energía en la superficie de la Tierra', y=1,fontsize=18)
#plt.tight_layout()
plt.savefig('variacion_COMPLETO_hoy.png', format='png', dpi=300)
plt.show()

########################### Dipolo axial
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['PROJ_LIB'] = '/home/noelia/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap

# Mean magnitude of the Earth's magnetic field at the equator in nT
B0 = 31200

n=1000

theta = np.linspace(180,0,n)
 
intensidad =B0*np.sqrt(4*np.cos(theta*np.pi/180)**2+np.sin(theta*np.pi/180)**2)

lat=90*np.ones(n)-theta # Defino la latitud entre +90 y -90
lon = np.linspace(-180,180,100) #Defino la longitud

new_data=np.zeros((n,n))
for i in range(len(intensidad)):
	new_data[i,:]=np.full((1,n), intensidad[i])

new_lat=np.zeros((n,n))
for i in range(len(lat)):
	new_lat[i,:]=np.full((1,n), lat[i])

new_lon=np.zeros((n,n))
for i in range(len(lon)):
	new_lon[i,:]=np.full((1,n), lon[i])


inclinacion = np.arctan(2*np.tan(lat*np.pi/180))
inc=inclinacion*180/(np.pi)

new_inc=np.zeros((n,n))
for i in range(len(intensidad)):
	new_inc[i,:]=np.full((1,n), inc[i])


#latt=45.00*np.ones(n)

fig = plt.figure(num=None, figsize=(12, 8))
#m = Basemap(projection='hammer',lon_0=0,resolution='c')
#m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
parallels = [45]
m.drawparallels(parallels, linewidth=2,labels=[1,0,0,0])
parallels = [45]
m.drawparallels(parallels, linewidth=2,labels=[1,0,0,0])
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,new_inc.T,cmap='jet')
m.colorbar()
plt.text(1111248.7, 21194087.36, 'I = 63.43°',fontsize=10,fontweight='bold',ha='left',va='top',color='k')
plt.title('Inclinación I para un dipolo axial centrado')
plt.show()



