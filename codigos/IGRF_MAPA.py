import pyIGRF
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
os.environ['PROJ_LIB'] = '/home/noelia/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
import pandas as pd


### Ejercicio 16
path= '/home/noelia/Escritorio/Geomagnetismo_curso/igrf13coeffs-OE.xlsx'
df = pd.read_excel(path)

df_array=np.array(df)
coeficientes=df_array[2:11,3:28]

lat=np.linspace(-90,90,100) # Defino la latitud entre +90 y -90
lon = np.linspace(-180,180,100) #Defino la longitud
alt = 0
date_2020 = 2020
date_1900 = 1900

theta = lat - 90*np.ones(100)
g10 = coeficientes[1][0]
g11 = coeficientes[2][0]
h11 = coeficientes[3][0]
g20 = coeficientes[4][0]
g21 = coeficientes[4][0]
h21 = coeficientes[6][0]
g22 = coeficientes[7][0]
h22 =coeficientes[8][0]
B0 = np.sqrt(g10**2+g11**2+h11**2)


intensidad_20 = np.zeros((len(lat),len(lon)))
intensidad_00 = np.zeros((len(lat),len(lon)))
intensidad_dipolo_00 = np.zeros((len(lat),len(lon)))
for i in range(len(lat)):
	for j in range(len(lon)):
		intensidad_20[i,j] = pyIGRF.igrf_value(lat[i], lon[j], alt, date_2020)[-1]
		intensidad_00[i,j] = pyIGRF.igrf_value(lat[i], lon[j], alt, date_1900)[-1]
		intensidad_dipolo_00[i,j] =B0*np.sqrt(4*np.cos(theta[i]*np.pi/180)**2+np.sin(theta[i]*np.pi/180)**2)

n = 100
new_lat=np.zeros((n,n))
for i in range(len(lat)):
	new_lat[i,:]=np.full((1,n), lat[i])

new_lon=np.zeros((n,n))
for i in range(len(lon)):
	new_lon[i,:]=np.full((1,n), lon[i])



fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121)
#ax.set_title('IGRF 1900, Intensidad del campo magnético [nT]',fontsize=16)
ax.set_title('Diferencia: Intensidad IGRF 1900 - Intensidad dipolo axial 1900 [nT]',size=16)


m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_00.T-intensidad_dipolo_00.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(122)
ax.set_title('Diferencia: Intensidad IGRF 2020 - Intensidad dipolo axial 2020 [nT]',fontsize=16)
#ax.set_title('IGRF 2020, Intensidad del campo magnético [nT]',fontsize=16)
m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_20.T-intensidad_dipolo_20.T,cmap='jet')
m.colorbar(pad=0.04)
fig.tight_layout()
plt.savefig('IGRF_dif.png', format='png', dpi=300)
plt.show()






##############################################3 Ejercicio 17

path= '/home/noelia/Escritorio/Geomagnetismo_curso/igrf13coeffs-OE.xlsx'
df = pd.read_excel(path)

df_array=np.array(df)
coeficientes=df_array[2:11,3:28]

# coeficientes del 2020
g10 = coeficientes[1][-1]
g11 = coeficientes[2][-1]
h11 = coeficientes[3][-1]
g20 = coeficientes[4][-1]
g21 = coeficientes[4][-1]
h21 = coeficientes[6][-1]
g22 = coeficientes[7][-1]
h22 =coeficientes[8][-1]
B0 = np.sqrt(g10**2+g11**2+h11**2)

RT=6371

#Para dipolo inclinado

#Coordenadas del polo norte, se puede hacer para diferentes años

thetaN=np.arccos(-g10/B0) 
phiN=np.arctan(h11/g11) 

t= thetaN
l= phiN

m_1=[ np.cos(t)*np.cos(l),np.cos(t)*np.sin(l),-np.sin(t)]
m_2=[-np.sin(l),np.cos(l),0]
m_3=[ np.sin(t)*np.cos(l),np.sin(t)*np.sin(l),np.cos(t)]

matrix = np.array([m_1,m_2,m_3])

lat=np.linspace(-90,90,100) # Defino la latitud entre +90 y -90
lon = np.linspace(-180,180,100) #Defino la longitud


intensidad_dipolo_inclinado_20 = np.zeros((len(lat),len(lon)))
for i in range(len(lat)):
	for j in range(len(lon)):
		theta = 90-lat[i]
		phi=lon[j]
		x = RT*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
		y = RT*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
		z = RT*np.cos(theta*np.pi/180)
		r = np.array((x,y,z))
		r_prima = matrix.dot(r)
		theta_cd = np.arctan(np.sqrt(r_prima[0]**2+r_prima[1]**2)/r_prima[2])
		theta_cd = np.pi-theta_cd
		intensidad_dipolo_inclinado_20[i,j] = B0*np.sqrt(4*np.cos(theta_cd)**2+np.sin(theta_cd)**2)


n = 100
new_lat=np.zeros((n,n))
for i in range(len(lat)):
	new_lat[i,:]=np.full((1,n), lat[i])

new_lon=np.zeros((n,n))
for i in range(len(lon)):
	new_lon[i,:]=np.full((1,n), lon[i])



######################################################33

#Para dipolo excentrico
    
L0=2*g10*g20+np.sqrt(3)*(g11*g21+h11*h21)
L1=-g11*g20+np.sqrt(3)*(g10*g21+g11*g22+h11*h22)
L2=-h11*g20+np.sqrt(3)*(g10*h21-h11*g22+g11*h22)
E=(L0*g10+L1*g11+L2*h11)/(4*B0**2)

#Posicion del excentrico
xi=(L0-g10*E)/(3*B0**2)
eta=(L1-g11*E)/(3*B0**2)
zeta=(L2-h11*E)/(3*B0**2)
        
intensidad_dipolo_excentrico_20= np.zeros((len(lat),len(lon)))
for i in range(len(lat)):
	for j in range(len(lon)):
		theta = 90-lat[i]
		phi=lon[j]
		x = RT*np.sin(theta*np.pi/180)*np.cos(phi*np.pi/180)
		y = RT*np.sin(theta*np.pi/180)*np.sin(phi*np.pi/180)
		z = RT*np.cos(theta*np.pi/180)
		r = np.array((x,y,z))
		rc = RT*np.array((xi,eta,zeta))
		r_ED = matrix.dot(r-rc)
		theta_ed=np.arctan(np.sqrt(r_ED[0]**2+r_ED[1]**2)/r_ED[2])
		theta_ed = np.pi-theta_ed
		intensidad_dipolo_excentrico_20[i,j] = B0*np.sqrt(4*np.cos(theta_ed)**2+np.sin(theta_ed)**2)





########################################################################
alt = 0
date_2020 = 2020


theta = lat - 90*np.ones(100)


intensidad_20 = np.zeros((len(lat),len(lon)))
intensidad_dipolo_20 = np.zeros((len(lat),len(lon)))

for i in range(len(lat)):
	for j in range(len(lon)):
		intensidad_20[i,j] = pyIGRF.igrf_value(lat[i], lon[j], alt, date_2020)[-1]
		intensidad_dipolo_20[i,j] =B0*np.sqrt(4*np.cos(theta[i]*np.pi/180)**2+np.sin(theta[i]*np.pi/180)**2)





fig = plt.figure(figsize=(13, 9))

ax = fig.add_subplot(222)
ax.set_title( 'Intensidad dipolo inclinado 2020 [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_inclinado_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(221)
ax.set_title('Intensidad dipolo inclinado 1900 [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_inclinado_00.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(224)
ax.set_title('Intensidad dipolo excéntrico 2020 [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_excentrico_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(223)
ax.set_title('Intensidad dipolo excéntrico 1900 [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_excentrico_00.T,cmap='jet')
m.colorbar(pad=0.04)

plt.tight_layout()
plt.savefig('COMPARACION.png', format='png',dpi=300)
plt.show()







fig = plt.figure(figsize=(13, 9))

ax = fig.add_subplot(121)
ax.set_title('Dif: Intensidad dipolo inclinado (2020-1900) [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_inclinado_20.T-intensidad_dipolo_inclinado_00.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(122)
ax.set_title('Dif: Intensidad dipolo excéntrico (2020-1900) [nT]')
map = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_excentrico_20.T-intensidad_dipolo_excentrico_00.T,cmap='jet')
m.colorbar(pad=0.04)


plt.tight_layout()
plt.savefig('COMPARACION_DIF.png', format='png',dpi=300)
plt.show()



































########################
######################################################33
lat_nor=[]
lat_sur=[]
lon_sur=[]
lon_nor=[]
for j in range(len(coeficientes[0,:])):
	g10 = coeficientes[1][j]
	g11 = coeficientes[2][j]
	h11 = coeficientes[3][j]
	g20 = coeficientes[4][j]
	g21 = coeficientes[4][j]
	h21 = coeficientes[6][j]
	h22 = coeficientes[7][j]
	B0 = np.sqrt(g10**2+g11**2+h11**2)
	thetaN=np.arccos(-g10/B0)*(180/np.pi)
	phiN=np.arctan(h11/g11) *(180/np.pi)
	PNlatCD=90-thetaN
	PNlongCD=phiN
	thetaSdeg=180-thetaN
	phiSdeg=180+phiN
	PSlatCD=90-thetaSdeg
	PSlongCD=phiSdeg
	lat_nor.append(PNlatCD)
	lat_sur.append(PSlatCD)
	lon_nor.append(PNlongCD)
	lon_sur.append(PSlongCD)
	


fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(121)
ax.set_title( 'POLO NORTE GEOMAGNÉTICO',fontsize=15,y = -0.15)

m = Basemap(projection='merc',
    resolution = 'l', area_thresh = 0.1,
    llcrnrlon=-150, llcrnrlat=50,
    urcrnrlon=10, urcrnrlat=85)
m.drawmapboundary(fill_color='aqua')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(40,90,10),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],fontsize=15)

x, y = m(lon_nor, lat_nor)
for j in range(len(lon_nor)):
	if j==0:
		plt.plot(x[j], y[j],'ro', markersize=8,label='1900')
	elif j==24:
		plt.plot(x[j], y[j],'go', markersize=8,label='2020')
	else: 
		plt.plot(x[j], y[j],'-o',color='blue', markersize=6)

plt.plot(x[0], y[0],'ro', markersize=8)
plt.legend(loc=1,fontsize=15)
ax = fig.add_subplot(122)
ax.set_title( 'POLO SUR GEOMAGNÉTICO',fontsize=15,y = -0.15)

m = Basemap(projection='merc',
    resolution = 'l', area_thresh = 0.1,
    llcrnrlon=-10, llcrnrlat=-85,
    urcrnrlon=150, urcrnrlat=-50)
m.drawmapboundary(fill_color='aqua')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(-90,-40,10),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],fontsize=15)

x, y = m(lon_sur, lat_sur)
for j in range(len(lon_nor)):
	if j==0:
		plt.plot(x[j], y[j],'ro', markersize=10,label='1900')
	elif j==24:
		plt.plot(x[j], y[j],'go', markersize=10,label='2020')
	else: 
		plt.plot(x[j], y[j],'-o', color='blue',markersize=5)

plt.plot(x[0], y[0],'ro', markersize=8)
plt.legend(loc=1,fontsize=15)
fig.suptitle('Evolución temporal de la posición de los polos geomagnéticos para el dipolo inclinado',y=0.92,size=20)
fig.tight_layout()
#fig.subplots_adjust(top=0.97)
plt.savefig('ET1.png', format='png', dpi=300)
plt.show()

########################################33################################33

#Evolucion temporal polos para el excentrico


lat_nore=[]
lat_sure=[]
lon_sure=[]
lon_nore=[]
for j in range(len(coeficientes[0,:])):
	g10 = coeficientes[1][j]
	g11 = coeficientes[2][j]
	h11 = coeficientes[3][j]
	g20 = coeficientes[4][j]
	g21 = coeficientes[4][j]
	h21 = coeficientes[6][j]
	g22 = coeficientes[7][j]
	h22 =coeficientes[8][-1]
	B0 = np.sqrt(g10**2+g11**2+h11**2)
	L0=2*g10*g20+np.sqrt(3)*(g11*g21+h11*h21)
	L1=-g11*g20+np.sqrt(3)*(g10*g21+g11*g22+h11*h22)
	L2=-h11*g20+np.sqrt(3)*(g10*h21-h11*g22+g11*h22)
	E=(L0*g10+L1*g11+L2*h11)/(4*B0**2)
	#En radianes
	theta_n=np.arccos(-g10/B0)
	phi_n=np.arctan(h11/g11) 
	theta_s=np.pi-theta_n
	phi_s=np.pi+phi_n	
	#Posicion del excentrico en cartesianas
	xi=(L0-g10*E)/(3*B0**2)
	eta=(L1-g11*E)/(3*B0**2)
	zeta=(L2-h11*E)/(3*B0**2)
	#EDcenterDist(yind)=RT*sqrt(xi^2+eta^2+zeta^2); % km
	RT=6371
	#North pole
	Z_n = RT*(np.cos(theta_n)-zeta)
	x_n= np.sin(theta_n)*np.cos(phi_n)*Z_n+RT*xi
	y_n = np.sin(theta_n)*np.sin(phi_n)*Z_n+RT*eta
	phi_nn = np.arctan(y_n/x_n)
	theta_nn = np.arctan(np.sqrt(x_n**2+y_n**2)/Z_n)
	PNlatED=90-theta_nn*180/np.pi
	PNlongED=phi_nn*180/np.pi
	Z_s = RT*(np.cos(theta_s)-zeta)
	x_s= np.sin(theta_s)*np.cos(phi_s)*Z_s+RT*xi
	y_s = np.sin(theta_s)*np.sin(phi_s)*Z_s+RT*eta
	phi_ss = np.arctan2(y_s,x_s)# Prboelam aqui!
	theta_ss = np.arctan2(np.sqrt(x_s**2+y_s**2),Z_s)
	PSlatED=90-theta_ss*180/np.pi
	PSlongED=180 +phi_ss*180/np.pi
	#phiNED=np.arctan(((RT-ddz)*np.sin(phiN)*np.tan(thetaN)+ddy)/((RT-ddz)*np.cos(phiN)*np.tan(thetaN)+ddx))
   	#if phiNED>=0; phiNED=phiNED-pi;end
	#thetaNED=np.arcsin((ddx*np.sin(phiN)-ddy*np.cos(phiN))/(RT*np.sin(phiN-phiNED)))
	#PNlatED=90-thetaNED*180/np.pi
	#PNlongED=phiNED*180/np.pi
	#%South pole
	#phiSED=np.arctan(((RT+ddz)*np.sin(phiN)*np.tan(thetaN)-ddy)/((RT+ddz)*np.cos(phiN)*np.tan(thetaN)-ddx))
    	#if phiSED<=0; phiSED=pi+phiSED;end
	#thetaSED=np.pi-np.arcsin((ddx*np.sin(phiN)-ddy*np.cos(phiN))/(RT*np.sin(phiN-phiSED)))
	#PSlatED=90-thetaSED*180/np.pi
	#PSlongED=phiSED*180/np.pi
	lat_nore.append(PNlatED)
	lat_sure.append(PSlatED)
	lon_nore.append(PNlongED)
	lon_sure.append(PSlongED)







fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(121)
ax.set_title( 'POLO NORTE GEOMAGNÉTICO',fontsize=15,y=-0.15)

m = Basemap(projection='merc',
    resolution = 'l', area_thresh = 0.1,
    llcrnrlon=-150, llcrnrlat=50,
    urcrnrlon=10, urcrnrlat=85)
m.drawmapboundary(fill_color='aqua')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(40,90,10),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],fontsize=15)

x, y = m(lon_nore, lat_nore)
for j in range(len(lon_nore)):
	if j==0:
		plt.plot(x[j], y[j],'ro', markersize=8,label='1900')
	elif j==24:
		plt.plot(x[j], y[j],'go', markersize=8,label='2020')
	else: 
		plt.plot(x[j], y[j],'-o',color='blue', markersize=6)

plt.plot(x[0], y[0],'ro', markersize=8)
plt.legend(loc=1,fontsize=15)
ax = fig.add_subplot(122)
ax.set_title( 'POLO SUR GEOMAGNÉTICO',fontsize=15,y = -0.15)

m = Basemap(projection='merc',
    resolution = 'l', area_thresh = 0.1,
    llcrnrlon=-10, llcrnrlat=-85,
    urcrnrlon=150, urcrnrlat=-50)
m.drawmapboundary(fill_color='aqua')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawparallels(np.arange(-90,-40,10),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],fontsize=15)

x, y = m(lon_sure, lat_sure)
for j in range(len(lon_sure)):
	if j==0:
		plt.plot(x[j], y[j],'ro', markersize=10,label='1900')
	elif j==24:
		plt.plot(x[j], y[j],'go', markersize=10,label='2020')
	else: 
		plt.plot(x[j], y[j],'-o', color='blue',markersize=5)

plt.plot(x[0], y[0],'ro', markersize=8)
plt.legend(loc=1,fontsize=15)
fig.suptitle('Evolución temporal de la posición de los polos geomagnéticos para el dipolo excéntrico',y=0.92,size=20)
fig.tight_layout()
#fig.subplots_adjust(top=0.97)
plt.savefig('ET2.png', format='png', dpi=300)
plt.show()



#####################################################################################



fig = plt.figure(figsize=(14, 12))

ax = fig.add_subplot(323)
ax.set_title( 'Intensidad dipolo inclinado [nT]',fontsize=15)
m= Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_inclinado_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(324)
ax.set_title('Dif: Intensidad IGRF - Intensidad dipolo inclinado [nT]',fontsize=15)
m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.))
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_20.T-intensidad_dipolo_inclinado_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(325)
ax.set_title('Intensidad dipolo excéntrico [nT]',fontsize=15)
m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_excentrico_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(326)
ax.set_title('Dif: Intensidad IGRF - Intensidad dipolo excéntrico [nT]',fontsize=15)
m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_20.T-intensidad_dipolo_excentrico_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(321)
ax.set_title( 'Intensidad dipolo axial [nT]',fontsize=15)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0],fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_dipolo_20.T,cmap='jet')
m.colorbar(pad=0.04)

ax = fig.add_subplot(322)
ax.set_title( 'Dif: Intensidad IGRF - Intensidad dipolo axial [nT]',fontsize=15)

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,90.,30.),fontsize=15)
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1],fontsize=15)
x,y = m(new_lon,new_lat) # paso a coordenadas cartesianas
m.pcolormesh(x,y.T,intensidad_20.T-intensidad_dipolo_20.T,cmap='jet')
m.colorbar(pad=0.04)


fig.suptitle('2020',y=1,size=22)
plt.tight_layout()
plt.savefig('COMPARACION_2020.png', format='png',dpi=300)
plt.show()






