
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import pearsonr
from dateutil.relativedelta import relativedelta

path= '/home/noelia/Escritorio/Geomagnetismo_curso/data.xlsx'
df = pd.read_excel(path)
df_array=np.array(df)

aa = df_array[:,2] 
rz = df_array[:,3]
f = df_array[:,4] 
ap=df_array[:,5] 


time= df_array[:,0:2] 
pearsonr(rz, aa)

date=[]
for t in time:
	tiempo=str(int(t[0]))+str(int(t[1]))
	tiempo= datetime.datetime.strptime(tiempo, '%Y%m')
	date.append(tiempo)

aa1= (aa-np.ones(len(aa))*np.mean(aa))/np.mean(aa)
rz1= (rz-np.ones(len(rz))*np.mean(rz))/np.mean(rz)
f1= (f-np.ones(len(f))*np.mean(f))/np.mean(f)
ap1= (ap-np.ones(len(ap))*np.mean(ap))/np.mean(ap)

# Promedio movil 11 meses. 


start = date[0]
end = date[-1]


escala_temporal=[]
while start <=end:
	start +=relativedelta(months=+6)
	escala_temporal.append(start)

aaanual = []
rzanual = []
fanual=[]
apanual=[]
date=np.array(date)
	
for j in range(len(escala_temporal)-1):
	choice=np.logical_and(date >= escala_temporal[j], date < escala_temporal[j+1])
	aam=np.extract(choice, aa)
	rzm = np.extract(choice,rz)
	fm=np.extract(choice,f)
	apm =np.extract(choice,ap)
	aaanual.append(np.mean(aam))
	rzanual.append(np.mean(rzm))
	fanual.append(np.mean(fm))
	apanual.append(np.mean(apm))

			
aa12= (aaanual-np.ones(len(aaanual))*np.mean(aaanual))/np.mean(aaanual)
rz12= (rzanual-np.ones(len(rzanual))*np.mean(rzanual))/np.mean(rzanual)
f12= (fanual-np.ones(len(fanual))*np.mean(fanual))/np.mean(fanual)
ap12= (apanual-np.ones(len(apanual))*np.mean(apanual))/np.mean(apanual)


############ Promedio movil 11 años 
start = date[0]
end = date[-1]


escala_temporal_=[]
while start <end:
	start +=relativedelta(months=+66)
	escala_temporal_.append(start)

aaanual = []
rzanual = []
fanual=[]
apanual=[]
date=np.array(date)
	
for j in range(len(escala_temporal)-1):
	choice=np.logical_and(date >= escala_temporal_[j], date < escala_temporal_[j+1])
	aam=np.extract(choice, aa)
	rzm = np.extract(choice,rz)
	fm=np.extract(choice,f)
	apm =np.extract(choice,ap)
	aaanual.append(np.mean(aam))
	rzanual.append(np.mean(rzm))
	fanual.append(np.mean(fm))
	apanual.append(np.mean(apm))

			
aa13= (aaanual-np.ones(len(aaanual))*np.mean(aaanual))/np.mean(aaanual)
rz13= (rzanual-np.ones(len(rzanual))*np.mean(rzanual))/np.mean(rzanual)
f13= (fanual-np.ones(len(fanual))*np.mean(fanual))/np.mean(fanual)
ap13= (apanual-np.ones(len(apanual))*np.mean(apanual))/np.mean(apanual)


fig, (ax, ax1, ax2) = plt.subplots(nrows=3, sharex=True,figsize=(9,9))
fig.suptitle('Series  temporales', 
             fontsize=14, fontweight='bold')
ax.set_title( 'Promedio mensual')
plt.text(0.5, 0.95, 'Series temporales', color='r',fontsize=20)
ax.plot(date,aa1,'r',label='aa')
ax.plot(date,rz1,'b',label='Rz')
ax.plot(date,f1,'k',label='F10.7')
ax.plot(date,ap1,'g',label='Ap')
ax.legend(loc='best')
ax.set(ylabel="%")
ax.set_ylim([-2, 3])
ax1.set_title('Promedio mensual móvil 12 meses')
ax1.plot(escala_temporal[0:-1],aa12,'r',label='aa')
ax1.plot(escala_temporal[0:-1],rz12,'b',label='Rz')
ax1.plot(escala_temporal[0:-1], f12,'k',label='F10.7')
ax1.plot(escala_temporal[0:-1],ap12,'g',label='Ap')
ax1.legend(loc='best')
ax1.set(ylabel="%")
ax1.set_ylim([-2, 3])
ax2.set_title('Promedio mensual móvil 11 años')
ax2.plot(escala_temporal_[0:-1],aa13,'r',label='aa')
ax2.plot(escala_temporal_[0:-1],rz13,'b',label='Rz')
ax2.plot(escala_temporal_[0:-1], f13,'k',label='F10.7')
ax2.plot(escala_temporal_[0:-1],ap13,'g',label='Ap')
ax2.legend(loc=1)
ax2.set(xlabel= "Año", ylabel="%")
ax2.set_ylim([-1, 1])
fig.subplots_adjust(hspace=0.5, bottom=0.1)
fig.tight_layout()
plt.savefig('indices.png', format='png', dpi=300)
plt.show()


 
