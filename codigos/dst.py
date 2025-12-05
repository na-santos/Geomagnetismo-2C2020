# To run this, download the BeautifulSoup zip file
# http://www.py4e.com/code3/bs4.zip
# and unzip it in the same directory as this file


import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl
from datetime import datetime,timedelta,date
import matplotlib.pyplot as plt
import datetime 
import matplotlib.dates as mdates
import numpy as np

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


import datetime

date_time_str = '2001/01'
start = datetime.datetime.strptime(date_time_str, '%Y/%m')

date_time_str = '2001/07'
now =datetime.datetime.strptime(date_time_str, '%Y/%m')

escala_temporal=[]
while start <= now:
	escala_temporal.append(start)
	start +=timedelta(hours=1)


######################################################################################33



def kyoto(fecha):
	url = 'http://wdc.kugi.kyoto-u.ac.jp/dst_final/'+fecha[0:4]+fecha[4:7]+'/index.html'
	html = urllib.request.urlopen(url, context=ctx).read()
	#Abro el html
	soup = BeautifulSoup(html, 'html.parser')
	#Me fije en que tag esta la data que quiero, se llama pre
	table = soup.find("pre").contents
	#separo por /n (son espacios que hay en determinadas horas), elimino los strings que no me interesan
	u=table[2].split('\n')[7:-1]
	str_list = list(filter(None, u))
	# Separo por espacios
	new_list=list()
	for s in str_list:
		new_list.extend(s.split()[1:])
	cor = []	
	for ele in new_list:
		if len(ele)>4:
			a=ele.split('-')
			a =a[1:]
			for al in a:
				al='-'+al
				cor.append(al)
			print(a)
		else:
			cor.append(ele)
	# Convierto los strings en enteros y me quedo con los datos hasta el dia de hoy
	new_list_int=[int(j) for j in cor]
	index = [index for index,value in enumerate(new_list_int) if abs(value) > 1000]
	if len(index)!=0:
		new_list_e =new_list_int[0:index[0]]
	else:
		new_list_e = new_list_int 
	return new_list_e
	


###############################################################################



	
completo = []
for fecha in escala_temporal: 
	if str(fecha)[8:13] == '01 00':
		fecha = str(fecha)[0:4]+str(fecha)[5:7]
		somelist=kyoto(fecha)
		completo.extend(somelist)
		#seven_days = somelist_pre[-len(escala_temporal):]
#####################################################################################

l=len(escala_temporal)
leve=np.linspace(-30,-30,l)
moderada=np.linspace(-50,-50,l)
intensa=np.linspace(-100,-100,l)
cero =np.linspace(0,0,l)
#a=min(completo)
#minpos=seven_days.index(min(completo))
#time_min = escala_temporal[minpos]


ax = plt.subplot()
ax.plot(escala_temporal,cero,'--',color='gray')
ax.plot(escala_temporal,intensa,color='#ef3f23')
plt.style.use('default')
plt.rc('font', family='serif')
plt.title('Índice DST',fontsize=15)
#plt.suptitle('Figura generada por LAMP', fontsize=9,x=0.16)
plt.xlabel('Fecha dd-mm-aa',fontsize=15,labelpad=7.5)
plt.ylabel( 'DST (nT)',fontsize=15,labelpad=7.5)
plt.ylim(-400,50)
plt.xlim(escala_temporal[0],escala_temporal[-1])
ax.text(0.15, 0.52,'Intensa',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='#ef3f23', fontsize=12)
ax.plot(escala_temporal,moderada,color='#f36f21')
ax.text(0.19, 0.695, 'Moderada',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='#f36f21', fontsize=12)
ax.plot(escala_temporal,leve,color='#faa61a')
ax.text(0.1,0.775, 'Leve',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='#faa61a', fontsize=12)
ax.text(0.12, 0.93,'Calmo',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='#c7dbf0', fontsize=12)
ax.axhspan(50, -30, facecolor='azure', alpha=0.5)
ax.axhspan(-30, -50, facecolor='lemonchiffon', alpha=0.5)
ax.axhspan(-50, -100, facecolor='peachpuff', alpha=0.5)
ax.axhspan(-100, -400, facecolor='Coral', alpha=0.5)
ax.plot(escala_temporal,completo[0:4345],color='k')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
#ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24,3)))

plt.savefig('DST.png', format='png', dpi=300)
plt.show()

#############################

import wget
from datetime import datetime,timedelta,date
import matplotlib.pyplot as plt
import datetime 
import matplotlib.dates as mdates
import numpy as np
import os
import gzip
import glob

lista=glog.glob('*.gz')

datos = []
for j in lista:
	f = gzip.open(j, 'rb')
	file_content = f.read()
	f.close()
	datos.extend(file_content)



list_orc=glob.glob('*.min')
list_orc=sorted(list_orc)

lines=file.readlines()
ind=[index for index in range(len(lines)) if lines[index][0:2] == 'IA']

##################################################################

orc_F=[]
time=[]

for i in range(len(list_orc)):
	time_str = np.genfromtxt(list_orc[i],dtype='str',usecols = (0,1) ,skip_header=22)
	datos = np.genfromtxt(list_orc[i],usecols = (6) ,skip_header=22)
	time.extend(time_str[1:])	
	orc_F.extend(datos[1:])

time_datetime=[]
for t in time:
	time_str=str(t)
	time_str=time_str[2:12]+' '+time_str[15:23]
	date_time_obj = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')	
	time_datetime.append(date_time_obj)

time_datetime=np.array(time_datetime)
orc_F=np.array(orc_F)


indices = [i for i, x in enumerate(orc_F) if x != 99999.0]

orc_FF=orc_F[indices]
time_datetime_F=time_datetime[indices]


l=len(time_datetime_F)
media=np.linspace(26408,26408,l)

fig,f=plt.subplots(1,sharex=True)
plt.style.use('default')
plt.rc('font', family='serif')
plt.title('Intensidad del campo magnético en Trelew (Magnétrometro)',fontsize=15)
#plt.suptitle('Figura generada por LAMP', fontsize=9,x=0.16)
f.plot(time_datetime_F,orc_FF,'k')
f.plot(time_datetime_F,media,'r',label='Media en días calmos')
f.set(ylabel='F (nT)')
f.set(xlabel='Fecha dd-mm (UT)')
f.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
f.legend(loc='best')

plt.savefig('trelew_.png', format='png', dpi=300)
plt.show()



#variacion 174 nT.

import pyIGRF
alt = 0
lat=-43
lon=-65

años = []
año=1900
while año<2025:
	años.append(año)
	año+=+10
		
trelew=[]

for l in años:
	intensidad= pyIGRF.igrf_value(lat, lon, alt,l)
	trelew.append(intensidad[-1])


fig,(f)=plt.subplots(1,sharex=True)
plt.style.use('default')
plt.rc('font', family='serif')
plt.title('Intensidad del campo magnético en Trelew (IGRF)',fontsize=15)
f.plot(años,trelew,'k')
f.plot(años[-3],trelew[-3],'rx',markersize=15,label='Año donde se seleccionó la tormenta')
f.set(ylabel='F (nT)')
f.set(xlabel='Año')
#f.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(loc='best')
plt.savefig('trelew_IGFR.png', format='png', dpi=300)
plt.show()



