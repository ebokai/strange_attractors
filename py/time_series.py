import numpy as np 
import matplotlib.pyplot  as plt 
import scipy.fftpack
import warnings 
warnings.simplefilter('ignore')

fname = "d_5_728058021_arr"
data = np.load(f'./iteration_data/{fname}.npy')

# PERIOD FINDING 
# MINIMUM OF STANDARD DEVIATION OF DIFFERENCE 

plt.subplot(321)

mini = 0
minf = 1000

sd = []
mi = []

for i in range(500):
	dsel = data[1024:2048,0]
	diff = dsel[(i+1):] - dsel[:-(i+1)]
	mi.append(i+1)
	sd.append(np.std(diff))

	if np.std(diff) < minf:
		minf = np.std(diff)
		mini = (i+1)
	plt.plot(i+1,1/np.std(diff),'.',c='k')
plt.title('INVERSE STANDARD DEVIATION OF DIFFERENCE [ISDD]')

mi_s = [x for _,x in sorted(zip(sd,mi))]
print(mi_s)

# AUTO CORRELATION
plt.subplot(323)
for i in range(500):
	dsel1 = data[1024:2048,0] 
	dsel1 -= np.mean(dsel1)
	dsel2 = data[1024+i:2048+i,0]
	dsel2 -= np.mean(dsel2)
	acor = np.mean(dsel1*dsel2)
	plt.plot(i, acor, '.', c='k')
plt.title('AUTOCORRELATION')
plt.axvline(mini)

# FOURIER ANALYSIS
plt.subplot(325)
dsel = np.zeros(1024)
p = int(mi_s[0])
print(p)
for i in range(1024):
	dsel[i] = (i % p)
dsel -= np.mean(dsel)
dsel = data[1024:2048,0]

f = scipy.fftpack.fft(dsel)
x = np.linspace(0,1,512)
X = 2/x
p = X[np.argmax(abs(f)[:512])]
print(p,mini,mini/p)
plt.plot(X,np.abs(f)[:512])
plt.xscale('log')
plt.title('FOURIER SPECTRUm')
plt.axvline(mi_s[0])

plt.subplot(222)
plt.title('DECIMATED ATTRACTOR')
dsel = data[1024:80000:mi_s[0]]
plt.plot(dsel[:,2],dsel[:,1],'-',ms=1)

plt.subplot(224)
plt.title('ATTRACTOR')
plt.plot(data[1024:8192,2],data[1024:8192,1],'.',ms=1)


plt.show()
