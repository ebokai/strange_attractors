import numpy as np 
import matplotlib.pyplot as plt

tmax = 50
res = 2000


def f(x,y,coeff):

	newx = coeff[0] * x + coeff[1] * y**2 + coeff[2]
	newy = coeff[3] * x**2 + coeff[4] * y + coeff[5]

	return newx, newy

def arclength(X,t):

	dx = X[0,1:,t] - X[0,:-1,t]
	dy = X[1,1:,t] - X[1,:-1,t]

	al = sum(np.sqrt(dx**2 + dy**2))
	

	return al

evx = -0.595264
evy = -0.80353
x0,y0 = 1.317350919980067260665126, 1.049298041535435674996651
x0,y0 = 1.31735, 1.04929


# x0,y0 = 0.9777812789842297539272774,-0.7692021310919156774166995
# x0,y0 = 0.9777812,-0.7692021

coeff = np.asarray([-0.2,  0.8,  0.7,  0.7,  0.7, -0.9])
X0 = np.linspace(x0,f(x0,y0,coeff)[0],res)
Y0 = np.linspace(y0,f(x0,y0,coeff)[1],res)
alldata = np.zeros((2,res,tmax+1))
alldata[0,:,0] = X0
alldata[1,:,0] = Y0

for t in range(tmax):

	X,Y = alldata[0,:,t],alldata[1,:,t]

	for i,(x,y) in enumerate(zip(X,Y)):

		fx,fy = f(x,y,coeff)

		alldata[0,i,t+1] = fx
		alldata[1,i,t+1] = fy

	print(f'{t}, {arclength(alldata,t):.3f}')

ax = plt.subplot(111, aspect='equal')
ax.plot(alldata[0,:,:],alldata[1,:,:],'-',c='k',alpha=0.7)
ax.plot(alldata[0,0,:],alldata[1,0,:],'o',c='r',alpha=0.2)

plt.show()





