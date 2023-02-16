import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as anim 

# FIND INTEGRAL CURVE FOR ITERATED MAP VECTOR FIELD

tmax = 100
res = 1000


# main equation
def f(x,y,coeff):

	newx = coeff[0] * x + coeff[1] * y**2 + coeff[2]
	newy = coeff[3] * x**2 + coeff[4] * y + coeff[5]

	return newx, newy

# arc length (not used)
def arclength(X,t):

	dx = X[0,1:,t] - X[0,:-1,t]
	dy = X[1,1:,t] - X[1,:-1,t]

	al = sum(np.sqrt(dx**2 + dy**2))

	return al 



evx = -0.595264
evy = -0.80353

# start from fixed point since dx/dt = 0 
x0, y0 = 1.317350919980067260665126, 1.049298041535435674996651
#x0, y0 = 1.31735, 1.04929

# equation coefficients
coeff = np.asarray([-0.2,  0.8,  0.7,  0.7,  0.7, -0.9])

# line from x0, y0 to f(x0), f(y0)
X0 = np.linspace(x0, f(x0, y0, coeff)[0], res)
Y0 = np.linspace(y0, f(x0, y0, coeff)[1], res)

# trajectory data
alldata = np.zeros((2, res, tmax+1))
alldata[0, :, 0] = X0
alldata[1, :, 0] = Y0

for t in range(tmax):

	# piece of arc at time t
	X,Y = alldata[0,:,t], alldata[1,:,t]

	# map arc to f(arc)
	for i,(x,y) in enumerate(zip(X,Y)):

		fx,fy = f(x,y,coeff)

		alldata[0, i, t+1] = fx
		alldata[1, i, t+1] = fy

	print(f'{t}, {arclength(alldata,t):.3f}')

animdata = np.zeros((2,res*(tmax+1)))

for t in range(tmax+1):
	animdata[0,t*res:(t+1)*res] = alldata[0,:,t]
	animdata[1,t*res:(t+1)*res] = alldata[1,:,t]

# FIND A WAY TO KEEP ARC LENGTH APPROX. CONSTANT THROUGHOUT RECONSTRUCTION

fig = plt.figure()
ax = plt.subplot(111,aspect='equal')
line, = ax.plot([], [], c='k', lw=1)

def init():
	ax.set_xlim(0,2)
	ax.set_ylim(-1.5,1.5)

	line.set_data([],[])

	return line, 

def animate(i):

	line.set_data(animdata[0,:i*10],animdata[1,:i*10])

	return line,

animation = anim.FuncAnimation(fig, animate, init_func=init, blit=True, interval=0)
plt.show()





