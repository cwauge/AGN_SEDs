import numpy as np 

def bootstrap2D(x,y,frac,N):

	x = np.asarray(x)
	y = np.asarray(y)

	y_out, x_out = [], []

	sort = x.argsort()
	x = x[sort]
	y = y[sort]

	for i in range(N):

		ind = np.arange(len(x))

		np.random.shuffle(ind)

		x_sample = x[ind[0:int(len(ind)*frac)]]
		y_sample = y[ind[0:int(len(ind)*frac)]]

		z = np.polyfit(x_sample,y_sample,1)
		p = np.poly1d(z)

		y_out.append(p(x))
		x_out.append(x)

	return x_out, y_out
		

