import numpy as np
import matplotlib.pyplot as plt

def genDataSet(N):
    x=np.random.normal(0,1,N)
    ytrue=(np.cos(x)+2)/(np.cos(x*1.4)+2)
    noise=np.random.normal(0,0.2,N)
    y=ytrue+noise
    return x,y,ytrue

x,y,ytrue=genDataSet(1000)
#plt.plot(x,y,'y.')
#plt.plot(x,ytrue,'kx')
#plt.show()