import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score


Beta_lower=.4
Beta_higher=.46


df=pd.read_csv('/home/alexa/Z2_Lattice/Susceptibility_Jacknife_Samples_Lattice_Size7_TS1000_MS100000,AS2000.csv')
array=df.to_numpy()

beta_array=array[:,0]
for i in range(len(beta_array)):
    if beta_array[i]>Beta_lower:
        index_lower=i
        break



beta_array=array[:,0]
for i in range(len(beta_array)):
    if beta_array[i]>Beta_higher:
        index_higher=i-1
        break

beta_array=array[index_lower:index_higher,0]
plt.plot(beta_array,array[index_lower:index_higher,1])
plt.show()


blocks=50 

def lorentzian(x,h,x0,w):
    return h*w**2/((x-x0)**2+w**2)

FWHM_jack_array=[]
for i in range(1,blocks+1):
    param,_=curve_fit(lorentzian,beta_array, array[index_lower:index_higher,i],p0=[0.04, 0.43, 0.005])
    fitA,fitB,fitC=param
    FWHM_jack_array.append(2*fitC)
    print(param)

print(FWHM_jack_array)
print(2*np.sqrt(np.var(FWHM_jack_array)))