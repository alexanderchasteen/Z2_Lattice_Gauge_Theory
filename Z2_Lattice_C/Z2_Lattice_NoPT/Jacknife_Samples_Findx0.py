import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score

Beta_lower=.43896
Beta_higher=0.43933

# Beta_lower=.435
# Beta_higher=0.445


# df=pd.read_csv('Cold/Size4/Susceptibility_Jacknife_Samples_Lattice_Size4_TS10000_MS30000,AS1000.csv')
df=pd.read_csv('/home/alexa/Z2_Lattice/Susceptibility_Jacknife_Samples_Lattice_Size5_TS5000000_MS1000000,AS100000 copy.csv')
array=df.to_numpy()


beta_array=array[:,0]
for i in range(len(beta_array)):
    if beta_array[i]>Beta_lower:
        index_lower=i
        break

for i in range(len(beta_array)):
    if beta_array[i]>Beta_higher:
        index_higher=i-1
        break

beta_array=array[index_lower:index_higher,0]

blocks=50 
# number of blocks set in C code

# def largest(arr):
#     max = arr[0]
#     for i in range(1, len(arr)):
#         if arr[i] > max:
#             max = arr[i]
#             j=i
#     return max

def lorentzian(x,h,x0,w):
    return h*w**2/((x-x0)**2+w**2)

beta_critical_jacknife_samples=[]
FWHM=[]
height=[]
for i in range(1,blocks+1):
    ithrow = array[index_lower:index_higher, i] 
    param,_=curve_fit(lorentzian,beta_array, ithrow,p0=[0.05,.43,0.01])
    fitA,fitB,fitC=param
    beta_critical_jacknife_samples.append(fitB)
    FWHM.append(2*fitC)
    height.append(fitA)


diff=0
for x in height:
    diff+=(0.046925145
-x)**2

diff=diff/len(height)

print(2*np.sqrt(diff))



