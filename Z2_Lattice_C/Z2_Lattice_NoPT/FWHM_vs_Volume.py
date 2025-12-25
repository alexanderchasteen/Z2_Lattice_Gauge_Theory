import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score

size_array=[3,4,5,6,7,8,9]
inverse_volume=[1/(x**4) for x in size_array]


FWHM_array=[.018011084,
0.004134116,
0.001863627,
0.000983935,
0.000551401,
0.000334577957361284,
0.000212754]

inverse_volume=np.array(inverse_volume)
FWHM_array=np.array(FWHM_array)

def linear(x,m,b):
    return m*x+b


cts_inverse_volume = np.linspace(np.min(inverse_volume), np.max(inverse_volume), 500)

param,_=curve_fit(linear,inverse_volume, FWHM_array)
fitA,fitB=param

x=r2_score(FWHM_array,linear(inverse_volume,fitA,fitB))
y=round(x,4)
fitY=linear(cts_inverse_volume,fitA,fitB)
plt.figure()
plt.scatter(inverse_volume,FWHM_array,color='blue', label='Data')
plt.plot(cts_inverse_volume, fitY, '-', label='Linear Fit R^2='+str(y),color='blue' )
plt.title('1/Volume vs Plaquette Susceptibility Width')
plt.xlabel('1/Volume')
plt.ylabel('FWHM of Plaquette Susceptibility')
plt.legend()
plt.show()


# Do it without 3 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score

size_array=[4,5,6,7,8,9]
inverse_volume=[1/(x**4) for x in size_array]


FWHM_array=[
0.004134116,
0.001863627,
0.000983935,
0.000551401,
0.000334577957361284,
0.000212754]

inverse_volume=np.array(inverse_volume)
FWHM_array=np.array(FWHM_array)

def linear(x,m,b):
    return m*x+b


cts_inverse_volume = np.linspace(np.min(inverse_volume), np.max(inverse_volume), 500)

param,_=curve_fit(linear,inverse_volume, FWHM_array)
fitA,fitB=param

x=r2_score(FWHM_array,linear(inverse_volume,fitA,fitB))
y=round(x,4)
fitY=linear(cts_inverse_volume,fitA,fitB)
plt.figure()
plt.scatter(inverse_volume,FWHM_array,color='blue', label='Data')
plt.plot(cts_inverse_volume, fitY, '-', label='Linear Fit R^2='+str(y),color='blue' )
plt.title('1/Volume vs Plaquette Susceptibility Width (Without Size 3)')
plt.xlabel('1/Volume')
plt.ylabel('FWHM of Plaquette Susceptibility')
plt.legend()
plt.show()

