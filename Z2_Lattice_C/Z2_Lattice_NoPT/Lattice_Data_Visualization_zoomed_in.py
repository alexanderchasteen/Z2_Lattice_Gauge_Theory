import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score


Beta_lower=.45
Beta_higher=.455



# read the data frame and extract the columns
df=pd.read_csv('/home/alexa/Z2_Lattice/Lattice_Size6_TS1000_MS10000,AS2000.csv')
beta_array=df.loc[:,'Beta'].to_numpy()
avg_plaq_action=df.loc[:,'Avg Plaq'].to_numpy()
plaq_sus=df.loc[:,'Plaq Sus'].to_numpy()
action_var=df.loc[:,'Jacknif Action Var'].to_numpy()
sus_var=df.loc[:,'Jacknife Sus Var '].to_numpy()

action_var=2*np.sqrt(action_var)
sus_var=2*np.sqrt(sus_var)

beta_array_zoomed_in=[]
plaq_sus_zoomed_in=[]
sus_var_zoomed_in=[]



for i in range(len(beta_array)):
    x=beta_array[i]
    if x>Beta_lower and x<Beta_higher:
       beta_array_zoomed_in.append(x) 
       plaq_sus_zoomed_in.append(plaq_sus[i])
       sus_var_zoomed_in.append(sus_var[i])


df2=pd.read_csv('/home/alexa/Z2_Lattice/Swendsen and Ferrenberg Sweeps_Lattice_Size6_TS1000_MS10000,AS2000.csv')
beta_array_svendesn=df2.loc[:,'beta'].to_numpy()
avg_plaq_svendesn=df2.loc[:,'avg_plaq'].to_numpy()
plaq_var_svendesn=df2.loc[:,'plaq_var'].to_numpy()
avg_plaq_sus=df2.loc[:,'avg_sus'].to_numpy()
sus_var_svendsen=df2.loc[:,'sus_var'].to_numpy()

#Computing Standard Error for Error bars
plaq_var_svendesn=2*np.sqrt(plaq_var_svendesn)
sus_var_svendsen=2*np.sqrt(sus_var_svendsen)

beta_array_svendesn_zoomed_in=[]
avg_plaq_sus_svendsen_zoomed_in=[]
sus_var_svendesen_zoomed_in=[]


for i in range(len(beta_array_svendesn)):
    x=beta_array_svendesn[i]
    if x>Beta_lower and x<Beta_higher:
       beta_array_svendesn_zoomed_in.append(beta_array_svendesn[i]) 
       avg_plaq_sus_svendsen_zoomed_in.append(avg_plaq_sus[i])
       sus_var_svendesen_zoomed_in.append(sus_var_svendsen[i])


plt.figure()
plt.errorbar(beta_array_svendesn_zoomed_in,avg_plaq_sus_svendsen_zoomed_in, yerr=sus_var_svendesen_zoomed_in, fmt=',',color='yellow')
plt.errorbar(beta_array_zoomed_in,plaq_sus_zoomed_in, yerr=sus_var_zoomed_in, fmt='o',color='red')
plt.errorbar(beta_array_svendesn_zoomed_in,avg_plaq_sus_svendsen_zoomed_in, yerr=0, fmt=',',color='red')
plt.title('Average Plaquette vs Beta')
plt.xlabel('Beta')
plt.ylabel('Avg Plaq')
plt.show()

 

beta_total=beta_array_svendesn_zoomed_in
plaq_sus_total=avg_plaq_sus_svendsen_zoomed_in


def lorentzian(x,h,x0,w):
    return h*w**2/((x-x0)**2+w**2)

cts_beta=np.arange(Beta_lower,Beta_higher,.001)

initial_guess=[0.043, 0.431, 0.0005]
param,_=curve_fit(lorentzian,beta_total, plaq_sus_total,p0=initial_guess)
fitA,fitB,fitC=param
fitY=lorentzian(cts_beta,fitA,fitB,fitC)
x=r2_score(plaq_sus_total,lorentzian(beta_total,fitA,fitB,fitC))
y=round(x,4)
plt.figure()
plt.errorbar(beta_total,plaq_sus_total, yerr=sus_var_svendesen_zoomed_in, fmt='o',color='red')
plt.errorbar(beta_total,plaq_sus_total, yerr=0, fmt=',',color='red') 
print('x0 Lorentzian='+str(fitB)+" R^2="+str(r2_score(plaq_sus_total,lorentzian(beta_total,fitA,fitB,fitC))))
plt.plot(cts_beta, fitY, '-', label='Lorentzian Fit R^2='+str(y),color='blue',alpha=0.5)
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')
plt.title('Plaquette Susceptibility vs Beta')
plt.legend()
plt.grid(True)
plt.show()




def parabola(x,a,b,c):
    return a*(x-b)**2+c

cts_beta=np.arange(Beta_lower,Beta_higher,.001)

initial_guess = [-0.0005, 0.431, 0.043]
parameters,_=curve_fit(parabola,beta_total, plaq_sus_total,p0=initial_guess)
fit_A,fit_B,fit_C=parameters
print(parameters)
fit_Y=parabola(cts_beta,fit_A,fit_B,fit_C)
x=r2_score(plaq_sus_total,parabola(beta_total,fit_A,fit_B,fit_C))
y=round(x,4)
plt.figure()
# plt.errorbar(beta_total,plaq_sus_total, yerr=sus_var_svendesen_zoomed_in, fmt='o',color='red')
plt.errorbar(beta_total,plaq_sus_total, yerr=0, fmt=',',color='red') 
print('x0 Parabolic='+str(fitB)+" R^2="+str(r2_score(plaq_sus_total,parabola(beta_total,fit_A,fit_B,fit_C))))
plt.plot(cts_beta, fit_Y, '-', label='Parabolic Fit R^2='+str(y),color='blue',alpha=0.5)
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')
plt.title('Plaquette Susceptibility vs Beta')
plt.legend()
plt.grid(True)

plt.show()





# def gauss(x,A, x0, sigma,H):
#     return H+A * np.exp(-(x - x0)**2 / (2 * sigma**2))


# initial_guess = [0.0034, 0.431, 0.0005, 0.040]
# parameters, _ = curve_fit(gauss, beta_total, plaq_sus_total,p0=initial_guess)
# fitA, fitB, fitC,fitD= parameters
# fitY = gauss(cts_beta, fitA, fitB, fitC,fitD)
# x=r2_score(plaq_sus,gauss(beta_array,fitA,fitB,fitC,fitD))
# y=round(x,4)
# plt.figure()
# plt.errorbar(beta_array_zoomed_in,plaq_sus_zoomed_in, yerr=sus_var_zoomed_in, fmt='o',color='red')
# plt.errorbar(beta_array_svendesn_zoomed_in,avg_plaq_sus_svendsen_zoomed_in, yerr=0, fmt=',',color='red') 
# print('x0 Gauss='+str(fitB)+" R^2="+str(r2_score(plaq_sus,gauss(beta_array,fitA,fitB,fitC,fitD))))
# plt.plot(cts_beta, fitY, '-', label='Gauss Fit R^2='+str(y),color='blue',alpha=0.5)
# plt.xlabel('Beta')
# plt.ylabel('Plaquette Susceptibility')
# plt.title('Plaquette Susceptibility vs Beta')
# plt.legend()
# plt.grid(True)
# plt.show()


