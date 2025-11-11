import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score


SIZE=5

df=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT'+str(SIZE)+'_analysis.csv')
beta_array=df['Beta'].to_numpy()
avg_plaq=df['Avg Plaq'].to_numpy()
plaq_sus=df['Plaq Sus'].to_numpy()
SEM_avg_plaq=df['Jackknife Action Var'].to_numpy()
SEM_plaq_sus=df['Jackknife Sus Var'].to_numpy()

df2=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT'+str(SIZE)+'_svendsen_analysis.csv')
svendsen_beta_array=df2['Beta'].to_numpy()
svendsen_plaq_array=df2['Avg Plaq'].to_numpy()          
svendsen_sus_array=df2['Plaq Sus'].to_numpy()
svendsen_plaq_SEM_array=df2['Jackknife Action Var'].to_numpy()
svendsen_sus_SEM_array=df2['Jackknife Sus Var'].to_numpy()


jacknife_data_array = np.loadtxt('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT'+str(SIZE)+'_jackknife_samples.csv', delimiter=',')
beta_array_jacknife=(jacknife_data_array[0,:])


plt.rc('font', size=15)
plt.errorbar(beta_array,avg_plaq,yerr=SEM_avg_plaq,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_plaq_array,yerr=svendsen_plaq_SEM_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.title('Average Plaquette vs Beta')
plt.xlabel('Beta')
plt.ylabel('Average Plaquette')
plt.show()




beta_max=0.45
beta_min=0.43

mask = (svendsen_beta_array >= beta_min) & (svendsen_beta_array <= beta_max)

svendsen_beta_array     = svendsen_beta_array[mask]
svendsen_sus_array      = svendsen_sus_array[mask]
svendsen_sus_SEM_array  = svendsen_sus_SEM_array[mask]

def lorentzian(x, x0, gamma, a, d):
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + d
initial_guess=[0.439, 0.005, 0.04, 0.014]
param,_=curve_fit(lorentzian,svendsen_beta_array, svendsen_sus_array,p0 = initial_guess )
fitA,fitB,fitC,fitD=param
R2=r2_score(svendsen_sus_array,lorentzian(svendsen_beta_array,fitA,fitB,fitC,fitD))
R2=round(R2,4)
x0=fitA
gamma=fitB
a=fitC

cts_beta=np.arange(beta_min,beta_max,.001)
fitY=lorentzian(cts_beta,fitA,fitB,fitC,fitD)

plt.rc('font', size=15)
plt.errorbar(beta_array,plaq_sus,yerr=SEM_plaq_sus,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SEM_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.plot(cts_beta, fitY, '-', label='Lorentzian Fit R^2='+str(R2),color='blue',alpha=0.5)
plt.legend()
plt.title('Plaquette Susceptibility vs Beta')
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')  
plt.show()

critical_beta_array = []
FWHM_array = []
peak_height_array = []
for i in range(1, jacknife_data_array.shape[0]):
    param,_=curve_fit(lorentzian,beta_array_jacknife, jacknife_data_array[i,:],p0=initial_guess)
    fitA,fitB,fitC,fitD=param
    critical_betas = fitA
    FWHM = 2*fitB
    peak_height = fitC
    critical_beta_array.append(critical_betas)
    FWHM_array.append(FWHM)
    peak_height_array.append(peak_height)

def variance(data,mean):
    N=len(data)
    var_sum=0
    for i in range(N):
        var_sum+=(data[i]-mean)**2
    variance= (N-1)/N * var_sum
    return variance

SEM_critical_beta=2*np.sqrt(variance(critical_beta_array,np.mean(critical_beta_array)))
SEM_FWHM=2*np.sqrt(variance(FWHM_array,np.mean(FWHM_array)))
SEM_peak_height=np.sqrt(variance(peak_height_array,np.mean(peak_height_array)))

print('Critical Beta:', np.mean(critical_beta_array), '+/-', SEM_critical_beta)
print('FWHM:', np.mean(FWHM_array), '+/-', SEM_FWHM)
print('Peak Height:', np.mean(peak_height_array), '+/-', SEM_peak_height)