import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score


df=pd.read_csv('Sus_Peak_Analysis.csv')
Lattice_Size = df['Lattice_Size'].to_numpy()
Critical_Beta = df['Critical_Beta'].to_numpy()
FWHM = df['FWHM'].to_numpy()
Peak = df['Peak'].to_numpy()
Critical_Beta_SD = df['Critical_Beta_SD'].to_numpy()
FWHM_SD = df['FWHM_SD'].to_numpy()
Peak_SD = df['Peak_SD'].to_numpy()



def linear(x,a,b):
    return a*x+b



volume = np.array([x**4 for x in Lattice_Size], dtype=float)
inverse_volume = np.array([1/x for x in volume], dtype=float)




param,pcov=curve_fit(linear,inverse_volume, Critical_Beta,sigma=Critical_Beta_SD, absolute_sigma=False )

fitA,fitB=param

perr = np.sqrt(np.diag(pcov))
print('optimal paramters: slope,intercept', fitA,fitB)
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

chi_squared = np.sum(((Critical_Beta - y_fit) / Critical_Beta_SD)**2)
print("Chi squared:", chi_squared)


R2=r2_score(Critical_Beta,linear(inverse_volume,fitA,fitB))
R2=round(R2,4)
plt.errorbar(inverse_volume,Critical_Beta,yerr=Critical_Beta_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'Critical $\beta_c$ ')
plt.plot(inverse_volume,linear(inverse_volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Inverse volume', fontsize=12)
plt.ylabel(r'Critical $\beta_c$', fontsize=12)
plt.title('Critical β vs Inverse volume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()





param,pcov=curve_fit(linear,inverse_volume, FWHM,sigma=FWHM_SD, absolute_sigma=False)
fitA,fitB=param



perr = np.sqrt(np.diag(pcov))
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

print(y_fit)


chi_squared = np.sum(((FWHM - y_fit) / FWHM_SD)**2)
print("Chi squared:", chi_squared)

R2=r2_score(FWHM,linear(inverse_volume,fitA,fitB))
R2=round(R2,4)
plt.errorbar(inverse_volume,FWHM,yerr=FWHM_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'Critical $\beta_c$')
plt.plot(inverse_volume,linear(inverse_volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Inverse volume', fontsize=12)
plt.ylabel('FWHM', fontsize=12)
plt.title(' FWHM vs Inversevolume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()



param,pcov=curve_fit(linear,volume, Peak)
fitA,fitB=param
perr = np.sqrt(np.diag(pcov))
print("Standard errors of parameters:", perr)
y_fit=linear(volume,fitA,fitB)

print(y_fit)


chi_squared = np.sum(((Peak - y_fit) / Peak_SD)**2)
print("Chi squared:", chi_squared)

R2=r2_score(Peak,linear(volume,fitA,fitB))
R2=round(R2,4)
plt.errorbar(volume,Peak,yerr=Peak_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'Critical $\beta_c$')
plt.plot(volume,linear(volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Volume', fontsize=12)
plt.ylabel('Peak', fontsize=12)
plt.title('Peak vs Volume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()

