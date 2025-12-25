import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.special import erf
from sklearn.metrics import r2_score

df=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/historgam_phase_trans_left_data copy.csv')
Lattice_Size_l = df['Lattice_Size'].to_numpy()
Histogram_Peak_l = df['Histogram_Peak'].to_numpy()
Histogram_mean_l = df['Histogram_mean'].to_numpy()
Histogtam_FWHM_l = df['Histogtam_FWHM'].to_numpy()
Peak_SD_l = df['Peak_SD'].to_numpy()
Mean_SD_l = df['Mean_SD'].to_numpy()
FWHM_SD_l = df['FWHM_SD'].to_numpy()





df=pd.read_csv('historgam_phase_trans_right_data.csv')
Lattice_Size_r = df['Lattice_Size'].to_numpy()
Histogram_Peak_r = df['Histogram_Peak'].to_numpy()
Histogram_mean_r = df['Histogram_mean'].to_numpy()
Histogtam_FWHM_r = df['Histogtam_FWHM'].to_numpy()
Peak_SD_r = df['Peak_SD'].to_numpy()
Mean_SD_r = df['Mean_SD'].to_numpy()
FWHM_SD_r = df['FWHM_SD'].to_numpy()


Lattice_Size=(Lattice_Size_r+Lattice_Size_l)/2
Histogram_Peak=(Histogram_Peak_l+Histogram_Peak_r)/2
Histogram_mean=(Histogram_mean_l+Histogram_mean_r)/2
Histogtam_FWHM=(Histogtam_FWHM_l+Histogtam_FWHM_r)/2
Peak_SD=(Peak_SD_l+Peak_SD_r)/2
Mean_SD=(Mean_SD_l+Mean_SD_r)/2
FWHM_SD=(FWHM_SD_l+FWHM_SD_r)/2




def linear(x,a,b):
    return a*x+b



volume = np.array([x**4 for x in Lattice_Size], dtype=float)
inverse_volume = np.array([1/x for x in volume], dtype=float)




param,pcov=curve_fit(linear,volume,Histogram_Peak,sigma=Peak_SD, absolute_sigma=False )

fitA,fitB=param

perr = np.sqrt(np.diag(pcov))
print('optimal paramters: slope,intercept', fitA,fitB)
print("Standard errors of parameters:", perr)
y_fit=linear(volume,fitA,fitB)

chi_squared = np.sum(((Histogram_Peak - y_fit) / Peak_SD)**2)
print("Chi squared:", chi_squared)


R2=r2_score(Histogram_Peak,linear(volume,fitA,fitB))
R2=round(R2,4)

plt.errorbar(volume,Histogram_Peak,yerr=Peak_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'Peak ')
plt.plot(volume,linear(volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Inverse volume', fontsize=12)
plt.ylabel(r'Peak Height', fontsize=12)
plt.title('Peak Height Inverse volume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()




param,pcov=curve_fit(linear,inverse_volume,Histogram_mean,sigma=Mean_SD, absolute_sigma=False )

fitA,fitB=param

perr = np.sqrt(np.diag(pcov))
print('optimal paramters: slope,intercept', fitA,fitB)
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

chi_squared = np.sum(((Histogram_mean - y_fit) / Mean_SD)**2)
print("Chi squared:", chi_squared)


R2=r2_score(Histogram_mean,linear(inverse_volume,fitA,fitB))
R2=round(R2,4)

plt.errorbar(inverse_volume,Histogram_mean,yerr=Mean_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'Mean')
plt.plot(inverse_volume,linear(inverse_volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Inverse volume', fontsize=12)
plt.ylabel(r'Mean', fontsize=12)
plt.title('Mean vs Inverse volume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()



param,pcov=curve_fit(linear,inverse_volume,Histogtam_FWHM,sigma=FWHM_SD, absolute_sigma=False )

fitA,fitB=param

perr = np.sqrt(np.diag(pcov))
print('optimal paramters: slope,intercept', fitA,fitB)
print("Standard errors of parameters:", perr)
y_fit=linear(inverse_volume,fitA,fitB)

chi_squared = np.sum(((Histogtam_FWHM - y_fit) / FWHM_SD)**2)
print("Chi squared:", chi_squared)


R2=r2_score(Histogtam_FWHM,linear(inverse_volume,fitA,fitB))
R2=round(R2,4)

plt.errorbar(inverse_volume,Histogtam_FWHM,yerr=FWHM_SD, fmt='o', capsize=4,elinewidth=1,markersize=6, color='teal',ecolor='gray', label=r'FWHM')
plt.plot(inverse_volume,linear(inverse_volume,fitA,fitB),linestyle="-",label=fr'Line of Best Fit: $R^2 = {R2}$')
plt.xlabel('Inverse volume', fontsize=12)
plt.ylabel(r'FWHM', fontsize=12)
plt.title('FWHM vs Inverse volume', fontsize=13, pad=10)
plt.legend(frameon=True, loc='best')
plt.show()