import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import os
from scipy.special import erf
from sklearn.metrics import r2_score

# read the data frame and extract the columns
#now #zooming in around beta=.4 to beta=.5
df=pd.read_csv('/home/alexa/Z2_Lattice/Cold/Size3/Lattice_Size3_TS10000_MS30000,AS1000.csv')
beta_array=df.loc[:,'Beta'].to_numpy()
avg_plaq_action=df.loc[:,'Avg Plaq'].to_numpy()
plaq_sus=df.loc[:,'Plaq Sus'].to_numpy()
action_var=df.loc[:,'Jacknif Action Var'].to_numpy()
sus_var=df.loc[:,'Jacknife Sus Var '].to_numpy()


df_prime=pd.read_csv('/home/alexa/Z2_Lattice/Cold/Size3/Swendsen and Ferrenberg Sweeps_Lattice_Size3_TS10000_MS30000,AS1000.csv')
beta_array_prime=df_prime.loc[:,'Beta'].to_numpy()
avg_plaq_action_prime=df_prime.loc[:,'Avg Plaq'].to_numpy()
plaq_sus_prime=df_prime.loc[:,'Plaq Sus'].to_numpy()
action_var_prime=df_prime.loc[:,'Jacknif Action Var'].to_numpy()
sus_var_prime=df_prime.loc[:,'Jacknife Sus Var '].to_numpy()

beta_array=np.concatenate((beta_array,beta_array_prime))
avg_plaq_action=np.concatenate((avg_plaq_action,avg_plaq_action_prime))
plaq_sus=np.concatenate((plaq_sus,plaq_sus_prime))
action_var=np.concatenate((action_var,action_var_prime))
sus_var=np.concatenate((sus_var,sus_var_prime))

action_var=2*np.sqrt(action_var)
sus_var=2*np.sqrt(sus_var)




df2=pd.read_csv('/home/alexa/Z2_Lattice/Cold/Size5/Swendsen and Ferrenberg Sweeps_Lattice_Size5_TS1000001_MS100000,AS10000.csv')
beta_array_svendesn=df2.loc[:,'beta'].to_numpy()
avg_plaq_svendesn=df2.loc[:,'avg_plaq'].to_numpy()
plaq_var_svendesn=df2.loc[:,'plaq_var'].to_numpy()
avg_plaq_sus=df2.loc[:,'avg_sus'].to_numpy()
sus_var_svendsen=df2.loc[:,'sus_var'].to_numpy()


df2_prime=pd.read_csv('/home/alexa/Z2_Lattice/Swendsen and Ferrenberg Sweeps_Lattice_Size5_TS5000002_MS100000,AS100000.csv')
beta_array_svendesn_prime=df2_prime.loc[:,'beta'].to_numpy()
avg_plaq_svendesn_prime=df2_prime.loc[:,'avg_plaq'].to_numpy()
plaq_var_svendesn_prime=df2_prime.loc[:,'plaq_var'].to_numpy()
avg_plaq_sus_prime=df2_prime.loc[:,'avg_sus'].to_numpy()
sus_var_svendsen_prime=df2_prime.loc[:,'sus_var'].to_numpy()


beta_array_svendesn=np.concatenate((beta_array_svendesn,beta_array_svendesn_prime))
avg_plaq_svendesn=np.concatenate((avg_plaq_svendesn,avg_plaq_svendesn_prime))
plaq_var_svendesn=np.concatenate((plaq_var_svendesn,plaq_var_svendesn_prime))
avg_plaq_sus=np.concatenate((avg_plaq_sus,avg_plaq_sus_prime))
sus_var_svendsen=np.concatenate((sus_var_svendsen,sus_var_svendsen_prime))

#Computing Standard Error for Error bars
plaq_var_svendesn=2*np.sqrt(plaq_var_svendesn)
sus_var_svendsen=2*np.sqrt(sus_var_svendsen)

# Visualize the Data

#without error bars
plt.figure()
plt.errorbar(beta_array_svendesn,avg_plaq_svendesn, yerr=plaq_var_svendesn, fmt=',',color='yellow')
plt.errorbar(beta_array,avg_plaq_action, yerr=action_var, fmt='o',color='red')
plt.errorbar(beta_array_svendesn,avg_plaq_svendesn, yerr=0, fmt=',',color='red')
plt.title('Average Plaquette vs Beta')
plt.xlabel('Beta')
plt.ylabel('Avg Plaq')
plt.show()

 



plt.figure()
plt.errorbar(beta_array_svendesn, avg_plaq_sus,yerr=sus_var_svendsen, fmt=',', color='yellow')
plt.errorbar(beta_array, plaq_sus, yerr=sus_var, fmt='o', color='red')
plt.errorbar(beta_array_svendesn, avg_plaq_sus,yerr=0, fmt=',', color='red')
plt.title('Plaquette Susceptibility vs Beta')
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')
plt.show()

# index=0
# max=0  
# for i in range(len(avg_plaq_sus)):
#     if avg_plaq_sus[i]>max:
#         max=avg_plaq_sus[i]
#         index=i

# print(beta_array_svendesn[index])

beta_total=beta_array_svendesn
plaq_sus_total=avg_plaq_sus

beta_min=.439
beta_max=.4393


beta_array_smaller=[]
plaq_sus_total_smaller=[]
for i in range(len(beta_total)):
    x=beta_total[i]
    if x>beta_min and x<beta_max:
        beta_array_smaller.append(x)
        plaq_sus_total_smaller.append(plaq_sus_total[i])

beta_total=beta_array_smaller
plaq_sus_total=plaq_sus_total_smaller




plt.scatter(beta_total,plaq_sus_total)


def lorentzian(x,h,x0,w):
    return h*w**2/((x-x0)**2+w**2)

cts_beta=np.arange(0.42,.45,.001).tolist()


param,_=curve_fit(lorentzian,beta_total, plaq_sus_total,p0=[0.05,.43,0.01])
fitA,fitB,fitC=param
fitY=lorentzian(cts_beta,fitA,fitB,fitC)
x=r2_score(plaq_sus,lorentzian(beta_array,fitA,fitB,fitC))
y=round(x,4)
plt.figure()
plt.errorbar(beta_total,plaq_sus_total, yerr=0,fmt='o', color='red')

# print('x0 Gaussian='+str(fit_B)+" R^2="+str(r2_score(plaq_sus,gauss(beta_array,fit_A,fit_B,fit_C,fit_D))))  
print('x0 Lorentzian='+str(fitB)+" R^2="+str(r2_score(plaq_sus,lorentzian(beta_array,fitA,fitB,fitC)))+' FWHF='+str(2*np.abs(fitC))+ ' Peak='+str(fitA))
# plt.plot(cts_beta, fit_y, '-', label='Gaussian Fit')
plt.plot(cts_beta, fitY, '-', label='Lorentzian Fit R^2='+str(y),color='blue',alpha=0.5)
plt.xlabel('Beta')
plt.ylabel('Plaquette Susceptibility')
plt.title('Plaquette Susceptibility vs Beta')
plt.legend()
plt.grid(True)
plt.show()



