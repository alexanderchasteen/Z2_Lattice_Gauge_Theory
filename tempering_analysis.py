import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv



SIZE=5
df=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT'+str(SIZE)+'_RawMC.csv')
beta_array=df.columns.to_numpy()
beta_array=beta_array.astype(float)
CONFIGS=len(beta_array)
volume=(SIZE**4)*6
beta_nbhd_radius=(beta_array[1]-beta_array[0])/2




def jacknife_variance(array):
    n=len(array)
    mean=np.mean(array)
    jacknife_means=[]
    for i in range(n):
        jacknife_sample=np.delete(array,i)
        jacknife_mean=np.mean(jacknife_sample)
        jacknife_means.append(jacknife_mean)
    jacknife_means=np.array(jacknife_means)
    variance=(n-1)/n * np.sum((jacknife_means - mean)**2)
    return variance



def blocking(array,blocks):
    size_of_array=len(array)
    block_size=int(size_of_array/blocks)
    blocked_array=[]
    for id in range(blocks):
        start=id*block_size
        end=start+block_size
        block=array[start:end]
        block_mean=np.mean(block)
        blocked_array.append(block_mean)
    return np.array(blocked_array)


def svendsen_blocking(array,weights,blocks):
    size_of_array=len(array)
    block_size=int(size_of_array/blocks)
    blocked_array=[]
    for id in range(blocks):
        start=id*block_size
        end=start+block_size
        block=array[start:end]
        block_weights=weights[start:end]
        normalize=np.sum(block_weights)
        block_mean=np.sum(block)/normalize
        blocked_array.append(block_mean)
    return np.array(blocked_array)


def jacknife_samples(array):
    jacknife_samples=[]
    for i in range(len(array)):
        v=0
        for j in range(len(array)):
            if i==j:
                v+=0
            else:
                v+=array[j]
        v=v/(len(array)-1)
        jacknife_samples.append(v)
    return jacknife_samples


blocks=50
avg_plaq=[]
plaq_sus=[]
SEM_avg_plaq=[]
SEM_plaq_sus=[]
svendsen_beta_array=[]
svendsen_plaq_array=[]
svendsen_sus_array=[]
svendsen_plaq_SEM_array=[]
svendsen_sus_SEM_array=[]

all_jackknife_samples = []
for i in range(CONFIGS):
    array=df.iloc[:,i].to_numpy()
    avg_plaq.append(np.mean(array))
    SEM_avg_plaq.append(2*np.sqrt(jacknife_variance(blocking(array,blocks))))
    plaq_sus.append(np.var(array))

    plaq_sus_measruments=[]
    for j in range(len(array)):
        plaq_sus_measruments.append((array[j]-np.mean(array))**2)
    SEM_plaq_sus.append(2*np.sqrt(jacknife_variance(blocking(plaq_sus_measruments,blocks))))
    svendsen_length=500
    for k in np.linspace(-beta_nbhd_radius, beta_nbhd_radius, 1000):
        svendsen_beta_array.append(beta_array[i]+k)
        delta_beta=k
        maxlogw=-1e308
        logw_array=[]
        for j in range(len(array)):
            logw=delta_beta*volume*array[j]
            logw_array.append(logw)
            if logw>maxlogw:
                maxlogw=logw
        wsum=0.0
        w_array=[]
        for j in range(len(array)):
            w=np.exp(logw_array[j]-maxlogw)
            w_array.append(w)
            wsum+=w



        avg_plq_svendsen_array=[]
        for j in range(len(array)):
            w=w_array[j]
            plaq=array[j]
            avg_plq_svendsen_array.append(w*plaq)
        avg_plq_svendsen=np.sum(avg_plq_svendsen_array)/wsum
        svendsen_plaq_array.append(avg_plq_svendsen)
        svendsen_plaq_SEM_array.append(2*np.sqrt(jacknife_variance(svendsen_blocking(avg_plq_svendsen_array,w_array,blocks))))

        plaq_sus_svendsen_array=[] 
        for j in range(len(array)):
            w=w_array[j]
            plaq=array[j]
            diff=plaq-avg_plq_svendsen
            plaq_sus_svendsen_array.append(w*diff*diff)
        plaq_sus_svendsen=np.sum(plaq_sus_svendsen_array)/wsum
        svendsen_sus_array.append(plaq_sus_svendsen)
        svendsen_sus_SEM_array.append(2*np.sqrt(jacknife_variance(svendsen_blocking(plaq_sus_svendsen_array,w_array,blocks))))

        jacknife_samples_at_svendesen_beta=jacknife_samples(svendsen_blocking(plaq_sus_svendsen_array,w_array,blocks))
        all_jackknife_samples.append(jacknife_samples_at_svendesen_beta)


plt.errorbar(beta_array,avg_plaq,yerr=SEM_avg_plaq,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_plaq_array,yerr=svendsen_plaq_SEM_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()
plt.errorbar(beta_array,plaq_sus,yerr=SEM_plaq_sus,fmt='o',label='Parallel Tempering Data',color='blue',ecolor='red',elinewidth=3)
plt.errorbar(svendsen_beta_array,svendsen_sus_array,yerr=svendsen_sus_SEM_array,fmt='o',label='Svendsen Reweighted Data',color='blue',ecolor='yellow',elinewidth=3,capsize=0,markersize=1)
plt.show()


montecarlo_data=np.transpose(np.array([beta_array,avg_plaq,plaq_sus,SEM_avg_plaq,SEM_plaq_sus]))
filename='PT'+str(SIZE)+'_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus", "Jackknife Action Var", "Jackknife Sus Var"]
with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(montecarlo_data)


print(f"CSV file '{filename}' created successfully.")


svendsen_data=np.transpose(np.array([svendsen_beta_array,svendsen_plaq_array,svendsen_sus_array,svendsen_plaq_SEM_array,svendsen_sus_SEM_array]))
filename2='PT'+str(SIZE)+'_svendsen_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus", "Jackknife Action Var", "Jackknife Sus Var"]
with open(filename2, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)

    # Write all rows at once
    csv_writer.writerow(header)
    csv_writer.writerows(svendsen_data)


print(f"CSV file '{filename2}' created successfully.")


jackknife_filename = 'PT'+str(SIZE)+'_jackknife_samples.csv'


jackknife_matrix = np.transpose(np.array(all_jackknife_samples))

# Create header row (beta values)
header = [f"{beta:.6f}" for beta in svendsen_beta_array]

with open(jackknife_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(jackknife_matrix)

print(f"CSV file '{jackknife_filename}' created successfully with betas as columns.")

