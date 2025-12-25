import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SIZE=6
df=pd.read_csv('/home/alexa/Z2_Lattice/Z2_Lattice_PT/PT'+str(SIZE)+'_RawMC.csv')

beta_array=df.columns.to_numpy()
colors = ['skyblue', 'salmon', 'lightgreen', 'violet', 'gold']
for i in range(len(beta_array)):
    x=beta_array[i]
    histogram_data=df[x]
    plt.hist(histogram_data,bins=100,edgecolor='black',color=colors[i % len(colors)], alpha=0.7, label=x)

plt.show()