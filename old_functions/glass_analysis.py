import numpy as np 
import sys 
sys.path.append('/Users/adamkrueger/Documents/GitHub_repos/sample_processing')
from myfunctions import plot_df, convert_data, clean_df, plot_features
import os
import pandas as pd
import matplotlib.pyplot as plt



root ='/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Yunker Lab/glass/230711_glass slide cell counts'

ddic = {i[:-12]: convert_data(os.path.join(root,i),remove=False) for i in os.listdir(root) if '.datx' in i and 'fill' in i}
df = pd.DataFrame({
    'Dilution':[i[0] for i in ddic.keys()],
    'Replicate':[i[-1] for i in ddic.keys()],
    'Strain ID':[i[2] for i in ddic.keys()],
    'Data-array': [i['Heights'] for i in ddic.values()],
    'Folder': ['hey' for i in range(len(ddic))],
    'FileBase':[i for i in range(len(ddic))],
    'Treatment': ['glass' for i in range(len(ddic))],
    'Lateral Resolution': [i['Resolution'].item() for i in ddic.values()]
    })

df = clean_df(df,response_init='lows',degree_of_fit=2)
df['Volume'] = df['Cleaned-Data-array'].apply(lambda x: np.sum(x)/1000)
plot_features(df, y_data='Volume',x_data='Dilution', scale=True, color='Strain ID')
plt.yscale('log')

plot_df(df, close=False, data_column='Cleaned-Data-array',split='Strain ID',sort=['Dilution','Replicate'],vlims=[-50,100])
plt.show()