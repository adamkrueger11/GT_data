import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import itertools
from tqdm import tqdm
import sys 

from myfunctions import convert_data,plot_contour,cartesian,sym_reg,fit_lows,save_pickle,read_pickle,get_features_ID,new_get_all_fft_features
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)




top = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/ID/'

if 'all' in dates: dates = [i[:6] for i in os.listdir(top+time) if f'ID-{time}' in i]
obj = {'0hr':50,'4hr':10}[time]
all_data,meta={},pd.DataFrame()


for time in ['0hr','4hr']:
    main_root = top+time+'/'
    
    
    print('Assembling Data')
    for r,folds,files in os.walk(top+f'{time}/'):
        if not r.endswith('-'+time): continue
        date = r.split('/')[-1][:6]
    
        root = top + f'{time}/{date}_ID-{time}/'
        print(f'Gathering data from {date} - {time}')
        for file in tqdm(os.listdir(root)):
            if 'datx' not in file: continue
            raw,latres = convert_data(root+file,resolution=True,remove=False)
            if '50x' in file:
                c = fit_lows(raw,degree=1,N=3)
                #if np.abs(np.median(c.T[:5])-np.median(c.T[-5:]))>500:
                    
            else: #if '10x' in file:
                I,J = cartesian(raw) 
                masked = np.where(J>raw.shape[1]-100,raw,np.nan)
                c = sym_reg(masked,1,normal=True,full_image=raw)
            all_data['_'.join(file[:-5].split('_')+[date])] = c
                
            save_pickle(root+'all_data',all_data)
        clsi = {i:np.sum(np.array([j[0][0] for j in all_data.keys()])==str(i)) for i in range(3)}
        n_classes = len(clsi)-np.sum(np.array(clsi.values())==0)
        this_meta = pd.read_excel(top+f'METADATA/METADATA_{date}.xlsx',sheet_name='META')
        this_meta['Date'] = [date for row in range(this_meta.shape[0])]
        this_meta['Lateral Resolution'] = [latres for row in range(this_meta.shape[0])]
        meta = pd.concat([meta,this_meta])
        
        meta_dic = {sn: {i:d for i,d in zip(meta.columns,meta.loc[meta['Strain']==sn].values[0])} for sn in np.unique(meta['Strain'])}
    