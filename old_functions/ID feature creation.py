import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import itertools
from tqdm import tqdm
import sys 
sys.path.append(os.path.dirname('__file__'))
from functions import convert_data,plot_contour,cartesian,sym_reg,fit_lows,save_pickle,read_pickle,get_features_ID,new_get_all_fft_features
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)


##Collection/Data inputs
dates = ['all']
time = '4hr'






plt.close('all')
top = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/ID/'

if 'all' in dates: dates = [i[:6] for i in os.listdir(top+time) if f'ID-{time}' in i]
obj = {'0hr':50,'4hr':10}[time]
all_data,meta={},pd.DataFrame()


for time in ['0hr','4hr']:
    main_root = top+time+'/'
    
    print('Assembling Data')
    this_time_data, meta_dic = {}, {}
    for r,folds,files in os.walk(top+f'{time}/'):
        if not r.endswith('-'+time): continue
        date = r.split('/')[-1][:6]
    
        root = top + f'{time}/{date}_ID-{time}/'
        print(f'Gathering data from {date} - {time}')
        this_data = {}
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
            this_data['_'.join(file[:-5].split('_')+[date])] = c
                
            save_pickle(root+'all_data',this_data)

        this_time_data = this_time_data | this_data
        clsi = {i:np.sum(np.array([j[0][0] for j in all_data.keys()])==str(i)) for i in range(3)}
        n_classes = len(clsi)-np.sum(np.array(clsi.values())==0)
        this_meta = pd.read_excel(top+f'METADATA/METADATA_{date}.xlsx',sheet_name='META')
        this_meta['Date'] = [date for row in range(this_meta.shape[0])]
        this_meta['Lateral Resolution'] = [latres for row in range(this_meta.shape[0])]
        meta = pd.concat([meta,this_meta])
        
        meta_dic = meta_dic | {sn: {i:d for i,d in zip(this_meta.columns,this_meta.loc[this_meta['Strain']==sn].values[0])} for sn in this_meta['Strain'].unique()}
        
        
    all_data = all_data | this_time_data
    save_pickle(main_root+'all_data',this_time_data)

    
    print('Creating features')
    feature_names = get_features_ID(None,latres=None,region='edge',fftval=None)
    feat_df = pd.DataFrame(columns = feature_names)
    rows = []
    for i,d in tqdm(this_time_data.items()):
        feats={}
        i = tuple(i.split('_'))
        this_meta = meta_dic[int(i[0])].copy()
        this_meta['Filename'] = '_'.join(i)
        this_meta['Replicates'] = i[0]
        latres = this_meta['Lateral Resolution']
        rows.append(this_meta)
        
        try:
            height_feats = get_features_ID(d,latres*1e6,region='edge', obj=5,fftval=None)
            feats = feats|{i:[d] for i,d in height_feats.items()}
            
            fft_feats = new_get_all_fft_features(d,latres*1e6, region='edge', wavelengths=10)
            feats = feats|{i:[d] for i,d in fft_feats.items()}
        except Exception as err:
            print(f'Error processing {this_meta["Filename"]}, skipping...')
            failures += 1
            print(err)
            continue

        feat_df = pd.concat([feat_df,pd.DataFrame.from_dict(feats,orient='columns')],ignore_index = True,axis=0)
    
    feat_df = pd.concat([feat_df,pd.DataFrame(rows,columns=this_meta.keys())],ignore_index=False,axis=1)
    #Create test(s) for usability of data
    feat_df['Usable'] = [True for i in feat_df['Strain']]
    strains = np.unique(feat_df['Strain'])
    for strain in strains:
        mask = feat_df['Strain'] == strain
        test_df = feat_df.loc[mask].copy()
        #test ring calculation
        test = all(np.sum([np.greater(start,peak) for peak,start in test_df[['R-peak','R-start']].values],axis=1)==0)
        feat_df.loc[mask,'Usable'] = feat_df['Usable'].loc[mask] & np.bool_(np.ones(np.sum(mask))*test)
    save_pickle(main_root+'features',feat_df)
    fast_features = True

