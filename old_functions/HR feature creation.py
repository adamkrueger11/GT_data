import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import itertools
from tqdm import tqdm
import sys 
sys.path.append(os.path.dirname('__file__'))
from functions import convert_data,get_corners,plot_contour,cartesian,sym_reg,fit_lows,save_pickle,read_pickle,get_features_ID,new_get_all_fft_features,plot_all
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)

##Collection/Data inputs
dates = ['all']
fast_features = False
fast_images   = False
get_images = True
include_ffts = True
plot_ffts  = True

##ML inputs
classify_by = 'Group'
min_count = 5
show_wrongs = True
show_fails  = True
svm = True
forest = False
pairwise = False
include_fails=True





plt.close('all')

all_data,meta={},pd.DataFrame()
cmap = lambda a: ['blue','red','black'][a]#plt.get_cmap('tab10')

meta = pd.DataFrame()
top = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/HR/'
    
for root,folds,files in os.walk(top):
    if '1-per-agar' not in root: continue
    if 'filled' != root.split('/')[-1]: continue
    date = root.split('/')[-2][:6]
    print(root)

    reses = []
    strains = []
    for file in tqdm(files):
        if 'datx' not in file: continue
        if 'no_ab' in file: continue
        if 'bad' in file: continue
        strains.append(float(file.split('_')[0]))
        raw,latres = convert_data(os.path.join(root,file),resolution=True,remove=False)
        reses.append(latres)
        corners = get_corners(raw)
        c = sym_reg(corners,2,full_image=raw)
        all_data['_'.join(file[:-5].split('_')+[date])] = c
            
        save_pickle(root+'/all_data',all_data)

    this_meta = pd.read_excel(root + '/Daily classification.xlsx')
    this_meta = this_meta.loc[this_meta['Strain Number'].astype(float).isin(strains)]
    print(f'Strains in folder: {len(strains)}\nStrains used: {len(this_meta)}')
    this_meta['Date'] = [date for _ in range(this_meta.shape[0])]

    this_meta['Lateral Resolution'] = reses
    meta = pd.concat([meta,this_meta])


plot_all(all_data)


print('Creating features')
feature_names = height_feats = get_features_ID(None,latres*1e6,region='full',obj=1.375,fftval=None)
feat_df = pd.DataFrame(columns = feature_names)
rows = []
for n, (i,d) in enumerate(tqdm(all_data.items())):
    feats={}
    i = tuple(i.split('_'))
    this_meta = meta.loc[meta['Strain Number']==float(i[0])].copy()
    this_meta['Filename'] = '_'.join(i)
    this_meta['Replicates'] = i[0]
    latres = this_meta['Lateral Resolution'].values[0]
    rows.extend(list(this_meta.values))
    
    height_feats = get_features_ID(d,latres*1e6,region='full',obj=1.375,fftval=None)
    feats = feats|{i:[d] for i,d in height_feats.items()}
    
    if include_ffts:
        fft_feats = new_get_all_fft_features(d,latres*1e6, region='full')
        feats = feats|{i:[d] for i,d in fft_feats.items()}
    
    temp_df = pd.DataFrame.from_dict(feats,orient='columns')
    feat_df = pd.concat([feat_df,temp_df],ignore_index = True,axis=0)
    feature_names = [name for name,typ in zip(feat_df.columns,feat_df.dtypes) if typ==float and not any([i in name for i in ['CFU','OD','Res']])]



feat_df = pd.concat([feat_df,pd.DataFrame(rows,columns=this_meta.keys())],ignore_index=False,axis=1)

print(f'feat_df shape: {feat_df.shape}')
save_pickle(top+'/features',feat_df)


plot_dic = {date:{i:all_data[i] for i in feat_df.loc[feat_df['Date']==date]['Filename']} for date in np.unique(feat_df['Date'])}
feature_names = [name for name,typ in zip(feat_df.columns,feat_df.dtypes) if typ==float and not any([i in name for i in ['CFU','OD','Res']])]


plot_all(plot_dic)

print(feat_df.shape)

plt.show()