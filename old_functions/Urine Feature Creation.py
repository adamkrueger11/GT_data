import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import itertools
from tqdm import tqdm
import sys 
sys.path.append(os.path.dirname('__file__'))
from functions import convert_data,cartesian,sym_reg,save_pickle,read_pickle,get_features_ID,new_get_all_fft_features,plot_df
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=FutureWarning)





plt.close('all')

all_data,meta={},pd.DataFrame()
cmap = lambda a: ['blue','red','black'][a]#plt.get_cmap('tab10')

meta = pd.DataFrame()
root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Urine/'




this_meta = pd.read_excel(root + '/strain_info.xlsx')

path = root+'/data.pickle'
if os.path.exists(path) and False:
    data_df = read_pickle(path)
    data_read = True
else:
    data_read = False
    data = {i: [] for i in ['Data-array','Filename','FileBase','Phenotype','Date','Replicate','Lateral Resolution','Replicates','Drug','Plated CFU','Drug CFU','Patient']}
    length = 0
    these_data = {}
    for n, file in enumerate(tqdm(os.listdir(root))):
        if 'datx' not in file: continue
        if 'nab' in file: continue
        if 'bad' in file:
            oldfile=file
            file = oldfile.replace('bad-', '', 1)
            os.rename(os.path.join(root,oldfile),os.path.join(root,file))
        
        raw,latres = convert_data(os.path.join(root,file),resolution=True,remove=False)
        
        I,J = cartesian(raw)
        mask = J>raw.shape[1]-125
        agar_edge = np.where(mask,raw,np.nan)
        c = sym_reg(agar_edge,1,normal=True,full_image=raw)
        all_data[file[:-5]] = raw
        
        drug = file.split('_')[2]
        data['Filename'].append(file[:-5])
        data['Data-array'].append(c)
        data['Patient'].append(float(file.split('_')[0]))
        data['Date'].append('241007')
        data['Replicate'].append(file.split('_')[1])
        data['Lateral Resolution'].append(latres)
        data['Replicates'].append('_'.join([data['Date'][-1],str(data['Patient'][-1]),drug]))
        data['Drug'].append(drug)
        data['FileBase'].append(n)

        other_meta = this_meta.loc[(this_meta['Patient'].astype(float)== data['Patient'][-1]) & (this_meta['Drug']==drug)].copy()
        phen_col = [c for c in other_meta.columns if 'phenotype' in c.lower()][0]
        data['Phenotype'].append(other_meta[phen_col].values[0])
        data['Plated CFU'].append(other_meta['Plated CFU'].values[0])
        data['Drug CFU'].append(other_meta['Drug CFU'].values[0])
        length += 1
    #save_pickle(root+'/all_data',all_data)

    data['Folder'] = ['Urine'] * length
    data_df = pd.DataFrame(data)
meta = pd.concat([meta,data_df],ignore_index=True,axis=0)
if not data_read:
    save_pickle(root+'/data',data_df)

meta['Data-array'] = [data for data in meta['Data-array'].values]
plot_df(meta, split = ['Drug'], sort=['Phenotype','FileBase'], background='Phenotype')
plt.show()
'''
from functions import plot_contour
for n, (_, row) in enumerate(tqdm(meta.iterrows(),total=meta.shape[0])):
    if n != 122: continue
    d = row['Data-array']
    print(row)
    plot_contour(d)
plt.show()
'''

print('Creating features')
feature_names = get_features_ID(None,latres=None,region='full',fftval=None)
feat_df,bad_df = None, None
rows = []
for n, (_, row) in enumerate(tqdm(meta.iterrows(),total=meta.shape[0])):
    feats={}
    d = row['Data-array']
    latres = row['Lateral Resolution']


    try:
        height_feats = get_features_ID(d,latres*1e6,region='edge', obj=5,fftval=None)
        feats = feats|{i:[d] for i,d in height_feats.items()}
        
        fft_feats = new_get_all_fft_features(d,latres*1e6, region='edge', wavelengths='all')
        feats = feats|{i:[d] for i,d in fft_feats.items()}
    except Exception as err:
        if bad_df is None:
            bad_df = row.to_frame().T  # Convert Series to DataFrame
        else:
            bad_df = pd.concat([bad_df, row.to_frame().T], axis=0, ignore_index=True)
        print(f'Error processing {row["Filename"]}, skipping...')
        print(err)
        continue



    features = pd.DataFrame.from_dict(feats,orient='columns')
    if isinstance(row, pd.Series):
        row = row.to_frame().T  # Convert Series to DataFrame
    temp_df = pd.concat([row.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    if feat_df is None:
        feat_df = temp_df.copy()
    else:
        feat_df = pd.concat([feat_df,temp_df],ignore_index = True,axis=0)
    feature_names = [name for name,typ in zip(feat_df.columns,feat_df.dtypes) if typ in (float,int) and not any([name in meta.columns])]
    
    
feat_df = feat_df.drop(columns=['Data-array'])
save_pickle(root+'/features',feat_df)

plot_df(bad_df)
plt.suptitle('Fails')

plt.show()