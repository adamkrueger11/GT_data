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



noise = None  #std in nm
save = True
plt.close('all')

all_data,meta={},pd.DataFrame()
cmap = lambda a: ['blue','red','black'][a]#plt.get_cmap('tab10')

meta = pd.DataFrame()
top = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST/'

for root,folds,files in os.walk(top):
    if not root.endswith('-4hr'): continue
    folder = root.split("/")[-1]
    if not os.path.exists(root + '/strain_info.xlsx'):
        print(f'strain_info.xlsx not found in {folder}, but here are the excel files:')
        for file in files:
            if file.endswith('.xlsx') and file != 'features.xlsx':
                os.rename(os.path.join(root, file), os.path.join(root, 'strain_info.xlsx'))
                print(f'Renamed {file} to strain_info.xlsx')
        print()
        continue
previously_bad = 0
bads = []
for root,folds,files in os.walk(top):
    if not root.endswith('-4hr'): continue
    print(f'Collecting data for {root.split("/")[-1]}')
    date = root.split('/')[-1][:6]
    drug = root.split('/')[-1][-7:-4]

    #if drug != 'TZP': continue

    if drug.islower(): continue # when drug is lowercase, it is not a drug, but rather the no ab control. Not needed here.
    family = root.split('/')[-2].split(' ')[-1]
    folder = root.split('/')[-1]
    this_meta = pd.read_excel(root + '/strain_info.xlsx')

    path = root+'/data.pickle'
    if os.path.exists(path) and False:
        data_df = read_pickle(path)
        data_read = True
    else:
        data_read = False
        data = {i: [] for i in ['Data-array','Filename','FileBase','Phenotype','Date','Replicate','Lateral Resolution','Replicates','Bad']}
        length = 0
        these_data = {}
        for file in tqdm(files):
            if 'datx' not in file or 'no_ab' in file: continue
            if 'bad' in file:
                previously_bad += 1
                bads.append([family,drug,file])
                bad = True 
            else:
                bad = False
            
            raw,latres = convert_data(os.path.join(root,file),resolution=True,remove=False)
            if noise is not None:
                raw = raw +np.random.normal(0, noise, size=raw.shape)
            I,J = cartesian(raw)
            mask = J>raw.shape[1]-125
            agar_edge = np.where(mask,raw,np.nan)
            c = sym_reg(agar_edge,1,normal=True,full_image=raw)
            all_data[file[:-5]] = raw


            data['Filename'].append(file[:-5])
            data['Data-array'].append(c)
            if bad:
                file = file.replace('bad_','')
                file = file.replace('bad-','')
                file = file.replace('bad','')
            data['FileBase'].append(abs(float(file.split('_')[0])))
            data['Date'].append(date)
            data['Replicate'].append(file.split('_')[1])
            data['Lateral Resolution'].append(latres)
            data['Replicates'].append('_'.join([data['Date'][-1],str(data['FileBase'][-1]),drug]))
            data['Bad'].append(bad)

            other_meta = this_meta.loc[this_meta['image_label'].astype(float)== data['FileBase'][-1]].copy()
            phen_col = [c for c in other_meta.columns if 'phenotype' in c.lower()][0]
            data['Phenotype'].append(other_meta[phen_col].values[0][0])
            length += 1
        #save_pickle(root+'/all_data',all_data)

        data['Drug'] = [drug] * length
        data['Family'] = [family] * length
        data['Folder'] = [folder] * length
        data_df = pd.DataFrame(data)
    meta = pd.concat([meta,data_df],ignore_index=True,axis=0)
    if not data_read and noise is None:
        save_pickle(root+'/data',data_df)
bad_fams, bad_drugs, bad_files = np.array(bads).T
fig,axs = plt.subplots(1,2,figsize=(12,6))
u,c = np.unique(bad_fams,return_counts=True)
axs[0].bar(u,c)
u,c = np.unique(bad_drugs,return_counts=True)
axs[1].bar(u,c)
meta['Data-array'] = [data for data in meta['Data-array'].values]

plot_df(meta, split=['Bad+1','Drug'], sort=['Phenotype','FileBase'],close=False)

bads = meta.loc[meta['Bad'],['Folder','Filename']]
for bad in bads.iterrows():
    print(bad)
# save_pickle(top+'/bad_files',bads)
# plt.show()

meta = meta.loc[~meta['Bad']].copy()
#plot_df(meta, split = ['Family','Drug'], sort=['Phenotype','FileBase'], background='Phenotype')
#plt.show()
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
feature_names = get_features_ID(None,latres=None,region='edge',fftval=None)
feat_df = None
rows = []
failures = 0
for n, (_, row) in enumerate(tqdm(meta.iterrows(),total=meta.shape[0])):
    feats={}
    d = row['Data-array']
    latres = row['Lateral Resolution']


    try:
        height_feats = get_features_ID(d,latres*1e6,region='edge', obj=5,fftval=None)
        feats = feats|{i:[d] for i,d in height_feats.items()}
        
        fft_feats = new_get_all_fft_features(d,latres*1e6, region='edge', wavelengths=10)
        feats = feats|{i:[d] for i,d in fft_feats.items()}
    except Exception as err:
        print(f'Error processing {row["Filename"]}, skipping...')
        failures += 1
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
if save:
    save_pickle(top+f'/features' +(f'{noise}' if noise is not None else ''),feat_df)
print(f'Failures: {failures} - {failures/len(meta)*100:.1f}%')
print(f'Previously bad: {previously_bad} - {previously_bad/(len(meta)+previously_bad)*100:.1f}%')
print(f'Features created: {feat_df.shape[1]}')
plt.show()