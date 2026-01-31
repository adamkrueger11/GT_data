import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import itertools
from tqdm import tqdm
from myfunctions import convert_data,plot_contour,cartesian,sym_reg,fit_lows,save_pickle,read_pickle,get_features_ID,get_all_fft_features

##Collection/Data inputs
dates = ['all','241114','241121']
time = '4hr'
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
top = 'my data/ID/'
if 'all' in dates: dates = [i[:6] for i in os.listdir(top+time) if f'ID-{time}' in i]
obj = {'0hr':50,'4hr':10}[time]
all_data,meta={},pd.DataFrame()
cmap = lambda a: ['blue','red','black'][a]#plt.get_cmap('tab10')
show_any = show_wrongs or show_fails

while True:
    main_root = top+time+'/'
    if os.path.exists(main_root+'features.pickle') and fast_features:
        feat_df = read_pickle(main_root+'features.pickle')
        print('Features Loaded')
        if all([date in np.unique(feat_df['Date']) for date in dates]):
            if (not get_images and not show_any) or ((get_images or show_any) and len(all_data)==feat_df.shape[0]): break
        else:
            print('Not all dates found')
            fast_features = False
    
    if get_images or show_any or not fast_features:
        print('Assembling Data')
        for r,folds,files in os.walk(top+f'{time}/'):
            if not r.endswith('-'+time): continue
            date = r.split('/')[-1][:6]
            print(r)
        
            root = f'my data/ID/{time}/{date}_ID-{time}/'
            if 'all_data.pickle' in os.listdir(root) and fast_images:
                all_data = all_data|read_pickle(root+'all_data')
            else:
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
            meta = pd.concat([meta,this_meta])
        
        meta_dic = {sn: {i:d for i,d in zip(meta.columns,meta.loc[meta['Strain']==sn].values[0])} for sn in np.unique(meta['Strain'])}
    if fast_features: continue
    print('Creating features')
    feature_names = get_features_ID(None,fftval=None)
    feat_df = pd.DataFrame(columns = feature_names)
    rows = []
    for i,d in tqdm(all_data.items()):
        feats={}
        i = tuple(i.split('_'))
        this_meta = meta_dic[int(i[0])].copy()
        this_meta['Filename'] = '_'.join(i)
        this_meta['Replicates'] = i[0]
        rows.append(this_meta)
        
        height_feats = get_features_ID(d,obj=obj,fftval=None)
        feats = feats|{i:[d] for i,d in height_feats.items()}
        
        if include_ffts:
            fft_feats = get_all_fft_features(d,obj=obj, wavelengths='all',plot=plot_ffts,c=cmap(int(i[0][0])),alpha=0.2)
            feats = feats|{i:[d] for i,d in fft_feats.items()}
        
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

plot_dic = {gen:{i:all_data[i] for i in feat_df.loc[feat_df['Genera']==gen]['Filename']} for gen in np.unique(feat_df['Genera'])}
feature_names = [name for name,typ in zip(feat_df.columns,feat_df.dtypes) if typ==float and not any([i in name for i in ['CFU','OD']])]
if not include_ffts:
    feature_names = [i for i in feature_names if 'power' not in i]
meta_names = [name for name in feat_df.columns if name not in feature_names and not name.startswith('R-')]
from ML_functions import ML as learner

all_classes,counts = np.unique(feat_df[classify_by],return_counts=True)
classes = all_classes[counts>=min_count]
test_df = feat_df.loc[np.array([i in classes for i in feat_df[classify_by]]) & feat_df['Usable']].copy()
unusable_files = feat_df.loc[np.logical_not(feat_df['Usable']),['Filename']].values
unusable_dic = {i:d for i,d in all_data.items() if i in unusable_files}

n_combos = 2 if pairwise else len(classes)
if svm:
    for combo in itertools.combinations(classes,n_combos):
        this_df = test_df.loc[[i in combo for i in test_df[classify_by]]].copy()
        classifier = learner(this_df,feature_names,truth_feature=classify_by)
        classifier.svm_loocv(use_replicates=True,include_fails=include_fails)
        classifier.svm_loocv.confusion_matrices()

    if show_wrongs and not pairwise:
        classifier.svm_loocv.plot_wrongs(all_data)
    
    if show_fails and not pairwise:
        classifier.svm_loocv.plot_fails(all_data)

if forest:
    classifier = learner(this_df,feature_names,truth_feature=classify_by)
    classifier.forest_loocv()
    classifier.forest_loocv.confusion_matrices()
    classifier.forest_loocv.plot_importances()






# =============================================================================
# plt.close('all') ## show the calculated ring for all data
# for sn in range(3):
#     this_dic = {i:d for i,d in all_data.items() if i.startswith(str(sn))}
#     fig,axs = plt.subplots(*define_subplot_size(len(this_dic)))
#     plt.suptitle(str(sn))
#     for ax,(i,dat) in tqdm(zip(axs.ravel(),this_dic.items())):
#         ring_width(dat,ax=ax)
# =============================================================================










# =============================================================================
# plt.close('all')
# fig,axs = plt.subplots(n_classes,max(clsi.values()))
# nums = np.zeros(3,dtype=int)
# cmap = plt.get_cmap('tab10')
# vols=[]
# for i,d in tqdm(all_data.items()):
#     #x = int(i[0])/1e3 
#     #plt.figure('Volume check')
#     #vol = np.nansum(d/1e3 *(1e6*latres)**2)
#     #vols.append(vol)
#     #plt.scatter(x,vol,c=cmap(int(np.floor(x))))
#     plot_contour(d,ax=axs[int(i[0][0]),nums[int(i[0][0])]],vlims=[-100,20e3],axis=False,cbar=False)
#     nums[int(i[0][0])]+=1
# #plt.figure('Volume check')
# #plt.ylabel('Volume (um^3)')
# #plt.gca().set_xticks([0,1,2],['Enterobacterales','Acinetobacter','Pseudomonas'],ha='center',rotation=0)
# =============================================================================
