from roughness_PB import getwloc,w_data_extraction
from scipy.stats import linregress
import numpy as np
from myfunctions import remove_outliers,ring_width,power_spectrum,convert_data,sym_reg,lmat,plot_contour,plot_all,plot_confusion,make_confusion,define_subplot_size
from myfunctions_full import polar,get_corners
from Find_best_algorithm import find_best_combination
import scipy
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#162,56,57,14,304
fast = True
classifying_feature='Class'
dates = ['230622','230713','230815','231004']
cut_features=[]
test_classes = ['H','S']
pairplot=False
homesize = 400 #not implemented
fftval = np.logspace(*np.log10(5.905723182647911e-06/np.array([0.    , 0.0125, 0.025 , 0.0375, 0.05  , 0.0625, 0.075 , 0.0875,
       0.1   , 0.1125, 0.125 , 0.1375, 0.15  , 0.1625, 0.175 , 0.1875,
       0.2   , 0.2125, 0.225 , 0.2375, 0.25  , 0.2625, 0.275 , 0.2875,
       0.3   , 0.3125, 0.325 , 0.3375, 0.35  , 0.3625, 0.375 , 0.3875,
       0.4   , 0.4125, 0.425 , 0.4375, 0.45  , 0.4625, 0.475 , 0.4875,
       0.5   , 0.5125, 0.525 , 0.5375, 0.55  , 0.5625, 0.575 , 0.5875,
       0.6   , 0.6125, 0.625 , 0.6375, 0.65  , 0.6625, 0.675 , 0.6875,
       0.7   ])*1e6)[[-1,1]],10,dtype=int)


def get_ring(arr, n=1000):
    dn = 2*np.pi/n
    r,th = polar(arr)
    ring,widths = [],[]
    for ang in np.arange(dn/2,2*np.pi,dn):
        mask = np.abs(th - ang)<dn/2
        ring.append(np.nanmax(arr[mask]))
        sector = arr[mask]
        slic   = r[mask]
        rpeak = slic[sector==ring[-1]]
        min_mask = r*mask*(r<rpeak)*(arr<ring[-1]/2)
        rmin = np.max(min_mask)
        max_mask = r*mask*(r>rpeak)*(arr<ring[-1]/2)
        max_mask = np.where(max_mask==0,np.max(r),max_mask)
        rmax = np.min(max_mask)
        w = rmax-rmin
        widths.append(w)
    return np.array(ring),np.array(widths)

def get_features(image,latres,fftval=18,homesize=homesize,ring_slices=1000,feature_names=['Home Height','Home Var','Home CoV','Ring Height','Ring Width','Ring CoV','Ring w_sat','Ring Hurst','ringH/homeH','ringH*homeH']):
    these_features = feature_names.copy()
    if fftval is not None:
        if not hasattr(fftval,'__iter__'): fftval=[fftval]
        for val in fftval:
            these_features.append('Power-{}um'.format(val))
    if image is None: return these_features
    image = np.where(np.isnan(image),np.nanmedian(image),image)
    home = image[300:700,300:700]
    ring,widths = get_ring(image,n=ring_slices)
                                            #make fft for coffee ring 
    coefs = np.polyfit(np.arange(len(ring)),ring,1)
    fit = np.sum([np.arange(len(ring))**(1-d)*c for d,c in enumerate(coefs)],axis=0)
    flucs = ring-fit
    rh = np.median(ring); rv = np.var(flucs)
    rw = np.median(widths)
    med = np.median(home); var = np.var(home)
    if fftval is not None:
        x,spect=np.array(power_spectrum(home,plot=False))
        powerfeats = []
        for val in fftval:
            index = np.argmin(np.abs(1/x*latres-val))
            powerfeats.append(np.log(spect[index]))

    loc, wloc = getwloc(flucs, latres, rx=0.3)
    l_sat, w_sat, h = w_data_extraction(loc,wloc)
    hurst = linregress(np.log10(loc)[:15],np.log10(wloc)[:15]).slope

    feats = [med,var,var/med,rh,rw,rv/rh,w_sat,hurst,rh/med,rh*med]+powerfeats
    feats = {i:d for i,d in zip(feature_names,feats)}
    return feats

roots = ['Weiss Lab Data/HR/{}_1-per-agar/filled/'.format(date) for date in dates]

plt.close('all')
full_df = pd.DataFrame()
data ={}
for root in roots:
    date = root.split('/')[2][:6]
    print('Working on date: ',date)
    new_data = {}
    files,phenotypes,strains = [],[],[]
    info = pd.read_excel(root+'Daily classification.xlsx')
    all_feature_names = get_features(None,None,fftval=fftval)
    features = {i:[] for i in all_feature_names}
    
    if fast and os.path.exists(root+'cleaned_data.mat') and os.path.exists(root+'features.xlsx'):
        new_data = lmat(root+'cleaned_data.mat',{})
        df = pd.read_excel(root+'features.xlsx')
    else:
        for file in tqdm(os.listdir(root)):
            if 'datx' not in file: continue
            if 'ab' in file.lower(): continue
            sn = int(file.split('_')[0])
            mask = info['Strain Number']==sn
            phen = info['Classification'].loc[mask].values[0]
            raw,latres = convert_data(root+file,resolution=True)
            corners = get_corners(raw)
            c = sym_reg(corners,1,normal=True,full_image=raw)
            new_data[file.split('_')[0]+'__'+phen] = c
            
            features_dict = get_features(c,latres,fftval=fftval)
            for name,feat in features_dict.items():
                features[name].append(feat)
            files.append(file.split('_')[0]+'_' +date)
            phenotypes.append(phen)
            
        df = pd.DataFrame(features)
        df['Class'] = phenotypes
        df['Filename'] = files
        df['Replicates']=files
        
        df.to_excel(root+'features.xlsx',index=False)
        scipy.io.savemat(root+'cleaned_data.mat',new_data)
    full_df = pd.concat([full_df,df],axis=0)
    data = data | new_data

scaled_df = full_df.copy()
for i in all_feature_names:
    dat = np.array(scaled_df[i])
    dat = (dat-np.mean(dat))/np.std(dat)
    scaled_df[i] = dat
if pairplot:
    sns.pairplot(scaled_df,hue='Class',vars=all_feature_names)    

import sklearn 
feature_names = [i for i in all_feature_names if i not in cut_features]
phen_mask = np.array([i in test_classes for i in scaled_df[classifying_feature]])
test_df = scaled_df.loc[phen_mask].copy()
classes = np.unique(test_df[classifying_feature]).tolist()
model = sklearn.cluster.KMeans(len(classes))
model.fit(np.array(test_df[feature_names]))
test_df['Model_KMeans']=[i+'->'+str(j) for i,j in zip(test_df[classifying_feature],model.labels_)]
u = pd.unique(test_df['Model_KMeans'])

confusion_matrix,cipher = make_confusion(test_df,get_cipher=True,truth=classifying_feature)
plot_confusion(confusion_matrix,labels=classes,stats=True)
#not ideal because now x and o are same color..but good classification generalization
colors = {c:i for i,c in zip([[0,0,1],[1,0,0],[1,0.5,0],[0.5,1,0],[0.5,0.5,0]],np.unique(test_df[classifying_feature]))}
markers = [['o','X'][int(i.split('->')[0]!=cipher[i.split('->')[1]])] for i in u]
palette = {i:tuple(colors[i.split('->')[0]]+[0.1 if m=='o' else 1]) for i,m in zip(u,markers)}


u_classes = list(np.unique(test_df[classifying_feature]))

results = [cipher[i.split('->')[1]] for i in test_df['Model_KMeans']]#[cipher[i[-1]]==i.split('->')[0] for i in test_df['Model_KMeans']]

test_df.drop('model_KMeans-unsupervised',axis='columns',inplace=True,errors='ignore')
test_df['model_KMeans-unsupervised'] = results




mats = {'KMeans':confusion_matrix}
##SVM
if True:
    from sklearn import svm
    kernels= ['linear','poly','rbf','sigmoid']
    fig,axs = plt.subplots(*define_subplot_size(len(kernels)))
    X = np.array(list(test_df[feature_names].values))
    Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
    replicates = np.unique(test_df['Replicates'].values)
    wrongs,fails = [],[]
    for n_k,kernel in enumerate(kernels):
        test_df.drop('model_SVM-'+kernel,axis='columns',inplace=True,errors='ignore')
        test_df['model_SVM-'+kernel] = [None for i in range(test_df.shape[0])]
        mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
        wrong,fail = [],[]
        for n,(rep) in enumerate(tqdm(replicates)):
            to_keep = test_df['Replicates'].values!=rep
            train_x = X[to_keep]
            train_y = Y[to_keep]
            test_x = X[~to_keep]
            test_y = Y[~to_keep]
            clf = svm.SVC(kernel = kernel,class_weight='balanced')
            clf.fit(train_x,train_y)
            result=clf.predict(test_x)
            for m,(true,guess) in enumerate(zip(test_y,result)):
                fname = test_df['Filename'].values[~to_keep][m]            
                test_df.loc[test_df['Filename']==fname,'model_SVM-'+kernel]= u_classes[guess]#==true
                mat[true][guess]+=1
                if true!=guess: wrong.append(fname)
        del clf
        plot_confusion(mat,labels=u_classes,ax=np.array([axs]).flatten()[n_k],stats=1)
        title = np.array([axs]).flatten()[n_k].get_title();np.array([axs]).flatten()[n_k].set_title(title+'\nSVM-'+kernel)
        print(kernel,np.round(np.trace(mat)/np.sum(mat)*100,1))
        wrongs.append(wrong);fails.append(fail)
        mats = mats|{kernel:mat}
    plt.suptitle('SVM')
    wrongs_dic = {}
    for wrong in wrongs:
        for f in wrong:
            if f not in wrongs_dic: wrongs_dic[f]=0
            wrongs_dic[f]+=1
    fails_dic = {}
    for fail in fails:
        for f in fail:
            if f not in fails_dic: fails_dic[f]=0
            fails_dic[f]+=1

##Random Forest
if True:
    from sklearn.ensemble import RandomForestClassifier
    test_df.drop('model_Random Forest',axis='columns',inplace=True,errors='ignore')
    test_df['model_Random Forest'] = [None for i in range(test_df.shape[0])]
    X = np.array(list(test_df[feature_names].values))
    Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
    replicates = np.unique(test_df['Replicates'].values)
    mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
    importances = []
    for n,(rep) in enumerate(tqdm(replicates)):
        to_keep = test_df['Replicates'].values!=rep
        train_x = X[to_keep]
        train_y = Y[to_keep]
        test_x = X[~to_keep]
        test_y = Y[~to_keep]
        forest = RandomForestClassifier()
        forest.fit(train_x,train_y)
        result = forest.predict(test_x)
        importances.append(forest.feature_importances_)
        for m,(true,guess) in enumerate(zip(test_y,result)):
            fname = test_df['Filename'].values[~to_keep][m]            
            test_df.loc[test_df['Filename']==fname,'model_Random Forest']= u_classes[guess] #==true
            mat[true][guess]+=1
        del forest
    plot_confusion(mat,u_classes,stats=1)
    title = plt.gca().get_title();plt.title(title+'\nRandom Forest')
    importances = np.array(importances)
    #plt.figure('Importances');plt.errorbar(feature_names,np.sum(imp,axis=0),yerr=np.std(imp,axis=0),fmt='o')

##Multi-layer perceptron
if False:
    from sklearn.neural_network import MLPClassifier
    test_df.drop('model_MLP',axis='columns',inplace=True,errors='ignore')
    test_df['model_MLP'] = [None for i in range(test_df.shape[0])]
    X = np.array(list(test_df[feature_names].values))
    Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
    replicates = np.unique(test_df['Replicates'].values)
    mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
    importances = []
    for n,(rep) in enumerate(tqdm(replicates)):
        to_keep = test_df['Replicates'].values!=rep
        train_x = X[to_keep]
        train_y = Y[to_keep]
        test_x = X[~to_keep]
        test_y = Y[~to_keep]
        mlp = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(6,9))
        forest.fit(train_x,train_y)
        result = forest.predict(test_x)
        importances.append(forest.feature_importances_)
        for m,(true,guess) in enumerate(zip(test_y,result)):
            fname = test_df['Filename'].values[~to_keep][m]            
            test_df.loc[test_df['Filename']==fname,'model_MLP']= u_classes[guess]#==true
            mat[true][guess]+=1
        del forest
    plot_confusion(mat,u_classes)
    title = plt.gca().get_title();plt.title(title+'\nRandom Forest')


model_names = [col for col in test_df if 'model' in col]# and 'unsupervised' not in col]

strain_info_columns = [i for i in test_df.columns if i not in feature_names and i not in model_names]


results,wrong_reps,names,first_sorting = find_best_combination(test_df,model_names,classifying_feature)
#all_results[(str(family),str(drug))] = results
best = {n:d for n,d in zip(names,results[0])}
mat_best = best['Confusion Matrix']
plot_confusion(np.int_(mat_best),labels = u_classes)
mat = mat_best.copy()
#all_mats[str(drug)]=mat
failed = {'R':0,'S':0,'N':0}
if len(best['Failed replicates'])>0:
    for fail in np.array(best['Failed replicates'],dtype=object).T[0]:
        failed[fail[-1]]+=1
acc,sens,spec = plot_confusion(mat,['R','S'],plot=False)
columns = ['Successful Tests','Number R','Number S','Success Rate',
           'Accuracy','R Sensitivity','R Specificity','% VME','% ME',
           '# Failed Tests','# Failed R','# Failed S','Best Model']
summary_list = [int(np.sum(mat)),int(np.sum(mat[0])),int(np.sum(mat[1])),best['Success Rate'],
           acc,sens,spec,best['Very Major Error'],best['Major Error'],
           best['# fails'],failed['R'],failed['S'],best['Test names']
           ]

#summary[(family_identifier,drug_identifier)] = summary_list
print(np.sum(mat_best,axis=1))
    
    
    
# =============================================================================
#     
# plt.close('all')
# n=1000
# dn = 2*np.pi/n
# r,th = polar(arr)
# ring,widths = [],[]
# locmins,locs,locmaxs=[],[],[]
# for ang in np.arange(dn/2,2*np.pi,dn):
#     mask = np.abs(th - ang)<dn/2
#     ring.append(np.nanmax(arr[mask]))
#     rpeak = r[mask][arr[mask]==ring[-1]]
#     rmax  = r[mask&(r>rpeak)][np.nanargmin(np.abs(arr[mask&(r>rpeak)]-ring[-1]/2))]
#     rmin  = r[mask&(r<rpeak)][np.nanargmin(np.abs(arr[mask&(r<rpeak)]-ring[-1]/2))]
#     locmins.append(list(np.unravel_index(np.nanargmin(np.abs(arr*(mask&(r<rpeak))-ring[-1]/2)),arr.shape)))
#     locmaxs.append(list(np.unravel_index(np.nanargmin(np.abs(arr*(mask&(r>rpeak))-ring[-1]/2)),arr.shape)))
#     locs.append(list(np.unravel_index(np.nanargmax(arr*mask),arr.shape)))
#     w = rmax-rmin
#     widths.append(w)
# locs = np.array(locs)
# locmins = np.array(locmins).T[[1,0]]
# locmaxs = np.array(locmaxs).T[[1,0]]
# locmins[1] = np.ones_like(locmins[1])*locmins.shape[1] - locmins[1]
# locmaxs[1] = np.ones_like(locmaxs[1])*locmaxs.shape[1] - locmaxs[1]
# s=2
# plot_contour(arr);plt.scatter(*locmins.T[[1,0]],s=s,alpha=0.8,c='orange');plt.scatter(*locs.T[[1,0]],s=s,alpha=0.8,c='red')
# plt.scatter(*locmaxs.T[[1,0]],s=s,alpha=0.8,c='orange')
# =============================================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    