import numpy as np 
import os
import sys 
from ML_functions import ML
import matplotlib.pyplot as plt 
from tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pandas as pd
sys.path.append('/Users/adamkrueger/Downloads/Yunker Lab')
from functions import save_pickle, read_pickle, plot_df, make_confusion_from_results,plot_confusion
from umap import UMAP
import seaborn as sns
from warnings import simplefilter
from concurrent.futures import ProcessPoolExecutor
sys.path.append('/Users/adamkrueger/Downloads/Yunker Lab')
from functions import save_pickle, read_pickle
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)


from sklearn.preprocessing import StandardScaler

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict



truth_feature = 'Phenotype'
feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'
feat_df = read_pickle(os.path.join(feature_root,'features.pickle'))

myfeats = [
    name for name, typ in zip(feat_df.columns, feat_df.dtypes)
    if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
]

all_best_feats = []

n_features = 11
import umap
def perform_umap(full_df, best_feats, truth_feature, classes=['R', 'S'], n_components=2):
    y = [list(classes).index(f) for f in full_df[truth_feature].values]
    X = full_df[best_feats].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    umap_feats = [f'UMAP_{i}' for i in range(1, n_components + 1)]
    svd_feats = [f'SVD_{i}' for i in range(1,3)]

    # SVD
    U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)
    X_svd = X_scaled @ VT[:2].T  # Project onto top 2 directions
    df_umap = pd.DataFrame(X_umap, columns=umap_feats)
    df_umap['SVD_1'] = X_svd[:, 0]
    df_umap['SVD_2'] = X_svd[:, 1]

    # Metadata
    df_umap['Label'] = [classes[i] for i in y]
    df_umap['Replicates'] = full_df['Replicates'].values
    df_umap['Filename'] = full_df['Filename'].values

    # Accuracy from UMAP features
    ml = ML(df_umap, umap_feats, truth_feature='Label')
    ml.svm_loocv(include_fails=True)
    acc = ml.svm_loocv.best_acc

    # Plot UMAP pairplot
    sns.pairplot(df_umap, hue='Label', vars=umap_feats + svd_feats, plot_kws={'s': 15})
    plt.suptitle(f'UMAP Pairplot - {len(best_feats)} Features Selected\n{acc * 100:.1f}% accuracy')
    plt.tight_layout()

    # Plot singular values
    plt.figure("Singular Values", figsize=(6, 4))
    plt.plot(np.arange(1, len(S) + 1), S, marker='o')
    plt.title("SVD - Singular Values")
    plt.xlabel("Component")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.tight_layout()

    return acc

final_accuracies = defaultdict(list)
def perform_SVD(df,features,truth):
    global final_accuracies
    X = df[features].values
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svd_feats = [f'SVD_{i}' for i in range(1,5)]

    # SVD
    U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)
    X_svd = X_scaled @ VT[:4].T  # Project onto top 4 directions
    df_svd = pd.DataFrame(X_svd, columns=svd_feats)
    df_svd['Truth'] = df[truth].values 
     # Plot UMAP pairplot

    drug = df['Drug'].unique()[0]
    fam=df['Family'].unique()[0]
    plt.figure('Singular Values',figsize=(12,8))
    plt.semilogy(np.arange(1,len(S) + 1),S**2/np.sum(S**2), marker='o',label=drug,alpha=0.6)
    plt.gca().axhline(0.1,color='k',linestyle='--')
    plt.gca().axhline(0.01,color='k',linestyle='--')
    plt.gca().axhline(0.001,color='k',linestyle='--')
    plt.ylim([10**(-3.5),1e1])

    
    

    grouped = df.groupby('Replicates')

    
    all_mats = defaultdict(list)

    for rep, rep_df in tqdm(grouped,desc=drug):
        # Leave-one-replicate-out split
        train_df = df.loc[df['Replicates'] != rep].copy()
        test_df = rep_df.copy()

        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_test = scaler.transform(test_df[features].values)

        # SVD on training data only
        U, S, VT = np.linalg.svd(X_train, full_matrices=False)
        per_var = 1-np.cumsum(S**2 / np.sum(S**2))
        svd_feats = np.array([f'SVD_{i}' for i in range(1, len(per_var)+1)])

        # Project both sets onto top 5 components
        train_df[svd_feats] = X_train @ VT[:len(svd_feats)].T
        test_df[svd_feats] = X_test @ VT[:len(svd_feats)].T
        
        # Train and test classifier
        for var_remaining in [0.5,0.2,0.1,0.07, 0.05, 0.02, 0.01,0.007,0.005,0.002, 0.001]:
            feats = svd_feats[per_var>=var_remaining]
            if len(feats) == 0:
                feats = [svd_feats[0]]
            clf = ML(train_df, feats, 'Phenotype', scale=False)
            clf.svm_test(test_df)
            rep_mat, _ = clf.make_confusion(all_must_agree=True)
            all_mats[var_remaining].append(rep_mat)

    # Final confusion matrix and accuracy
    if len(all_mats)==1:
        mat = np.sum(all_mats.values()[0], axis=0)
        acc = np.trace(mat) / np.sum(mat)
        x_val = drug+'\n'+fam[0]
        final_accuracies[fam].append([acc,np.sum(mat),x_val])
        if False:
            plot_confusion(mat,labels=['R','S'])
        plt.figure('Accuracies',figsize=(12,8))
        plt.bar(x_val,acc)
        plt.title(f'Acc using {100*(1-var_remaining)}\% of variance vs Drug')
    else:
        accs = []
        for var, mats in all_mats.items():
            mat = np.sum(mats,axis=0)
            acc = np.trace(mat) / np.sum(mat)
            accs.append(acc)
            print(np.sum(mat))
        plt.figure('Accuracies',figsize=(12,8))
        plt.semilogx(all_mats.keys(),accs,label=drug+'-'+fam)
            


    if False:
        sns.pairplot(df_svd, hue='Truth', vars=svd_feats, plot_kws={'s': 15})
        plt.suptitle(f'SVD Pairplot - {acc * 100:.1f}%\nDrug: {drug}\nFamily: {fam}')
        plt.tight_layout()





for r,_,files in os.walk(feature_root):
    #if 'TZP' in r: continue
    if not any(['best_5_feature' in file for file in files]): continue

    feats = read_pickle(os.path.join(r,f'best_{n_features}_feature_names.pickle'))
    all_feats = feats['All']
    feats = feats['Best']
    all_best_feats.extend(feats)

    path = os.path.join(r,f"LOO/RFE_Features_{n_features}.pickle") 
    if not os.path.exists(path): continue
    feature_names = read_pickle(path)
    accuracies = read_pickle(os.path.join(r,f"LOO/RFE_Accuracies_{n_features}.pickle"))
    
    reps = list(feature_names.keys())
    this_df = feat_df.loc[feat_df['Replicates'].isin(reps)].copy()
    
    #perform_umap(this_df,feature_names['Best'],'Phenotype')
    all_feats = [f for f in feat_df.columns]
    all_feats = [f for f in all_feats if feat_df[f].map(type).values[0]==float and 'File' not in f and 'Res' not in f]
    
    perform_SVD(this_df,all_feats,'Phenotype')

    output_file = os.path.join(r,f'zreplicate_{n_features}-feature_comparison.xlsx')
    best_features = set(feature_names['Best'])
    best_missed = []
    bad_included= []
    n_wrong = 0
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for rep, accs in accuracies.items():
            if np.mean(accs) >= 1 or rep == 'Best':
                continue  # Skip perfect scores or the 'Best' entry

            rep_features = set(feature_names[rep])

            shared = sorted(best_features & rep_features)
            only_in_best = sorted(best_features - rep_features)
            only_in_rep = sorted(rep_features - best_features)

            df = pd.DataFrame({
                'Shared': pd.Series(shared),
                'Best': pd.Series(only_in_best),
                'Replicate': pd.Series(only_in_rep),
            })

            # Clean sheet name (Excel has sheet name limits)
            sheet_name = str(rep)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            n_wrong += 1
            best_missed.extend(list(only_in_best))
            bad_included.extend(list(only_in_rep))
    fig,axs = plt.subplots(1,2,figsize=(12,7))
    plt.suptitle(f'{rep}-{n_wrong} wrong')
    u,c = np.unique(best_missed,return_counts=True)
    args = np.argsort(c)
    u = u[args]
    c = c[args]
    axs[0].bar(u,c)
    plt.xticks(rotation=30,fontsize=8,ha='right')
    u,c = np.unique(bad_included,return_counts=True)
    args = np.argsort(c)
    u = u[args]
    c = c[args]
    axs[1].bar(u,c,color='red')
    plt.xticks(rotation=30,fontsize=8,ha='right')
    plt.close(fig)

fig = plt.figure('Singular Values')
plt.legend()
if len(final_accuracies)>0:
    plt.close('all')
    fig,ax = plt.subplots()
    fig.suptitle('Accuracies')
    all_accs = []
    for n, (fam, accs) in enumerate(final_accuracies.items()):
        for acc,num,drug in accs:
            plt.bar(drug,acc,color=plt.get_cmap('tab10')(n))
        these_accs,nums = np.float64(np.array(accs).T[:2])
        print(these_accs,nums)
        total_acc = np.sum(these_accs*nums)/np.sum(nums)
        plt.bar(fam[0],total_acc,color=plt.get_cmap('tab20')(2*n+1))
        all_accs.append([total_acc,np.sum(nums)])
    accs, nums = np.array(all_accs).T 
    final_acc = np.sum(accs*nums)/np.sum(nums)
    plt.bar('Total',final_acc,color='black')
    ax.axhline(0.95,linestyle='dashed',color='black')


plt.show()
raise

fig, axes = plt.subplots(3,4,figsize=(12,7))
groups = feat_df.groupby(['Drug','Family'])
total_mat = np.zeros((2,2))
for n,((drug,family), df) in enumerate(tqdm(groups)):
    clf = ML(df,feature_names = myfeats,truth_feature = 'Phenotype')
    clf.svm_loocv()
    mat = clf.svm_loocv.mat_dict['linear']
    acc = clf.svm_loocv.best_acc
    ax = axes.ravel()[n]
    plot_confusion(mat,ax=ax,labels=['R','S'])
    ax.set_title(f'{drug}-{acc:.1f}')
    ax.set_ylabel('')
    ax.set_xlabel('')
    total_mat += mat 

ax = axes.ravel()[-1]
plot_confusion(total_mat,ax=ax,labels=['R','S'])
ax.set_title(f'All-{100*np.trace(total_mat)/np.sum(total_mat):.1f}')
ax.set_ylabel('')
ax.set_xlabel('')
plt.close('all')
plt.show()





top = "/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST"
feat_df['InBest'] = [-1 for _ in range(len(feat_df))]
feat_df['Accuracy'] = [-1 for _ in range(len(feat_df))]
all_data_df = pd.DataFrame()
for r,_,files in os.walk(feature_root):
    #if 'TZP' in r: continue
    if not any(['best_5_feature' in file for file in files]): continue
    if not os.path.exists(r+"/LOO/RFE_Features.pickle"): continue
    feature_names = read_pickle(r+"/LOO/RFE_Features.pickle")
    accuracies = read_pickle(r+"/LOO/RFE_Accuracies.pickle")
    best_feats = feature_names['Best']

    folds = feat_df.loc[feat_df['Replicates'].isin(feature_names.keys()),'Folder'].unique()
    data_df = pd.DataFrame()
    for family in np.unique(feat_df['Family']):
        for fold in folds:
            path = os.path.join(top,f'Rapid R {family}',fold,'data.pickle')
            if not os.path.exists(path): continue
            new = read_pickle(path)
            data_df = pd.concat([data_df,new],axis=0,ignore_index=True)

    folds = []
    all_feats= []
    for rep, feats in feature_names.items():
        if 'Best'==rep: continue
        all_feats.extend(feats)
        n_reps = np.sum(feat_df['Replicates']==rep)
        n_in_best = [np.sum([f in best_feats for f in feats]) for _ in range(n_reps)]
        feat_df.loc[feat_df['Replicates']==rep,'InBest'] = n_in_best
        data_df.loc[data_df['Replicates']==rep,'InBest'] = n_in_best
        accuracy = [accuracies[rep] for _ in range(n_reps)]
        feat_df.loc[feat_df['Replicates']==rep,'Accuracy']=  accuracy 
        data_df.loc[data_df['Replicates']==rep,'Accuracy'] = accuracy
        folds.extend(feat_df.loc[feat_df['Replicates']==rep,'Folder'])
    #feat_df = feat_df.loc[feat_df['InBest']>=0]
    

    all_data_df = pd.concat([all_data_df,data_df],ignore_index=True,axis=0)

    ## plot from a dataframe that has a column based on the number of features of that rep that are in the best features
    ## could also do it from RFE_accuracies < 1

print(all_data_df.columns)

df = all_data_df.loc[all_data_df['Accuracy']<1].copy()
plot_df(df,split='Drug',sort=['InBest','Accuracy','Replicates'],background='Accuracy')

plt.show()
drug = 'TZP'
this_df = feat_df.loc[feat_df['Drug']==drug].copy()

clf = ML(this_df, best_feats, truth_feature='Phenotype')
clf.svm_loocv()
clf.svm_loocv.confusion_matrices()
plt.show()