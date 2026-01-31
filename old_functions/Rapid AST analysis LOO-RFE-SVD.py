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
from functions import save_pickle, read_pickle, plot_df, make_confusion_from_results,plot_confusion,plot_contour
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


from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

truth_feature = 'Phenotype'
feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'
feat_df = read_pickle(os.path.join(feature_root,'features.pickle'))

myfeats = [
    name for name, typ in zip(feat_df.columns, feat_df.dtypes)
    if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
]
def rfe_and_svd(args):
    train_df,test_df,var_remaining,num_features = args
    best_features = get_best_features(train_df, myfeats, num_features)

    # Fit scaler and perform SVD on training only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[best_features].values)
    X_test = scaler.transform(test_df[best_features].values)

    U, S, VT = np.linalg.svd(X_train, full_matrices=False)
    per_var = 1 - np.cumsum(S**2 / np.sum(S**2))
    svd_feats = np.array([f'SVD_{i}' for i in range(1, len(per_var) + 1)])

    X_train_svd = X_train @ VT.T
    X_test_svd = X_test @ VT.T

    train_df[svd_feats] = X_train_svd
    test_df[svd_feats]  = X_test_svd

    
    keep = per_var >= var_remaining
    if not np.any(keep):
        keep[0] = True
    feats = svd_feats[keep]

    clf = ML(train_df, feats, truth_feature, scale=False)
    clf.svm_test(test_df)
    mat, _ = clf.make_confusion()
    acc = np.trace(mat) / np.sum(mat)
    return mat, args

def get_best_features(train_df, all_features, num_features):
    """
    Use RFE with a linear SVM to select top `num_features` from `train_df`.
    """
    
    X = train_df[all_features].values
    y = train_df[truth_feature].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # RFE with linear SVM
    base_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    selector = RFE(base_model, n_features_to_select=num_features)
    selector.fit(X_scaled, y)

    selected = train_df[all_features].columns[selector.support_].tolist()
    return selected


def plot_paramspace(arr,variance_cutoffs,all_num_features):
    plot_contour(arr.T)
    plt.gcf().set_size_inches((13,7))
    plt.ylabel('SVD Remaining\nVariance Cutoff')
    plt.xlabel('RFE # features')
    plt.yticks(ticks=np.arange(len(variance_cutoffs))+0.5, labels=[f'{np.log10(var_c):.1f}' for var_c in variance_cutoffs[::-1]])
    plt.xticks(ticks=np.arange(len(all_num_features))+0.5, labels=all_num_features[::1])
    plt.tight_layout()

def evaluate_accuracies_varying_feature_and_variance(df, all_features, truth, max_features=40, variance_cutoffs=None):
    """
    Iterate through different numbers of features for RFE and different variance cutoffs in SVD.
    Plot accuracy vs. number of RFE features and % SVD variance retained.
    """
    if variance_cutoffs is None:
        variance_cutoffs = np.logspace(-3,np.log10(0.5),11)#[0.5, 0.2, 0.1, 0.07, 0.05, 0.02, 0.01, 0.007, 0.005, 0.002, 0.001]
    
    all_num_features = np.arange(10, max_features + 1)

    grouped = df.groupby('Replicates')

    
    tasks = []

    for rep, rep_df in grouped:
        for num_features in all_num_features:
            for var_remaining in variance_cutoffs:
                train_df = df.loc[df['Replicates'] != rep].copy()
                test_df = rep_df.copy()
                tasks.append((train_df,test_df,var_remaining,num_features))
    
    
    all_accs = defaultdict(lambda: defaultdict(list))  # {n_features: {var_cutoff: [accs]}}
    with ProcessPoolExecutor() as executor:
        for acc, task in tqdm(executor.map(rfe_and_svd, tasks), total=len(tasks),leave=True):
            _,_,var_remaining,num_features = task
            all_accs[num_features][var_remaining].append(acc)

    # Plot
    
    arr = np.zeros((len(all_accs),len(variance_cutoffs)))
    for n, (n_features, var_dict) in enumerate(all_accs.items()):
        for m, (var, mats) in enumerate(var_dict.items()):
            mat = np.sum(mats,axis=0)
            arr[n,m] = np.trace(mat)/np.sum(mat)
            
    plot_paramspace(arr,variance_cutoffs,all_num_features)
    return arr, variance_cutoffs, all_num_features



if __name__ == '__main__':
    variances = np.logspace(-3,np.log10(0.5),11)#[0.5, 0.2, 0.1, 0.07, 0.05, 0.02, 0.01, 0.007, 0.005, 0.002, 0.001]
    nums = np.arange(10, 40 + 1)

    all_best_feats =[]

    drug_group = feat_df.groupby(['Drug','Family'])


    all_feats = [f for f in feat_df.columns]
    all_feats = [f for f in all_feats if feat_df[f].map(type).values[0]==float and 'File' not in f and 'Res' not in f]
        

    for (drug,family), this_df in tqdm(drug_group):
        if drug!='TZP': continue
        drugbug = f'{drug}-{family}'
        n_samples = this_df['Replicates'].nunique()
        for frac_to_keep in np.arange(.2,1.01,.2):
            num_to_keep = int(frac_to_keep*n_samples)
            path = os.path.join(feature_root,'RFE-SVD Results',f'subset_{drugbug}',f'{num_to_keep}-{n_samples}')
            
            reps = this_df['Replicates'].unique().tolist()
            
            iter_accs = []
            for it in range(1,11):
                np.random.shuffle(reps)
                figpath = os.path.join(path,f'{drugbug}_{it}.png')
                arr_path = os.path.join(path,f'{drugbug}_{it}.npy')
                if os.path.exists(figpath): 
                    acc_arr = np.load(arr_path)
                else:
                    sub_df = this_df.loc[this_df['Replicates'].isin(reps[:num_to_keep])].copy()
                    acc_arr,variances,nums = evaluate_accuracies_varying_feature_and_variance(sub_df,all_feats,'Phenotype')
                    plt.suptitle(f'{drugbug}\nMax Acc: {np.max(acc_arr)*100:.1f}, Med Acc: {np.median(acc_arr)*100:.1f}')
                    plt.tight_layout()
                    
                    os.makedirs(path,exist_ok=True)
                    plt.savefig(figpath,dpi=300)
                    np.save(arr_path,acc_arr)
                    plt.close(plt.gcf())

                iter_accs.append(acc_arr)

                if frac_to_keep==1 and it==1: break
            iter_accs_avg = np.mean(iter_accs,axis=0)
            plot_paramspace(iter_accs_avg,variances,nums)
            plt.suptitle(f'{drugbug}\nMax Acc: {np.max(iter_accs_avg)*100:.1f}, Med Acc: {np.median(iter_accs_avg)*100:.1f}')
            plt.tight_layout()
            plt.savefig(os.path.join(path,f'{drugbug}_avg.png'),dpi=300)
            np.save(os.path.join(path,f'{drugbug}_avg.npy'),iter_accs_avg)
            
        

        
        

    plt.show()
