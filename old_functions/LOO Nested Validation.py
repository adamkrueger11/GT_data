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
from functions import save_pickle, read_pickle, plot_df, make_confusion_from_results,plot_confusion,plot_contour,plot_all
from umap import UMAP
import seaborn as sns
from warnings import simplefilter
from concurrent.futures import ProcessPoolExecutor
from statsmodels.stats.proportion import proportion_confint

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
from datetime import datetime

full_run = True
all_tests = False
truth_feature = 'Phenotype'
feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'
#feat_df = read_pickle(os.path.join(feature_root,'features.pickle'))

feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/'
for r,_,files in os.walk(feature_root):
    for file in files:
        if file!='features.pickle': continue
        filename = os.path.join(r,file)
        ctime = datetime.fromtimestamp(os.path.getctime(filename))
        if 'Rapid AST' in r:
            print(f'AST: {filename}')
            print(f"     {ctime}")
        if 'ID' in r:
            print(f'ID: {filename}')
            print(f"     {ctime}")
        if 'Urine' in r:
            print(f'Urine: {filename}')
            print(f"     {ctime}")
        if 'HR' in r:
            print(f'HR: {filename}')
            print(f"     {ctime}")

raise

                #feat_dfs['AST'] = read_pickle(os.path.join(r,file))
        #feature_root = os.path.join(feature_root,'Rapid AST')


if full_run:
    min_features = 10
    num_vars = 11
else:
    min_features = 40
    num_vars = 5 

myfeats = [
    name for name, typ in zip(feat_df.columns, feat_df.dtypes)
    if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
]
def rfe_and_svd(args):
    higher_rep, test_df, train_df ,num_features, var_cutoffs = args
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

    mats = []
    
    if not hasattr(var_cutoffs,'__iter__'):
        var_cutoffs = [var_cutoffs]
    for var_remaining in var_cutoffs:
        keep = per_var >= var_remaining
        if not np.any(keep):
            keep[0] = True
        feats = svd_feats[keep]

        clf = ML(train_df, feats, truth_feature, scale=False)
        clf.svm_test(test_df)
        mat, _ = clf.make_confusion()
        mats.append(mat)
        acc = np.trace(mat) / np.sum(mat)
    return mats, args

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

def evaluate_accuracies_varying_feature_and_variance(df, all_features, truth, acc_training_arrays = None, max_features=40, variance_cutoffs=None):
    """
    Iterate through different numbers of features for RFE and different variance cutoffs in SVD.
    Plot accuracy vs. number of RFE features and % SVD variance retained.
    """
    if variance_cutoffs is None:
        variance_cutoffs = np.logspace(-3,np.log10(0.5),num_vars)#[0.5, 0.2, 0.1, 0.07, 0.05, 0.02, 0.01, 0.007, 0.005, 0.002, 0.001]
    
    all_num_features = np.arange(min_features, max_features + 1)

    grouped = df.groupby('Replicates')

    
    train_tasks = []
    np.random.seed(11)
    for n, (test_rep, test_df) in enumerate(grouped):
        for num_features in all_num_features:
            new_df = df.loc[df['Replicates'] != test_rep].copy()
            if full_run:
                new_grouped = new_df.groupby('Replicates')
                for val_rep, val_df in new_grouped:
                    train_df = new_df.loc[new_df['Replicates'] != val_rep].copy()
                    train_tasks.append((test_rep, val_df, train_df, num_features, variance_cutoffs))
            else:
                reps = new_df['Replicates'].unique()
                iteration = 0
                while len(train_tasks)<(n+1)*5 and iteration < 100:
                    iteration += 1
                    np.random.shuffle(reps)
                    cut = int(len(reps)*.25)
                    val_reps = reps[:cut]
                    val_df = new_df.loc[new_df['Replicates'].isin(val_reps)].copy()
                    train_reps = reps[cut:]
                    train_df = new_df.loc[new_df['Replicates'].isin(train_reps)].copy()
                    if train_df[truth_feature].value_counts().min() < 3:
                        continue
                    train_tasks.append((test_rep, val_df, train_df, num_features, variance_cutoffs))
    
    # Convert the second and third levels of all_accs into a 2D array
    num_features_list = sorted(all_num_features)
    variance_cutoffs_list = sorted(variance_cutoffs, reverse=True)
    
    if acc_training_arrays is None:
        acc_training_arrays = {test_rep: np.zeros((len(num_features_list), len(variance_cutoffs_list))) for test_rep in grouped.groups.keys()}
        print('Training tasks are prepared. Moving to Train...')
        all_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {n_features: {var_cutoff: [accs]}}
        with ProcessPoolExecutor(max_workers=9) as executor:
            for conf_mats, task in tqdm(executor.map(rfe_and_svd, train_tasks), total=len(train_tasks),leave=True):
                test_rep, _, _, rfe_num, variances= task
                for var_remaining, conf_mat in zip(variances,conf_mats):
                    all_accs[test_rep][int(rfe_num)][var_remaining].append(conf_mat)

        for i, test_rep in enumerate(grouped.groups.keys()):
            for j, num_features in enumerate(num_features_list):
                for k, var_remaining in enumerate(variance_cutoffs_list):
                    mat = np.sum(all_accs[test_rep][int(num_features)][var_remaining],axis=0)
                    acc_training_arrays[test_rep][j, k] = np.trace(mat)/np.sum(mat)
        #plot_all(acc_training_arrays,fullname = False)
        #plt.show()

    test_tasks, used_var_cutoffs, used_num_features = [],[],[]
    for test_rep, arr in acc_training_arrays.items():
        ind = np.argmax(arr.ravel())
        if np.sum(arr==np.max(arr))>1:
            print(f'{test_rep} has multiple solutions')
        num_features_value = num_features_list[ind // len(variance_cutoffs_list)]
        variance_cutoff_value = variance_cutoffs_list[ind % len(variance_cutoffs_list)]
        test_df = df.loc[df['Replicates']==test_rep].copy()
        train_df= df.loc[df['Replicates']!=test_rep].copy()
        args = (test_rep, test_df, train_df, num_features_value, variance_cutoff_value)
        used_var_cutoffs.append(variance_cutoff_value)
        used_num_features.append(num_features_value)
        test_tasks.append(args)
    


    accs = []  # {n_features: {var_cutoff: [accs]}}
    with ProcessPoolExecutor() as executor:
        for conf_mat, task in tqdm(executor.map(rfe_and_svd, test_tasks), total=len(test_tasks),leave=False):
            test_rep, _, _, _, _ = task
            accs.append(conf_mat[0])
    
            
    return accs, used_var_cutoffs, used_num_features, acc_training_arrays



if __name__ == '__main__':
    
    all_best_feats =[]

    drug_group = feat_df.groupby(['Drug','Family'])


    all_feats = [f for f in feat_df.columns]
    all_feats = [f for f in all_feats if feat_df[f].map(type).values[0]==float and 'File' not in f and 'Res' not in f]
    
    ns,ks = [],[]
    fam_ns, fam_ks, drugs = defaultdict(list), defaultdict(list), defaultdict(list)
    colors = plt.cm.tab10.colors  # Tab10 color wheel
    family_color_map = {family: colors[i % len(colors)] for i, family in enumerate(sorted(feat_df['Family'].unique()))}

    for (drug,family), this_df in tqdm(drug_group):
        #if drug!='LEV' and not full_run: continue
        drugbug = f'{drug}-{family}'
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{current_time}] Working on {drugbug}')
        path = os.path.join(feature_root, 'RFE-SVD Results', 'Nested CV')
        new_values = False
        if any([drugbug in file for file in os.listdir(path)]) and full_run:
            training_arrs = np.load(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), allow_pickle=True).item()
            accs = np.load(os.path.join(path, f'{drugbug}-ListofConfusionMatricies.npy'), allow_pickle=True)
            mat = np.load(os.path.join(path, f'{drugbug}-ConfusionMatrix.npy'), allow_pickle=True)
            variances = np.load(os.path.join(path, f'{drugbug}-Variances.npy'), allow_pickle=True)
            nums = np.load(os.path.join(path, f'{drugbug}-RFENums.npy'), allow_pickle=True)
        else:
            #training_arrs = np.load(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), allow_pickle=True).item()
            training_arrs = None
            accs, variances, nums, training_arrs = evaluate_accuracies_varying_feature_and_variance(this_df, all_feats, 'Phenotype', acc_training_arrays=training_arrs)
            new_values = True
        if all_tests:
            mat = np.sum(accs,axis=0)
        else:
            # Filter out confusion matrices with disagreements between replicates
            mat = np.sum([acc for acc in accs if np.trace(acc)/np.sum(acc) in [0.,1.] and np.sum(acc)==2],axis=0)
        
        n = np.sum(mat)
        k = np.trace(mat)
        fam_ns[family].append(n)
        fam_ks[family].append(k)
        drugs[family].append(drug)
        ns.append(n)
        ks.append(k)
        ci_low, ci_high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')

        acc = np.trace(mat) / np.sum(mat)
        plt.figure('Bar Accuracies-live',figsize=(13,7))

        x_val = '\n'.join([drug,family[0]])
        this_bar = plt.bar(x_val, acc*100, yerr=100*np.array([[max([acc - ci_low,0])], [max([ci_high - acc,0])]]), capsize=5, color=family_color_map[family], error_kw={'ecolor': 'black'})
        bar_text = f'{k/n*100:.1f}\n{n}'
        plt.text(this_bar[0].get_x() + this_bar[0].get_width() / 2, acc*100 - (acc - ci_low)*100 - 1, bar_text, ha='center', va='top', color='white', fontsize=8)
    
        #plt.savefig(os.path.join(path,f'{drugbug}_avg.png'),dpi=300)
        #np.save(os.path.join(path,f'{drugbug}_avg.npy'),iter_accs_avg)
        fig, axs = plt.subplots(1,2, figsize=(13, 7))
        axs[0].hist(np.log10(variances), bins='auto', align='mid')
        axs[0].set_ylabel('Number of Occurences')
        axs[0].set_xlabel('Log10(Variance Remaining Cutoff)')
        axs[1].hist(nums, bins='auto', align='mid')
        axs[1].set_xlabel('Number of Features in RFE')
        plt.suptitle(f'Chosen params for {drugbug}')
        if full_run and new_values:
            print('Saving results...')
            path = os.path.join(feature_root, 'RFE-SVD Results', 'Nested CV')
            os.makedirs(path,exist_ok=True)
            np.save(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), training_arrs)
            np.save(os.path.join(path, f'{drugbug}-ConfusionMatrix.npy'), mat)
            np.save(os.path.join(path, f'{drugbug}-Variances.npy'), variances)
            np.save(os.path.join(path, f'{drugbug}-RFENums.npy'), nums)
            np.save(os.path.join(path, f'{drugbug}-ListofConfusionMatricies.npy'), accs)
            fig.savefig(os.path.join(path, f'ParamHistograms-{drugbug}.png'), dpi=300)
        
    
    fig = plt.figure('Bar Accuracies-live')
    plt.close(fig)
            
    finalfig = plt.figure('Bar Accuracies',figsize=(13,7))
    finalfig.suptitle('Accuracy vs Drug\nUsing Nested CV', fontsize=16)
    for family, all_nums in sorted(fam_ns.items()):
        for n, k, drug in zip(fam_ns[family], fam_ks[family], drugs[family]):
            ci_low, ci_high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
            acc = k / n
            family_color = family_color_map[family]
            x_val = f'{drug}-{family}'  # Create a unique x value for each bar
            plt.bar(x_val, 100 * acc, yerr=100*np.array([[max([acc - ci_low,0])], [max([ci_high - acc,0])]]), capsize=3, color=family_color)
            bar_text = f'{k/n*100:.1f}\n{int(n / (1 if all_tests else 2))}'
            y_height = 100 * (acc - (acc - ci_low))
            plt.text(x_val, y_height - 1, bar_text, ha='center', va='top', color='white', fontsize=12)

     
        tot_cor = np.sum(fam_ks[family])
        tot_num = np.sum(fam_ns[family])
        lighter_color = plt.cm.tab20(list(plt.cm.tab10.colors).index(family_color_map[family]) * 2 + 1)
        bar = plt.bar(family[0], 100 * tot_cor / tot_num, color=lighter_color)
        bar_text = f'{tot_cor/tot_num*100:.1f}\n{int(tot_num / (1 if all_tests else 2))}'
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() - 1, bar_text, ha='center', va='top', color='white', fontsize=12)
    total_bar = plt.bar('Total',np.sum(ks)/np.sum(ns)*100,color='black')
    bar_text = f'{np.sum(ks)/np.sum(ns)*100:.1f}\n{int(np.sum(ns) / (1 if all_tests else 2))}'
    plt.text(total_bar[0].get_x() + total_bar[0].get_width() / 2, total_bar[0].get_height() - 1, bar_text, ha='center', va='top', color='white', fontsize=12)
    plt.gca().axhline(95, color='gray', linestyle='--')
    # Update x-axis tick labels to show only the drug names
    plt.xticks(ticks=plt.xticks()[0], labels=[tick.get_text().split('-')[0] for tick in plt.xticks()[1]])
   
        
        

    plt.show()
