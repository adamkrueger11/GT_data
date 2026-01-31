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
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.stats.proportion import proportion_confint

simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)


from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.simplefilter("ignore", RuntimeWarning)


full_run = True
all_tests = True
#feat_df = read_pickle(os.path.join(feature_root,'features.pickle'))

def rfe_and_svd(args):
    warnings.simplefilter("ignore", RuntimeWarning)
    GLOBAL_df, higher_rep, test_reps, num_features, var_cutoffs, (allfeatures, truth_feature) = args
    if not isinstance(test_reps, (list, tuple,np.ndarray)):
        test_reps = [test_reps]
    this_df = GLOBAL_df[GLOBAL_df['Replicates'] != higher_rep].copy()
    test_df = this_df[this_df['Replicates'].isin(test_reps)].copy()
    train_df = this_df[~this_df['Replicates'].isin(test_reps)].copy()
    best_features = get_best_features(train_df, allfeatures, num_features, truth_feature)
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
    return mats #, args[1:]

def get_best_features(train_df, all_features, num_features, truth_feature):
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

def evaluate_accuracies_varying_feature_and_variance(df, all_features, truth, acc_training_arrays = None, max_features=40, variance_cutoffs=None, urine = False, testing_mode = 'First'):
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
        if urine:
            if not all(test_df['Urine'].values):
                continue
        for num_features in all_num_features:
            new_df = df.loc[df['Replicates'] != test_rep].copy()
            if full_run:
                for val_rep in new_df['Replicates'].unique():
                    train_tasks.append((new_df, test_rep, val_rep, num_features, variance_cutoffs, (all_features, truth)))
            else:
                if n > 5 and len(train_tasks)>0: continue
                reps = new_df['Replicates'].unique()
                iteration = 0
                while len(train_tasks)<(n+1)*2 and iteration < 100:
                    iteration += 1
                    np.random.shuffle(reps)
                    cut = int(len(reps)*.25)
                    val_reps = reps[:cut]
                    val_df = new_df.loc[new_df['Replicates'].isin(val_reps)].copy()
                    train_reps = reps[cut:]
                    train_df = new_df.loc[new_df['Replicates'].isin(train_reps)].copy()
                    if train_df[truth].value_counts().min() < 3:
                        continue
                    if not test_df[truth].isin(train_df[truth]).any() or not val_df[truth].isin(train_df[truth]).any():
                        continue
                    train_tasks.append((new_df, test_rep, val_reps, num_features, variance_cutoffs, (all_features, truth)))
                if iteration >= 100:
                    print(f'Iteration limit reached for {test_rep} with {num_features} features')
                    continue
    
    # Convert the second and third levels of all_accs into a 2D array
    num_features_list = sorted(all_num_features)
    variance_cutoffs_list = sorted(variance_cutoffs, reverse=True)
    
    if acc_training_arrays is None:
        print(f'{len(train_tasks)} training tasks are prepared. Moving to Train...')

        all_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {n_features: {var_cutoff: [accs]}}
        
        #batch_size = 5000  # You can tune this (see below)
        #for batch_start in tqdm(range(0, len(train_tasks), batch_size),desc='Batch',leave=False):
        #    batch = train_tasks[batch_start : batch_start + batch_size]
        batch = train_tasks[:]
        results = Parallel(n_jobs= 8, verbose=0, backend='loky')(delayed(rfe_and_svd)(task) for task in tqdm(batch))
        for task, result in zip(batch,results):
            test_rep, _, num_features, var_cutoffs, _ = task[1:]
            
            if isinstance(result, Exception):
                print(f"Task failed for {test_rep}: {result}")
                continue 

            conf_mats = result
            for var_remaining, conf_mat in zip(var_cutoffs, conf_mats):
                all_accs[test_rep][int(num_features)][var_remaining].append(conf_mat)

        acc_training_arrays = {test_rep: np.zeros((len(num_features_list), len(variance_cutoffs_list))) for test_rep in all_accs.keys()}
        for i, tasks in enumerate(train_tasks):
            test_rep = tasks[1]
            for j, num_features in enumerate(num_features_list):
                for k, var_remaining in enumerate(variance_cutoffs_list):
                    mat = np.sum(all_accs[test_rep][int(num_features)][var_remaining],axis=0)
                    acc_training_arrays[test_rep][j, k] = np.trace(mat)/np.sum(mat)
        #plot_all(acc_training_arrays,fullname = False)
        #plt.show()

    test_tasks, used_var_cutoffs, used_num_features, used_inds = [],[],[], []
    for test_rep, arr in acc_training_arrays.items():
        sort_args = np.argsort(arr.ravel())
        inds = sort_args[(arr>=np.max(arr)-1/len(acc_training_arrays)).ravel()]
        if np.sum(arr>=np.max(arr)-0/len(acc_training_arrays))>1:
            print(f'{test_rep} has multiple solutions')
        if testing_mode == 'Generalize':
            inds_2d = [np.unravel_index(i, arr.shape) for i in inds]
            inds = sorted(inds_2d, key=lambda idx: (-num_features_list[idx[0]], -variance_cutoffs_list[idx[1]]))
        
        for n_ind, ind in enumerate(inds):
            if isinstance(ind, tuple):
                ind = np.ravel_multi_index(ind, arr.shape)
            num_features_value = num_features_list[ind // len(variance_cutoffs_list)]
            variance_cutoff_value = variance_cutoffs_list[ind % len(variance_cutoffs_list)]
            test_df = df.loc[df['Replicates']==test_rep].copy()
            train_df= df.loc[df['Replicates']!=test_rep].copy()
            if not test_df[truth].isin(train_df[truth]).any():
                print(f'{test_rep} has no training data for {truth}')
                continue
            args = (df.copy(), '', test_rep, num_features_value, variance_cutoff_value, (all_features, truth))
            used_var_cutoffs.append(variance_cutoff_value)
            used_num_features.append(num_features_value)
            test_tasks.append(args)
            if testing_mode in ['First']:
                used_inds.append((test_rep,ind))
                break
            elif testing_mode=='Generalize' and n_ind == 0:
                used_inds.append((test_rep,ind))
                test_rep, ind
    
    accs = defaultdict(list) 
    results = Parallel(n_jobs= 8, verbose=0, backend='loky')(delayed(rfe_and_svd)(task) for task in tqdm(test_tasks))
    big_acc = {i: np.nan * np.zeros((len(variance_cutoffs_list), len(num_features_list))) for i in df['Replicates'].unique()}
    for task, conf_mat in zip(test_tasks,results):
        _, test_rep, num_feat, var_cut, _ = task[1:]

        
        if isinstance(conf_mat, Exception):
            print(f"Task failed for {test_rep}: {conf_mat}")
            continue 
        j = num_features_list.index(num_feat)
        i = variance_cutoffs_list.index(var_cut)
        big_acc[test_rep][i,j] = np.trace(conf_mat[0])/np.sum(conf_mat[0])
        if testing_mode in ['First','Generalize']:
            usable_ind = [ind[1] for ind in used_inds if ind[0]==test_rep][0]
            if np.ravel_multi_index((j,i), big_acc[test_rep].T.shape) != usable_ind:
                continue
        accs[test_rep].append(conf_mat[0])
    
    figs = len(plt.get_fignums())
    plot_all({i:d.T for i,d in acc_training_arrays.items()},titles=False,vlims=[0.5,1])
    to_plot = {str(figs) :{test_rep:d for test_rep,d in big_acc.items() if any([test_rep in a for a in used_inds]) and d.ravel()[[ind[1] for ind in used_inds if ind[0]==test_rep][0]] < 1 }}
    axes = list(plot_all(to_plot,vlims=[0,1],titles=False,fullname=True,axis=True).values())[0]

    for test_rep,ind in used_inds:
        for ax in axes:
            if ax.get_gid() == test_rep:
                x,y_temp = np.unravel_index(ind,(len(num_features_list), len(variance_cutoffs_list)))
                y = len(variance_cutoffs_list) - y_temp - 1
                circ = plt.Circle((x+0.5,y+0.5), .5, color='black', fill=False, lw=1)
                ax.add_patch(circ)
                break
        
    #plt.figure('Hist2')
    #plt.hist(used_inds, bins = 17, range=(0, len(variance_cutoffs_list)*len(num_features_list)), align='mid')
    #accs = defaultdict(list)  # {n_features: {var_cutoff: [accs]}}
    #for task in tqdm(test_tasks, desc='Performing LOO Testing:', leave=False):
    #    _, _, test_rep, _, _, _ = task
    #    conf_mat = rfe_and_svd(task)
    if testing_mode == 'Max':
        accs = [max(mats, key=lambda mat: np.trace(mat)/np.sum(mat)) for mats in accs.values()]
    elif testing_mode in ['Average']:
        accs = [np.mean(mats, axis=0) for mats in accs.values()]
    elif testing_mode in ['First', 'Generalize']:
        accs = [mats[0] for mats in accs.values()]
        
    
    #plt.figure('Accuracies')
    #plt.hist([np.trace(mat)/np.sum(mat) for mat in accs], bins=20, range=(0,1))
    
    

    return accs, used_var_cutoffs, used_num_features, acc_training_arrays

if __name__ == '__main__':
    global GLOBAL_df
    feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/'
    allfeats = {}
    truth_features = {}
    for r,_,files in os.walk(feature_root):
        for file in files:
            if file!='features.pickle': continue
            if 'others' in r: continue
            filename = os.path.join(r,file)
            ctime = datetime.fromtimestamp(os.path.getctime(filename))
            this_feat_df = read_pickle(filename)
            if 'Rapid AST' in r:
                allfeats['Rapid AST'] = this_feat_df
                truth_features['Rapid AST'] = ['Phenotype']
                print(f'AST: {filename}')
                print(f"     {ctime}")
            if 'ID' in r:
                t = r.split('/')[-1]
                if 'ID' not in allfeats:
                    allfeats['ID'] = {}
                this_feat_df = this_feat_df.loc[this_feat_df['Genera']!='Enterobacter'].copy()
                allfeats['ID'][t] = this_feat_df
                truth_features['ID'] = ['Genera', 'Group']
                print(f'ID-{t}: {filename}')
                print(f"     {ctime}")
            if 'Urine' in r:
                print(this_feat_df.columns)
                allfeats['Urine'] = this_feat_df
                truth_features['Urine'] = ['Phenotype']
                print(f'Urine: {filename}')
                print(f"     {ctime}")
            if 'HR' in r:
                truth_features['HR'] = ['Classification']
                allfeats['HR'] = {}
                allfeats['HR']['RHS'] = this_feat_df.copy()
                allfeats['HR']['HS'] = this_feat_df.loc[this_feat_df['Classification']!='R',:].copy()
                print(f'HR: {filename}')
                print(f"     {ctime}")
            print(f"    {this_feat_df.shape}")
            print()
    
    

    if full_run:
        min_features = 10
        num_vars = 11
    else:
        min_features = 40
        num_vars = 1 

    for test,feat_df in allfeats.items():

        if test != 'ID':
            continue

        if type(feat_df) is dict:
            new_df = pd.DataFrame()
            for t, df in feat_df.items():
                df['Time'] = [t for _ in range(len(df))]
                new_df = pd.concat([new_df, df], axis=0, ignore_index=True)
            feat_df = new_df.copy()
        
        colors = plt.cm.tab10.colors  # Tab10 color wheel
        if 'Time' in feat_df.columns:
            color_map = {t: colors[i % len(colors)] for i, t in enumerate(sorted(feat_df['Time'].unique()))}
        elif 'Family' in feat_df.columns:
            color_map = {family: colors[i % len(colors)] for i, family in enumerate(sorted(feat_df['Family'].unique()))}
        else:
            color_map = {'Enterobacteriaceae': colors[0]}

        ns,ks = [],[]
        col_ns, col_ks, drugs = defaultdict(list), defaultdict(list), defaultdict(list)
            
        for truth_feature in truth_features[test]:
            if test == 'Urine':
                mask = allfeats['Rapid AST']['Drug'].isin(feat_df['Drug']) & (allfeats['Rapid AST']['Family']=='Enterobacteriaceae')
                feat_df['Urine'] = [True for _ in range(len(feat_df))]
                feat_df['Family'] = ['Enterobacteriaceae' for _ in range(len(feat_df))]
                feat_df = feat_df.loc[feat_df['Phenotype']!='H'].copy()
                train_df = allfeats['Rapid AST'].loc[mask].copy()
                train_df['Urine'] = [False for _ in range(len(train_df))]
                feat_df = pd.concat([feat_df, train_df],axis=0, ignore_index=True)

            if test == 'Rapid AST' or test == 'Urine':
                grouped = feat_df.groupby(['Drug','Family'])
            elif test == 'ID' or test == 'HR':
                grouped = feat_df.groupby(['Time'])
            else:
                grouped = [((test,'Family'), feat_df)]

            all_feats = [
                name for name, typ in zip(feat_df.columns, feat_df.dtypes)
                if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
            ]
            #all_feats = [f for f in feat_df.columns]
            #all_feats = [f for f in all_feats if feat_df[f].map(type).values[0]==float and 'File' not in f and 'Res' not in f]
            
            
            for group, this_df in tqdm(grouped,desc = f'Running {test}-{truth_feature}',leave=True):
                #if drug!='LEV' and not full_run: continue
                # if '4hr' in group:
                #     continue
                if len(group) == 1:
                    group = tuple([truth_feature]+list(group))
                drugbug = '-'.join(group)
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'[{current_time}] Working on {drugbug}')
                path = os.path.join(feature_root, test, 'RFE-SVD Results', 'Nested CV', group[1])
                
                os.makedirs(path, exist_ok=True)
                new_values = False
                if any([drugbug in file for file in os.listdir(path)]) and full_run:
                    training_arrs = np.load(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), allow_pickle=True).item()

                    accs, _, _, _ = evaluate_accuracies_varying_feature_and_variance(this_df, all_feats, truth_feature, acc_training_arrays=training_arrs, urine = test=='Urine', testing_mode='Generalize')
                    #accs = np.load(os.path.join(path, f'{drugbug}-ListofConfusionMatricies.npy'), allow_pickle=True)
                    mat = np.load(os.path.join(path, f'{drugbug}-ConfusionMatrix.npy'), allow_pickle=True)
                    variances = np.load(os.path.join(path, f'{drugbug}-Variances.npy'), allow_pickle=True)
                    nums = np.load(os.path.join(path, f'{drugbug}-RFENums.npy'), allow_pickle=True)



                else:
                    #training_arrs = np.load(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), allow_pickle=True).item()
                    training_arrs = None
                    GLOBAL_df = this_df.copy()
                    accs, variances, nums, training_arrs = evaluate_accuracies_varying_feature_and_variance(this_df, all_feats, truth_feature, acc_training_arrays=training_arrs, urine = test=='Urine')
                    new_values = True
                if all_tests:
                    mat = np.sum(accs,axis=0)
                else:
                    # Filter out confusion matrices with disagreements between replicates
                    mat = np.sum([acc for acc in accs if np.trace(acc)/np.sum(acc) in [0.,1.] and np.sum(acc)==2],axis=0)
                

                
                n = np.sum(mat)
                k = np.trace(mat)
                col_ns[group[1]].append(n)
                col_ks[group[1]].append(k)
                drugs[group[1]].append(group[0])
                ns.append(n)
                ks.append(k)
                ci_low, ci_high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')

                acc = np.trace(mat) / np.sum(mat)

                #plt.figure('Bar Accuracies-live',figsize=(13,7))
                #x_val = '\n'.join([group[0],group[1][0]])
                #this_bar = plt.bar(x_val, acc*100, yerr=100*np.array([[max([acc - ci_low,0])], [max([ci_high - acc,0])]]), capsize=5, color=color_map[group[1]], error_kw={'ecolor': 'black'})
                #bar_text = f'{k/n*100:.1f}\n{n}'
                #plt.text(this_bar[0].get_x() + this_bar[0].get_width() / 2, acc*100 - (acc - ci_low)*100 - 1, bar_text, ha='center', va='top', color='white', fontsize=8)
            
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
                    np.save(os.path.join(path, f'{drugbug}-ParamSweepArrays.npy'), training_arrs)
                    np.save(os.path.join(path, f'{drugbug}-ConfusionMatrix.npy'), mat)
                    np.save(os.path.join(path, f'{drugbug}-Variances.npy'), variances)
                    np.save(os.path.join(path, f'{drugbug}-RFENums.npy'), nums)
                    np.save(os.path.join(path, f'{drugbug}-ListofConfusionMatricies.npy'), accs)
                    fig.savefig(os.path.join(path, f'ParamHistograms-{drugbug}.png'), dpi=300)
                plt.close(fig)
            break
            
            
                    
        finalfig = plt.figure(f'Bar Accuracies-{test}',figsize=(13,7))
        finalfig.suptitle('Accuracy vs Drug\nUsing Nested CV', fontsize=16)
        for color_ind, all_nums in sorted(col_ns.items()):
            for n, k, drug in zip(col_ns[color_ind], col_ks[color_ind], drugs[color_ind]):
                ci_low, ci_high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
                acc = k / n
                family_color = color_map[color_ind]
                x_val = '-'.join([drug,color_ind]) # Create a unique x value for each bar
                plt.bar(x_val, 100 * acc, yerr=100*np.array([[max([acc - ci_low,0])], [max([ci_high - acc,0])]]), capsize=3, color=family_color)
                bar_text = f'{k/n*100:.1f}\n{int(n / (1 if all_tests else 2))}'
                y_height = 100 * (acc - (acc - ci_low))
                plt.text(x_val, y_height - 1, bar_text, ha='center', va='top', color='white', fontsize=12)

        
            tot_cor = np.sum(col_ks[color_ind])
            tot_num = np.sum(col_ns[color_ind])
            lighter_color = plt.cm.tab20(list(plt.cm.tab10.colors).index(color_map[color_ind]) * 2 + 1)
            bar = plt.bar(color_ind[:3], 100 * tot_cor / tot_num, color=lighter_color)
            bar_text = f'{tot_cor/tot_num*100:.1f}\n{int(tot_num / (1 if all_tests else 2))}'
            plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() - 1, bar_text, ha='center', va='top', color='white', fontsize=12)
        total_bar = plt.bar('Total',np.sum(ks)/np.sum(ns)*100,color='black')
        bar_text = f'{np.sum(ks)/np.sum(ns)*100:.1f}\n{int(np.sum(ns) / (1 if all_tests else 2))}'
        plt.text(total_bar[0].get_x() + total_bar[0].get_width() / 2, total_bar[0].get_height() - 1, bar_text, ha='center', va='top', color='white', fontsize=12)
        plt.gca().axhline(95, color='gray', linestyle='--')
        # Update x-axis tick labels to show only the drug names
        plt.xticks(ticks=plt.xticks()[0], labels=[tick.get_text().split('-')[0] for tick in plt.xticks()[1]])
        fullpath = os.path.join(feature_root, test, 'RFE-SVD Results', 'Nested CV')
        finalfig.savefig(os.path.join(fullpath, f'Bar Accuracies-{test}.png'), dpi=300)
                    
                    

    plt.show()
