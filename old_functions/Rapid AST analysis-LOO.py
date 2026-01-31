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

noise_levels = [None, '1000', '4000', '10000']
accuracy_values = []
for noise in noise_levels:
    if noise is None:
        noise = ''

    truth_feature = 'Phenotype'
    feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'#/notLOO'
    feat_df = read_pickle(os.path.join(feature_root,f'features{noise}.pickle'))
    feature_names = [
        name for name, typ in zip(feat_df.columns, feat_df.dtypes)
        if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
    ]

    feat_df['Family'] = feat_df['Family'].replace('Enterobacteriaceae', 'Enterobacterales')
    feat_df['Family'] = feat_df['Family'].replace('Moraxellaceae', 'Acinetobacter')
        

    def rfe_leave_one_rep_out(args):
        rep, train_df, test_df, feature_names, truth_feature, n_features_to_select = args

        # Scale using only training data
        scaler = StandardScaler()
        train_df[feature_names] = scaler.fit_transform(train_df[feature_names])
        test_df[feature_names] = scaler.transform(test_df[feature_names])

        # Run RFE on full training set
        base_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
        rfe.fit(train_df[feature_names], train_df[truth_feature])

        # Select best features
        best_selected_features = train_df[feature_names].columns[rfe.support_].tolist()
        #best_selected_features = best_selected_features + [feat for feat in feature_names if 'power' not in feat]
        # Train and test model
        ml = ML(train_df, best_selected_features, truth_feature=truth_feature, scale=False)
        ml.svm_test(test_df)

        result = np.array(ml.svm_result)
        truth = np.array(ml.__svmy__)
        acc = np.sum(truth == result)

        return np.int_(truth==result), best_selected_features, n_features_to_select

        


    if __name__ == '__main__':
        accs = {}
        all_feat_accs = defaultdict(lambda: defaultdict(list))
        splits = [('Drug',val) for val in feat_df['Drug'].unique()] # + [('Family', val) for val in feat_df['Family'].unique()] + [('Family', tuple(list(feat_df['Family'].unique())))]

        for col, split_val in tqdm(splits,desc='Drug-Bugs'):
            #if 'SAM' not in split_val: continue
            values = (split_val,) if isinstance(split_val, str) else split_val
            temp_df = feat_df.loc[feat_df[col].isin(values)].copy()
            if col != 'Family':
                families = temp_df['Family'].unique().tolist()
                families += [tuple(families)] if len(families) > 1 else []
            else:
                families = [split_val]
            for family in tqdm(families,desc='Families',leave=False):
                family = (family,) if not isinstance(family, tuple) else family
                this_df = temp_df.loc[temp_df['Family'].isin(family)].copy()

                val = f'{split_val}' + (f' - {family}' if col!='Family' else '')
                try:
                    clf = ML(this_df, feature_names, 'Phenotype')
                    clf.svm_loocv()
                    all_feat_accs[family][val.split(' - ')[0]+'\n'+', '.join([i[0] for i in family])] = [clf.svm_loocv.best_acc,len(this_df)]
                except:
                    print(f'{val} failed')
                continue
                rep_groups = this_df.groupby('Replicates')
                tasks = []
                
                for rep, test_df in rep_groups:
                    for n_features_to_select in np.arange(1,51,5):
                        train_df = this_df.loc[this_df['Replicates'] != rep].copy()
                        tasks.append((rep, train_df, test_df.copy(), feature_names, truth_feature, n_features_to_select))

                # Run in parallel
                acc_results = defaultdict(list)
                all_list = defaultdict(list)
                with ProcessPoolExecutor() as executor:
                    for acc, best_feats,n_features_to_select in tqdm(executor.map(rfe_leave_one_rep_out, tasks), total=len(tasks),leave=True):
                        all_list[n_features_to_select].append(best_feats)
                        
                        acc_results[n_features_to_select].append(acc)
                        

                
                
                for n_features_to_select, full_list in all_list.items():
                    u,c = np.unique(np.array(full_list).ravel(),return_counts = True)
                    args = np.argsort(c)
                    u = u[args]
                    c = c[args]
                    #plt.figure('Hist')
                    #plt.bar(u, c/len(tasks), label = 'New data')
                    best_feats = read_pickle(os.path.join(feature_root,f'results - {val}',f'best_{n_features_to_select}_feature_names.pickle'))
                    feats = best_feats['All'][-n_features_to_select:]
                    #plt.bar(feats,np.ones(len(feats)),color='red',alpha=0.4)

                    in_best = [np.sum([l in feats for l in L]) for L in full_list]
                    #plt.figure('Hist of times in best')
                    #fig, axs = plt.subplots(1,2,figsize=(10,6))
                    #axs[0].hist(in_best,bins=np.arange(-0.5,26))
                    
                    #accuracies = [np.mean(ac) for ac in acc_results[n_features_to_select]]
                    #axs[1].scatter(in_best,accuracies)

                    
                    all_feats = {'Best':u[-n_features_to_select:]} #feats
                    all_accs = {}
                    these_tasks = [task for task in tasks if task[-1]==n_features_to_select]
                    for l,t,a in zip(full_list,these_tasks,acc_results[n_features_to_select]):
                        all_feats[t[0]] = l
                        all_accs[t[0]] = a
                    newroot = os.path.join(feature_root,f'results - {val}','LOO')
                    if not os.path.exists(newroot):
                        os.makedirs(newroot)
                    save_pickle(os.path.join(newroot,f'RFE_Features_{n_features_to_select}.pickle'),all_feats)
                    save_pickle(os.path.join(newroot,f'RFE_Accuracies_{n_features_to_select}.pickle'),all_accs)
                    
                    #fig.suptitle(val)
                    #plt.close(fig)
                    #fig.savefig(os.path.join(newroot,'Features in best.png'),dpi=300)
            try:
                plt.figure('acc-vs-n') 
                n_feats = list(acc_results.keys())
                my_accs = [np.mean([j for i in val for j in i]) for val in acc_results.values()]
                plt.plot(n_feats, my_accs,label=split_val)
                plt.xlabel('Number of Features')
                plt.ylabel('Accuracy')
                plt.legend(fontsize=8)
                plt.tight_layout()
            except: 
                pass
        plt.figure(figsize=(13,7))
        tot_correct, tot_num = 0,0
        for n, (family,fam_dict) in enumerate(all_feat_accs.items()):
            accs, nums = np.array(list(fam_dict.values())).T
            bars = plt.bar(fam_dict.keys(), accs, color=plt.get_cmap('tab10')(n))
            for bar, acc, num in zip(bars, accs, nums):
                bar_text = f'{acc:.1f}\n{int(num)}'
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, bar_text, 
                    ha='center', va='top', color='white', fontsize=8)
            correct = np.sum(accs*nums)
            tot_correct += correct
            num = nums.sum()
            tot_num += num
            fam_acc = correct/num
            bar = plt.bar(family[0][0], fam_acc, color=plt.get_cmap('tab20')(2*n+1))
            bar_text = f'{fam_acc:.1f}\n{int(num)}'
            plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() - 0.05, bar_text, 
                ha='center', va='top', color='white', fontsize=8)
            print(n)
        bar = plt.bar('Total',tot_correct/tot_num,color='black')
        bar_text = f'{tot_correct/tot_num:.1f}\n{int(tot_num)}'
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() - 0.05, bar_text, 
                ha='center', va='top', color='white', fontsize=8)
        plt.gca().axhline(95,linestyle='dashed',color='gray')
        plt.title(f'Accuracies using all 75 features\nNoise: {noise if len(noise)>0 else "None"}')
        # Final bar plot with noise on the x-axis and tot_correct/tot_num on the y-axis
        accuracy_values.append([tot_correct/tot_num,tot_num])
    # Plot the final bar chart


plt.figure(figsize=(8, 5))
noise_levels = [n if n is not None else 'None' for n in noise_levels ]
accs,nums = np.array(accuracy_values).T
bars = plt.bar(noise_levels,accs)
adjust_accs = [a*n/max(nums) for a,n in zip(accs,nums)]
bars2 = plt.bar(noise_levels,adjust_accs,alpha=0.3)
for bar, (acc,num) in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{acc:.1f}', 
        ha='center', va='top', color='white', fontsize=8)
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, str(int(num)), 
        ha='center', va='top', color='white', fontsize=8)
for bar, acc,num in zip(bars2, adjust_accs, nums):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{acc:.1f}', 
        ha='center', va='top', color='white', fontsize=8)
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5, str(int(max(nums))), 
        ha='center', va='top', color='white', fontsize=8)
plt.xlabel('Noise Level (Normally distribution STD - nm)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Noise Level')
plt.ylim(0, 100)

plt.show()


    
'''
mat = {}
for drug in tqdm(feat_df['Drug'].unique()):
    this_df = feat_df.loc[feat_df['Drug']==drug].copy()
    
    clf = ML(this_df,feature_names,truth_feature='Phenotype')
    clf.svm_loocv()
    mat[drug] = clf.svm_loocv.mat_dict['linear']
fig,axs = plt.subplots(3,4)
for ax, (drug, mat) in zip(axs.ravel(),mat.items()):
    plot_confusion(mat,ax=ax)
    ax.set_title(f'{drug}-{100*np.trace(mat)/np.sum(mat):.1f}')
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.suptitle('Using All Features')
plt.tight_layout()
'''
    