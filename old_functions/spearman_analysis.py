import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
from functions import plot_contour
from collections import defaultdict
from statsmodels.stats.proportion import proportion_confint
import pickle

drugs_to_examine = []
fam_to_examine = []
to_examine = drugs_to_examine + fam_to_examine

feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'
feat_df = pickle.load(open(os.path.join(feature_root, 'features.pickle'), 'rb'))
path = os.path.join(feature_root,'RFE-SVD Results')

accs = {}
for file in os.listdir(path):
    if '.npy' not in file: continue
    drugbug = file[:-4]
    accs[drugbug] = np.load(os.path.join(path,f'{drugbug}.npy'))

pca_df = pd.DataFrame(columns=['RFE', 'PCA corr', 'PCA pval', 'DrugBug'])
rfe_df = pd.DataFrame(columns=['PCA', 'RFE corr', 'RFE pval', 'DrugBug'])

all_accs = defaultdict(lambda: defaultdict(list))
for drugbug, arr in accs.items():
    if len(drugs_to_examine+fam_to_examine) > 0 and not any([db.lower() in drugbug.lower() for db in to_examine]):
        continue
    n_rfe, n_pca = arr.shape
    
    # Correlate each PCA column with RFE index
    rfe_index = np.arange(10,41)
    pca_index = np.logspace(-3,np.log10(0.5),11)

    rfe_corrs = np.array([[pca] + list(spearmanr(rfe_index, arr[:, i])) for i,pca in enumerate(pca_index)])
    pca_corrs = np.array([[rfe] + list(spearmanr(pca_index, arr[i, :])) for i,rfe in enumerate(rfe_index)])

    for i, (rfe, corr, pval) in enumerate(pca_corrs):
        pca_df.loc[len(pca_df)] = [rfe, corr, pval, drugbug]
    # Correlate each RFE row with PCA index
    
    for i, (pca, corr, pval) in enumerate(rfe_corrs):
        rfe_df.loc[len(rfe_df)] = [pca, corr, pval, drugbug]

    mask_row = rfe_corrs.T[2] > 0.05
    mask_col  = pca_corrs.T[2] > 0.05
    mask = np.outer(mask_row,mask_col)
    final_arr = arr.T * mask
    fig,axs = plt.subplots(1,2,figsize=(12,6))
    plot_contour(final_arr,ax = axs[1])
    plot_contour(arr.T,ax = axs[0])
    acc = np.max(arr)#np.mean(final_arr[final_arr!=0])
    drug,fam = drugbug.split('-')
    num = feat_df.loc[(feat_df['Family'] == fam) & (feat_df['Drug'] == drug)].shape[0]
    all_accs[fam][drug+'\n'+fam[0]] = [acc*100, num]

    plt.suptitle(f'Drug: {drug}\nFam: {fam}\nAccuracy: {acc*100:.1f}')
    

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x='RFE',
    y='PCA corr',
    hue='DrugBug',
    style=np.where(pca_df['PCA pval'] <= 0.05, 'Significant', 'Not Significant'),  # style by significance
    markers={'Not Significant': 'o', 'Significant': 'X'},
    s=100
)
plt.axhline(0, color='gray', linestyle='--')
plt.title('PCA Correlation vs RFE Features')
plt.tight_layout()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=rfe_df,
    x='PCA',
    y='RFE corr',
    hue='DrugBug',
    style=np.where(rfe_df['RFE pval'] <= 0.05, 'Significant', 'Not Significant'),
    markers={'Not Significant': 'o', 'Significant': 'X'},
    s=100
)
plt.axhline(0, color='gray', linestyle='--')
plt.title('RFE Correlation vs PCA Components')
try:
    plt.gca().set_xscale('log')
except:
    pass 
plt.xlabel('PCA Variance Remaining')
plt.tight_layout()

plt.close('all')
finalfig = plt.figure('Bar Accuracies', figsize=(13, 7))
finalfig.suptitle('Accuracy vs Drug\nUsing Maximum Accuracy from PCA and RFE Param Sweep', fontsize=16)

tot_correct, tot_num = 0, 0
colors = plt.get_cmap('tab10').colors
family_color_map = {family: colors[i % len(colors)] for i, family in enumerate(sorted(all_accs.keys()))}
xtick_labels = []

# Loop through families
for family_idx, (family, fam_dict) in enumerate(sorted(all_accs.items())):
    # Extract data for all drugs in this family
    accs, nums = np.array(list(fam_dict.values())).T
    drugs = list(fam_dict.keys())
    # Bar color for this family
    family_color = plt.get_cmap('tab10')(family_idx)
    family_color_map[family] = family_color

    # Plot each drug bar
    for drug, acc, n in zip(drugs, accs, nums):
        k = int(acc/100 * n)  # number correct
        ci_low, ci_high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
        bar_height = acc
        x_val = f'{drug}-{family}'
        yerr = 100*np.array([[max(acc/100 - ci_low, 0)], [max(ci_high - acc/100, 0)]])
        plt.bar(x_val, bar_height, yerr=yerr, capsize=3, color=family_color)

        bar_text = f'{acc:.1f}\n{int(n)}'
        y_text = (acc/100 - (acc/100 - ci_low)) * 100
        plt.text(x_val, y_text - 1, bar_text, ha='center', va='top', color='white', fontsize=8)

    # Family average bar
    correct = np.sum(accs * nums / 100)
    num = np.sum(nums)
    fam_acc = correct / num
    tot_correct += correct
    tot_num += num

    # Lighter family color for avg
    lighter_color = plt.cm.tab20(2 * family_idx + 1)
    bar = plt.bar(family[0], 100 * fam_acc, color=lighter_color)
    bar_text = f'{fam_acc * 100:.1f}\n{int(num)}'
    plt.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() - 1, bar_text,
             ha='center', va='top', color='white', fontsize=8)

# Total bar
total_acc = tot_correct / tot_num
total_bar = plt.bar('Total', 100 * total_acc, color='black')
bar_text = f'{100 * total_acc:.1f}\n{int(tot_num)}'
plt.text(total_bar[0].get_x() + total_bar[0].get_width() / 2, total_bar[0].get_height() - 1,
         bar_text, ha='center', va='top', color='white', fontsize=8)

# Dashed reference line at 95%
plt.gca().axhline(95, linestyle='dashed', color='gray')

# Set xticks to drug names only
xticks = plt.xticks()[0]
xtick_labels = [tick.get_text().split('\n')[0] for tick in plt.gca().get_xticklabels()]
plt.xticks(ticks=xticks, labels=xtick_labels)

plt.ylim((0, 105))
plt.tight_layout()
plt.show()