import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
from functions import plot_contour,plot_df
from collections import defaultdict
from scipy.stats import spearmanr


drugs_to_examine = []
fam_to_examine = []
to_examine = drugs_to_examine + fam_to_examine

feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'

path = os.path.join(feature_root,'RFE-SVD Results','subset_TZP-Enterobacteriaceae')
all_paths = [os.path.join(path,i) for i in os.listdir(path) if os.path.isdir(os.path.join(path,i))]

avg_accs = {}
accs = defaultdict(list)
for path in all_paths:
    for file in os.listdir(path):
        if '.npy' not in file: continue
        num_images_used = int(os.path.split(path)[1].split('-')[0])
        data =  np.load(os.path.join(path,file)).T
        if 'avg' in file: 
            avg_accs[num_images_used] = data
        else:
            accs[num_images_used].append(data)

nums = np.array(list(avg_accs.keys()))
args = np.argsort(nums)
nums = nums[args]
avgs = np.array(list(avg_accs.values()))[args]


final = avgs[-1].flatten()
correlations,pvals = np.array([spearmanr(arr.ravel(), final) for arr in avgs]).T


df = pd.DataFrame({
    'Number': nums,
    'Data-array': [arr for arr in avgs],
    'Correlation': correlations,
    'Pval': pvals
    })


plot_df(df,sort='Number',vlims=[0.85,1])

plt.figure(figsize=(13,7))
sns.scatterplot(df,x='Number',y='Correlation',style=np.where(df['Pval']<=0.05,'Significant','Not Significant'))
plt.xlabel('Number of images used in subsets')
plt.suptitle('Averaged over replicates at N')

all_df = pd.DataFrame()
for num, data in accs.items():
    correlations, pvals = np.array([spearmanr(arr.ravel(), final) for arr in data]).T
    this_df = pd.DataFrame({
        'Number': num * np.ones(len(data)),
        'Data-array': [arr for arr in data],
        'Correlation': correlations,
        'Pval': pvals
    })
    
    all_df = pd.concat([all_df,this_df],ignore_index=True, axis=0)

plot_df(all_df, split='Number',sort='Correlation',close=False,vlims=[0.85,1])
plt.figure(figsize=(13,7))
sns.violinplot(all_df,x='Number',y='Correlation',split=True)


def get_array_differences(arrays, metric='l2'):
    diffs = []
    for i in range(0, len(arrays)-1):
        diff = arrays[-1] - arrays[i]
        if metric == 'l2':
            diffs.append(np.linalg.norm(diff))  # Frobenius norm
        elif metric == 'l1':
            diffs.append(np.sum(np.abs(diff)))
        elif metric == 'max':
            diffs.append(np.max(np.abs(diff)))
    return np.array(diffs)

plt.figure(figsize=(10,6))
for metric in ['l2','l1','max']:
    diffs = get_array_differences(avgs, metric=metric)
    plt.plot(nums[:-1],diffs/diffs[0], marker='o',label=metric)
plt.title('Differences Between Sizes of Subsets and the Full Set')
plt.xlabel('Number of Images in Subsest')
plt.ylabel('Scaled Difference Metric')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(title='Diff Metric')

plt.show()
