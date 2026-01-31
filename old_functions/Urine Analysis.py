import numpy as np 
import os
import sys 
from ML_functions import ML
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
sys.path.append('/Users/adamkrueger/Downloads/Yunker Lab')
from functions import save_pickle, read_pickle, plot_df, make_confusion_from_results,plot_confusion
from umap import UMAP
import seaborn as sns

root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Urine/'

feature_root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST'
training_feats = read_pickle(os.path.join(feature_root,'features.pickle'))

feat_df = read_pickle(root+'features.pickle')
data_df = read_pickle(root+'data.pickle')
data_df = data_df.loc[data_df['Filename'].isin(feat_df['Filename'].values)].copy()
data_df['Result'] = ['' for _ in data_df.iterrows()]

#plot_df(data_df,split='Drug',sort='FileBase',background='Phenotype')

feat_df = feat_df.loc[(feat_df['Phenotype']!='H') & (feat_df['Plated CFU'].astype(float)>1e4)]

grouped_feat_df = feat_df.groupby('Drug')

accfig = plt.figure('Accuracy')
umap_fig, umap_axs = plt.subplots(2,2)
conf_fig, conf_axs = plt.subplots(2,2)
data_df['Result'] = ['' for i in range(len(data_df))]
for n, (drug, test_df) in enumerate(tqdm(grouped_feat_df)):
    
    accs = []
    val_accs = []
    for n_feats in tqdm(range(1,51),leave=False):
        folder = [f for f in os.listdir(feature_root) if drug in f and 'Enterobacterales' in f][0]
        feature_names = read_pickle(os.path.join(feature_root,folder,f'best_{n_feats}_feature_names.pickle'))['Best']

        train_df = training_feats.loc[(training_feats['Drug']==drug) & (training_feats['Family']=='Enterobacteriaceae')].copy() 
        
        truth, model, names = [],[],[]
        urine_test_groups = test_df.groupby('Replicates')
        for rep, urine_test_df in urine_test_groups:
            this_train_df = pd.concat([train_df,test_df.loc[test_df['Replicates']!=rep]],ignore_index=True,axis=0)
            this_test_df = urine_test_df.copy()
            classifier = ML(this_train_df, feature_names=feature_names, truth_feature='Phenotype')

            classifier.svm_test(urine_test_df)
        
            truth.extend([classifier.classes[int(i)] for i in classifier.__svmy__ ])
            model.extend([classifier.classes[int(i)] for i in classifier.svm_result])
            names.extend(urine_test_df['Filename'].values)
        
        results_df = pd.DataFrame()
        results_df['Truth'] = truth 
        results_df['model'] = model
        mat = make_confusion_from_results(results_df,model='model',truth='Truth')
        accs.append(np.trace(mat)/np.sum(mat) * 100)

        clf = ML(train_df, feature_names = feature_names, truth_feature='Phenotype')
        clf.svm_loocv()
        val_accs.append(clf.svm_loocv.best_acc)
    # Perform UMAP dimensionality reduction
    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    train_embedding = umap.fit_transform(train_df[feature_names])
    test_embedding = umap.transform(test_df[feature_names])

    # Create a DataFrame for train embedding
    train_umap_df = pd.DataFrame(train_embedding, columns=['UMAP1', 'UMAP2'])
    train_umap_df['Phenotype'] = train_df['Phenotype'].values
    train_umap_df['Dataset'] = 'Training'

    # Create a DataFrame for test embedding
    test_umap_df = pd.DataFrame(test_embedding, columns=['UMAP1', 'UMAP2'])
    test_umap_df['Phenotype'] = test_df['Phenotype'].values
    test_umap_df['Dataset'] = 'Testing'

    # Combine train and test DataFrames into one
    combined_umap_df = pd.concat([train_umap_df, test_umap_df], ignore_index=True)

    # Plot train data
    sns.scatterplot(data=combined_umap_df, x='UMAP1', y='UMAP2', hue='Phenotype', style='Dataset', alpha=0.7, ax=umap_axs.ravel()[n],legend=n==0)

    plt.title(drug)
    #data_df.loc[(data_df['Drug']==drug) & (data_df['Phenotype']!='H'),'Result'] = [f'{row.Truth}->{row.model}' for row in results_df.itertuples()]
    #mask = (data_df['Drug']==drug) & (data_df['Phenotype']=='H')
    #data_df.loc[mask,'Result'] = ['H' for _ in range(np.sum(mask))]
    if len(accs)>0:
        plot_confusion(mat,labels=['R','S'],ax = conf_axs.ravel()[n])
        conf_axs.ravel()[n].set_title(drug)
        plt.figure('Accuracy')
        lines = plt.plot(np.arange(1,51),accs,label=drug)
        plt.plot(np.arange(1,51),val_accs,linestyle='--',label=None, color=lines[0].get_color())
        for name,t,m in zip(names,truth,model):
            data_df.loc[data_df['Filename']==name,'Result'] = [f'{t}->{m}']
#plt.grid(True)
#plot_df(data_df,split='Patient',sort=['Drug','Replicate'],background='Result',close=False)
handles, labels = umap_axs.ravel()[0].get_legend_handles_labels()
umap_axs.ravel()[0].get_legend().remove()

# Add to the figure (can adjust location)
umap_fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=6,
    bbox_to_anchor=(0.5, 0.99),
    fontsize=8
)
umap_fig.tight_layout()
umap_fig.subplots_adjust(top=0.9)


plot_df(data_df,split=f'Drug',sort=['Patient','Replicate','Phenotype-H'],background='Result',close=False)
plt.pause(0.1)
'''
feat_names = read_pickle(os.path.join(feature_root,folder,f'best_{n_feats}_feature_names.pickle'))['All']
print(feat_names)
print(len(feat_names))

umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=5, random_state=42)
training_feats = training_feats.loc[training_feats['Drug']=='TZP']
train_umap = umap.fit_transform(training_feats[feat_names])

# Create a DataFrame for train embedding
train_umap_df = pd.DataFrame(train_umap, columns=[f'UMAP{i}' for i in range(1,6)])
train_umap_df['Phenotype'] = training_feats['Phenotype'].values
sns.pairplot(train_umap_df,hue='Phenotype',vars=[f'UMAP{i}' for i in range(1,6)])
'''



drug = 'TZP'
temp1 = training_feats.loc[training_feats['Drug']==drug].copy()
temp1['Dataset'] = ['PBS' for _ in range(temp1.shape[0])]
temp2 = feat_df.loc[feat_df['Drug']==drug].copy()
temp2['Dataset'] = ['Urine' for _ in range(temp2.shape[0])]
large = pd.concat([temp1,temp2],ignore_index=True,axis=0)
large['Group'] = large['Phenotype'] + "_" + large['Dataset']


hue = 'Group'
if hue == 'Group':
    hue_order = ['S_PBS', 'R_PBS', 'S_Urine', 'R_Urine']
    markers = ['o', 's', 'o', 's']
else:
    hue_order,markers = None, None
sns.pairplot(large[list(feature_names)+[hue]],vars=feature_names[:5],hue=hue,markers=markers,hue_order=hue_order)
sns.pairplot(large[list(feature_names)+[hue]],vars=feature_names[5:10],hue=hue,markers=markers,hue_order=hue_order)
#sns.pairplot(large[list(feature_names)+[hue]],vars=feature_names[10:15],hue=hue,markers=markers,hue_order=hue_order)
plt.show()