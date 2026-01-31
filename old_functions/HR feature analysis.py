import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import os 
import umap
import pandas as pd 
import seaborn as sns
import sys 
from ML_functions import ML
from warnings import simplefilter
import matplotlib.pyplot as plt
sys.path.append('/Users/adamkrueger/Downloads/Yunker Lab')
from functions import save_pickle, read_pickle
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)

truth_feature = 'Classification'
min_count = 10
model_type = 'svm'  # 'rf' or 'svm'
home_only = False
n_splits = 100
n_features_list = np.arange(1,51)#[20]
include_fails = True

def perform_rfe_svm(full_df, feature_names, truth_feature, root, n_features_to_select=20, n_splits=n_splits):
    # Rescale all features (important for SVM especially)
    filename = root+f'/best_{n_features_to_select}_feature_names.pickle'
    if not os.path.exists(filename):
        scaler = StandardScaler()
        full_df[feature_names] = scaler.fit_transform(full_df[feature_names])
        
        base_model = SVC(kernel='linear', class_weight='balanced', random_state=42)

        groups = {i:[] for i in full_df[truth_feature].unique()}
        for phen,rep in full_df[[truth_feature,'Replicates']].values:
            if rep in groups[phen]: continue
            groups[phen].append(rep)
        train_groups = []
        test_groups  = []
        for _ in range(n_splits):  # Replace "i" with "_" since it's unused
            temp_test,temp_train = [],[]
            for _,vals in groups.items():  # Replace "group" with "_" since it's unused
                np.random.shuffle(vals)
                split = int(np.floor(len(vals)* 0.1))
                temp_test.extend(vals[:split])
                temp_train.extend(vals[split:])
            train_groups.append(temp_train)
            test_groups.append(temp_test)
        accs = []
        select_feats = []
        
        for train,test in tqdm(zip(train_groups,test_groups),total=len(test_groups),leave=False):
            train_df = full_df.loc[full_df['Replicates'].isin(train)].copy()
            test_df  = full_df.loc[full_df['Replicates'].isin(test) ].copy()
            
            rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
            rfe.fit(train_df[feature_names], train_df[truth_feature])
            selected_features = train_df[feature_names].columns[rfe.support_].tolist()
            ml = ML(train_df, selected_features,truth_feature = truth_feature)
            ml.svm_test(test_df)
            result = np.array(ml.svm_result)
            truth  = np.array(ml.__svmy__)
            acc = np.mean(truth==result)
            accs.append(acc)
            select_feats.append(selected_features)
        
        plt.figure()
        plt.hist(accs,25)
        

        u,c = np.unique(np.array(select_feats).ravel(),return_counts=True)
        args = np.argsort(c)
        best_feats = u[args][-n_features_to_select:]
        plt.figure()
        plt.bar(u,c/n_splits*100)
        plt.bar(best_feats,100*np.ones(len(best_feats)),alpha=0.2,color='red')
        plt.suptitle(f'Feature Selection Hist - {n_features_to_select} Features')
        save_pickle(filename,{'Best':u[args][-n_features_to_select:],'Top5':u[args][-5],'All':u[args]})
    else:
        best_feats = read_pickle(filename)['Best']
    ml = ML(full_df, best_feats, truth_feature = truth_feature)
    ml.svm_loocv(include_fails=include_fails)
    ml.svm_loocv.confusion_matrices()
    return best_feats, ml.svm_loocv.best_acc

def perform_umap(full_df, best_feats, truth_feature, classes,n_components=5):
    y = [list(classes).index(f) for f in full_df[truth_feature].values]
    X = full_df[best_feats].values
    # Optional: standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    umap_feats = [f'UMAP_{i}' for i in range(1,n_components+1)]
    df_umap = pd.DataFrame(X_umap, columns=umap_feats)
    df_umap['Label'] = [classes[i] for i in y]  # Convert numeric labels to names
    df_umap['Replicates'] = full_df['Replicates'].values
    df_umap['Filename'] = full_df['Filename'].values
    df_umap['Spot'] = full_df['Spot'].values
    sns.pairplot(df_umap, hue='Label', plot_kws={'s': 15})
    plt.suptitle(f'UMAP Pairplot - {len(best_feats)} Features Selected')
    ml = ML(df_umap, umap_feats, truth_feature = 'Label')
    ml.svm_loocv(include_fails=include_fails)
    ml.svm_loocv.confusion_matrices()
    acc = ml.svm_loocv.best_acc
    return acc


root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/HR/'
    
full_df = pickle.load(open(root+ 'features.pickle','rb'))
feature_names = [
    name for name, typ in zip(full_df.columns, full_df.dtypes)
    if typ == float and not any(i in name for i in ['CFU', 'OD','Spot','Folder','Cells','Res', 'Number'])
]


feature_names = [
    feature for feature in feature_names 
    if np.std(full_df[feature]) != 0
]

full_df['Spot'] = full_df['Replicates'].astype(str)
#print(full_df.shape)
#full_df = full_df.loc[full_df['Classification'] != 'R'].copy() #####SHOULD I REMOVE THE R IMAGES??
#print(full_df.shape)


# Check for missing values in the selected feature

best_accs = {'Features': {}, 'UMAP': {}}
all_classes = [tuple(['R','H','S']), tuple(['H','S'])]
for classes in all_classes:
    best_accs['Features'][classes] = []
    best_accs['UMAP'][classes] = []
    all_best_feats = {n:[] for n in n_features_list}
    this_df = full_df.loc[full_df[truth_feature].isin(classes)].copy()

    results_dir = root + f'results - {classes}'
    os.makedirs(results_dir, exist_ok=True)

    for n_features in tqdm(n_features_list,leave=True):

        best_feats,best_acc = perform_rfe_svm(this_df, feature_names, truth_feature, results_dir, n_features_to_select=n_features, n_splits=n_splits)
        plt.suptitle(f'{n_features} Features')
        best_accs['Features'][classes].append(best_acc)

        umap_acc = perform_umap(this_df, best_feats, truth_feature, classes)
        plt.suptitle(f'UMAP\n{n_features} Features')
        best_accs['UMAP'][classes].append(umap_acc)

        all_best_feats[n_features].extend(best_feats)


        # Save all open figures
        

        for i in plt.get_fignums():
            fig = plt.figure(i)
            title = fig._suptitle.get_text() if fig._suptitle else f"Figure_{i}"
            filename = title.replace('\n', '').replace(' ', '_') + '.png'
            filepath = os.path.join(results_dir, filename)
            fig.savefig(filepath, dpi=300)
            plt.close(fig)


class_colors = {
    "your_method": "#daa03b",
    "control_method": "#4b8e9d",
    "susceptible": "#7fb069",
    "intermediate": "#a0a0a0",
    "resistant": "#b67219",
    "heteroresistant": "#8771b1"
}
class_colors_list = list(class_colors.values())
plt.figure('acc-vs-n',figsize=(10, 6))
feature_types = list(best_accs.keys())
for feature_type, acc_dict in best_accs.items():
    for class_type, acc_values in acc_dict.items():
        if feature_type.lower() == 'umap':
            continue
        plt.plot(
            n_features_list, acc_values, 
            label=f"{feature_type} - {class_type}",
            color=class_colors_list[all_classes.index(class_type)],
            linestyle='--' if feature_type.lower() == 'umap' else '-'
        )

plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs Number of Features\n{classes}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(root, f'Accuracy_vs_Number_of_Features.png'), dpi=300
)

# Save as Python figure object
with open(os.path.join(root, f'Accuracy_vs_Number_of_Features.fig.pickle'), 'wb') as fig_file:
    pickle.dump(plt.gcf(), fig_file)

plt.show()
