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
from concurrent.futures import ProcessPoolExecutor
sys.path.append('/Users/adamkrueger/Downloads/Yunker Lab')
from functions import save_pickle, read_pickle
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)

truth_feature = 'Phenotype'
min_count = 10
model_type = 'svm'  # 'rf' or 'svm'
home_only = False
n_splits = 100
n_features_list = np.arange(1,51)#[20]
include_fails = True

colors = {
    "brand_orange": "#daa03b",
    "burnt_orange": "#b67219",
    "light_tan": "#f4e9d0",
    "charcoal": "#2f2f2f",
    "cool_gray": "#a0a0a0",
    "brand_blue": "#84cbd8",
    "slate_blue": "#4b8e9d",
    "cream": "#fffdf6",
    "accent_green": "#7fb069",

    # Added below:
    "dusty_rose": "#d4a5a5",
    "sage_green": "#a8c3a0",
    "steel_blue": "#5a7d9a",
    "warm_sand": "#e1c699",
    "muted_mustard": "#c9a941",
    "cool_mint": "#b8e2dc",
    "deep_teal": "#2c5f60",
    "soft_lilac": "#cdb4db",
    "terracotta": "#d98473",
    "olive": "#7a8450",
    "pebble_gray": "#d8d8d8"
}
class_colors_list = list(colors.values())


# ----- Helper function must be top-level -----
def rfe_iteration(args):
    train_df, test_df, feature_names, truth_feature, n_features_to_select = args
    base_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
    rfe.fit(train_df[feature_names], train_df[truth_feature])
    
    selected_features = train_df[feature_names].columns[rfe.support_].tolist()
    
    ml = ML(train_df, selected_features, truth_feature=truth_feature)
    ml.svm_test(test_df)
    
    result = np.array(ml.svm_result)
    truth = np.array(ml.__svmy__)
    acc = np.mean(truth == result)
    
    return acc, selected_features, n_features_to_select

# ----- Main function -----
def perform_rfe_svm(full_df, feature_names, truth_feature, root, n_features_list=[20], n_splits=100):
    #not os.path.exists(filename) or 
    if True:
        scaler = StandardScaler()
        full_df[feature_names] = scaler.fit_transform(full_df[feature_names])
        
        # Build replicate group splits
        groups = {i: [] for i in full_df[truth_feature].unique()}
        for phen, rep in full_df[[truth_feature, 'Replicates']].values:
            if rep not in groups[phen]:
                groups[phen].append(rep)

        train_groups = []
        test_groups = []
        for _ in range(n_splits):
            temp_test, temp_train = [], []
            for vals in groups.values():
                vals = vals.copy()
                np.random.shuffle(vals)
                split = int(np.floor(len(vals) * 0.1))
                temp_test.extend(vals[:split])
                temp_train.extend(vals[split:])
            train_groups.append(temp_train)
            test_groups.append(temp_test)

        # Prepare parallel tasks
        tasks = []
        for n_features_to_select in n_features_list:
            for train, test in zip(train_groups, test_groups):
                train_df = full_df.loc[full_df['Replicates'].isin(train)].copy()
                test_df = full_df.loc[full_df['Replicates'].isin(test)].copy()
                tasks.append((train_df, test_df, feature_names, truth_feature, n_features_to_select))

        # Run in parallel
        all_best_feats, best_accs = {}, {}
        with ProcessPoolExecutor() as executor:
            all_results = list(tqdm(executor.map(rfe_iteration, tasks), total=len(tasks), leave=False))
        

        for n_features_to_select in n_features_list:
            results = [res for res in all_results if res[2] ==n_features_to_select ]

            filename = root + f'/best_{n_features_to_select}_feature_names.pickle'

            accs, select_feats, _ = zip(*results)
            accs = list(accs)
            select_feats = list(select_feats)

            # Plot results
            #plt.figure('Accuracy Histogram')
            #plt.hist(accs, 25)
            
            # Feature selection histogram
            u, c = np.unique(np.array(select_feats).ravel(), return_counts=True)
            args = np.argsort(c)
            best_feats = u[args][-n_features_to_select:]

            #plt.figure('Common Features')
            #plt.bar(u[args], c[args] / n_splits * 100)
            #plt.bar(best_feats, 100 * np.ones(len(best_feats)), alpha=0.2, color='red')
            #plt.suptitle(f'Feature Selection Hist - {n_features_to_select} Features')

            save_pickle(filename, {
                'Best': best_feats,
                'Top5': u[args][-5:],
                'All': u[args]
            })

            ml = ML(full_df, best_feats, truth_feature=truth_feature)
            ml.svm_loocv(include_fails=include_fails)
            #ml.svm_loocv.confusion_matrices()
            best_accs[n_features_to_select] = ml.svm_loocv.best_acc
            all_best_feats[n_features_to_select] = best_feats

    else:
        best_feats = read_pickle(filename)['Best']

    # Final model and evaluation
    #ml = ML(full_df, best_feats, truth_feature=truth_feature)
    #ml.svm_loocv(include_fails=include_fails)
    #ml.svm_loocv.confusion_matrices()

    return all_best_feats, best_accs

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


if __name__ == '__main__':
    root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/Rapid AST/'
        
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
    full_df['Family'] = full_df['Family'].replace('Enterobacteriaceae', 'Enterobacterales')
    full_df['Family'] = full_df['Family'].replace('Moraxellaceae', 'Acinetobacter')
    


    # Check for missing values in the selected feature

    best_accs = {'Features': {}, 'UMAP': {}}
    classes = ('R','S')
    splits = [('Drug',val) for val in full_df['Drug'].unique()] + [('Family', val) for val in full_df['Family'].unique()] + [('Family', tuple(list(full_df['Family'].unique())))]


    for col, split_val in tqdm(splits,desc='Drug-Bugs'):

        values = (split_val,) if isinstance(split_val, str) else split_val
        temp_df = full_df.loc[full_df[col].isin(values)].copy()
        if col != 'Family':
            families = temp_df['Family'].unique().tolist()
            families += [tuple(families)] if len(families) > 1 else []
        else:
            families = [split_val]
        for family in tqdm(families,desc='Families',leave=False):
            family = (family,) if not isinstance(family, tuple) else family
            this_df = temp_df.loc[temp_df['Family'].isin(family)].copy()

            val = f'{split_val}' + (f' - {family}' if col!='Family' else '')
            results_dir = root + f'results - {val}'

            best_accs['Features'][val] = []
            best_accs['UMAP'][val] = []

            os.makedirs(results_dir, exist_ok=True)
            #for n_features in tqdm(n_features_list,leave=True, desc='Number of Features'):
                
            best_feats,best_accs['Features'][val] = perform_rfe_svm(this_df, feature_names, truth_feature, results_dir, n_features_list=n_features_list, n_splits=n_splits)
            '''    plt.suptitle(f'{n_features} Features')
                best_accs['Features'][val].append(best_acc)

                umap_acc = perform_umap(this_df, best_feats, truth_feature, classes)
                plt.suptitle(f'UMAP\n{n_features} Features')
                best_accs['UMAP'][val].append(umap_acc)


                # Save all open figures
                

                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    title = fig._suptitle.get_text() if fig._suptitle else f"Figure_{i}"
                    filename = title.replace('\n', '').replace(' ', '_') + '.png'
                    filepath = os.path.join(results_dir, filename)
                    fig.savefig(filepath, dpi=300)
                    plt.close(fig)'''



    plt.figure('acc-vs-n',figsize=(10, 6))
    feature_types = list(best_accs.keys())
    for feature_type, acc_dict in best_accs.items():
        for n_group, (class_type, acc_values) in enumerate(acc_dict.items()):
            if feature_type.lower() == 'umap':
                continue
            plt.plot(
                acc_values.keys(), acc_values.values(), 
                label=f"{feature_type} - {class_type}",
                color=class_colors_list[n_group],
                linestyle='--' if feature_type.lower() == 'umap' else '-'
            )

    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Number of Features\n{classes}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save as PNG
    plt.savefig(
        os.path.join(root, f'Accuracy_vs_Number_of_Features.png'), dpi=300
    )
    
    # Save as Python figure object
    with open(os.path.join(root, f'Accuracy_vs_Number_of_Features.fig.pickle'), 'wb') as fig_file:
        pickle.dump(plt.gcf(), fig_file)

    plt.show()
