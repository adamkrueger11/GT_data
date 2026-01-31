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
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)
simplefilter(action='ignore', category=DeprecationWarning)


# Parameters
truth_feature = 'Genera'
min_count = 10
model_type = 'svm'  # 'rf' or 'svm'
home_only = False
n_splits = 100
n_features_list = np.arange(1,51)#[20]
include_fails = True

def get_xy(img):
    if img.shape[1]==1:
        X = np.arange(len(img)).reshape(len(img),1)
        Y = np.array([])
        return X, Y
    else:
        X = np.arange(len(img)).reshape(len(img),1)
        Y = np.arange(len(img[0])).reshape(1,len(img[0]))
        return X,Y
def plot_3d(data, new_fig = True,lims=[None,None],zlims=[None,None],consistent_lims=True,lat_res=1,view=(13.601693557484248, 7.066860584210744),count=50,cmap='Spectral_r'):
    X,Y = [i*lat_res for i in get_xy(data)]
    
    if new_fig:
        plt.figure()
        ax = plt.axes(projection = '3d')
    ax = plt.gca()
    if consistent_lims:
        zlims = lims
    p = ax.plot_surface(X,Y,data,cmap = cmap,vmin=lims[0], vmax=lims[1],rcount=count,ccount=count)
    ax.view_init(view[0],view[1])
    ax.set_zlim([zlims[0],zlims[1]])
    return p
import pickle
def save_pickle(filename,obj):
    with open(f'{filename}' + ('.pickle' if 'pickle' not in filename else ''),'wb') as handle:
        pickle.dump(obj,handle,protocol=pickle.HIGHEST_PROTOCOL)
        
def read_pickle(filename):
    with open(f'{filename}' + ('.pickle' if 'pickle' not in filename else ''),'rb') as handle:
        obj = pickle.load(handle)
    return obj

class_colors = {
    "your_method": "#daa03b",
    "control_method": "#4b8e9d",
    "susceptible": "#7fb069",
    "intermediate": "#a0a0a0",
    "resistant": "#b67219",
    "heteroresistant": "#8771b1"
}
colors = {
    "brand_orange": "#daa03b",
    "burnt_orange": "#b67219",
    "light_tan": "#f4e9d0",
    "charcoal": "#2f2f2f",
    "cool_gray": "#a0a0a0",
    "brand_blue": "#84cbd8",
    "slate_blue": "#4b8e9d",
    "cream": "#fffdf6",
    "accent_green": "#7fb069"
}
class_colors_list = list(colors.values())


plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=[
        colors["brand_orange"], colors["brand_blue"],
        colors["accent_green"], colors["cool_gray"],
        colors["burnt_orange"], colors["slate_blue"]
    ]),
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.labelcolor": colors["charcoal"],
    "xtick.color": colors["charcoal"],
    "ytick.color": colors["charcoal"]
})
plt.close('all')
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
    
    return acc, selected_features

# ----- Main function -----
def perform_rfe_svm(full_df, feature_names, truth_feature, root, n_features_to_select=20, n_splits=5):
    filename = root + f'/best_{n_features_to_select}_feature_names.pickle'
    
    if not os.path.exists(filename):
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
        for train, test in zip(train_groups, test_groups):
            train_df = full_df.loc[full_df['Replicates'].isin(train)].copy()
            test_df = full_df.loc[full_df['Replicates'].isin(test)].copy()
            tasks.append((train_df, test_df, feature_names, truth_feature, n_features_to_select))

        # Run in parallel
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(rfe_iteration, tasks), total=len(tasks), leave=False))

        accs, select_feats = zip(*results)
        accs = list(accs)
        select_feats = list(select_feats)

        # Plot results
        plt.figure('Accuracy Histogram')
        plt.hist(accs, 25)
        
        # Feature selection histogram
        u, c = np.unique(np.array(select_feats).ravel(), return_counts=True)
        args = np.argsort(c)
        best_feats = u[args][-n_features_to_select:]

        plt.figure('Common Features')
        plt.bar(u, c / n_splits * 100)
        plt.bar(best_feats, 100 * np.ones(len(best_feats)), alpha=0.2, color='red')
        plt.suptitle(f'Feature Selection Hist - {n_features_to_select} Features')

        save_pickle(filename, {
            'Best': best_feats,
            'Top5': u[args][-5:],
            'All': u[args]
        })

    else:
        best_feats = read_pickle(filename)['Best']

    # Final model and evaluation
    ml = ML(full_df, best_feats, truth_feature=truth_feature)
    ml.svm_loocv(include_fails=include_fails)
    ml.svm_loocv.confusion_matrices()

    return best_feats, ml.svm_loocv.best_acc

def perform_umap(full_df, best_feats, truth_feature, classes,n_components=5):
    y = [classes.tolist().index(f) for f in full_df[truth_feature].values]
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

    # Load your data
    allfeats={}
    all_dfs = []
    all_features = []
    all_best_feats = {n:[] for n in n_features_list}
    best_accs = {'Features': {}, 'UMAP': {}}
    for inc in tqdm([0,4],desc = 'Incubation Times'):
        all_data = {}
        root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/ID/{inc}hr/'
        full_df = pickle.load(open(root + 'features.pickle', 'rb'))
        subset = [i for i in full_df.columns if any([i.startswith(f'Ring{dir}power '+j) for j in ['cov','std'] for dir in ['L','P']])]
        full_df = full_df.drop(columns=subset, errors='ignore')
        full_df['Spot'] = full_df.apply(lambda row: '_'.join(row['Filename'].split('_')[:2]), axis=1)            
        # Filter classes
        all_classes, counts = np.unique(full_df[truth_feature], return_counts=True)
        classes = all_classes[counts >= min_count]
        full_df = full_df.loc[
            np.array([i in classes for i in full_df[truth_feature]]) & full_df['Usable']
        ].copy()
        # Find usable feature columns
        feature_names = [
            name for name, typ in zip(full_df.columns, full_df.dtypes)
            if typ == float and not any(i in name for i in ['CFU', 'OD','Spot'] + (['R', 'ring', 'Full'] if home_only else []))
        ]
        new_feature_names = [feat+f'_{inc}hr' for feat in feature_names]
        all_features.extend(new_feature_names)
        full_df.rename(columns=dict(zip(feature_names, new_feature_names)), inplace=True)
        all_dfs.append(full_df)
        best_accs['Features'][inc] = []
        best_accs['UMAP'][inc] = []
        # Save all open figures
        results_dir = root + f'{truth_feature}/results'
        os.makedirs(results_dir, exist_ok=True)
        
        for n_features in tqdm(n_features_list,leave=False):
            best_feats,best_acc = perform_rfe_svm(full_df, new_feature_names, truth_feature, results_dir, n_features_to_select=n_features, n_splits=n_splits)
            plt.suptitle(f'{inc}hr Data\n{n_features} Features')
            best_accs['Features'][inc].append(best_acc)

            umap_acc = perform_umap(full_df, best_feats, truth_feature, classes)
            plt.suptitle(f'{inc}hr Data - UMAP\n{n_features} Features')
            best_accs['UMAP'][inc].append(umap_acc)

            all_best_feats[n_features].extend(best_feats)

            for i in plt.get_fignums():
                fig = plt.figure(i)
                title = fig._suptitle.get_text() if fig._suptitle else f"Figure_{i}"
                filename = title.replace('\n', '').replace(' ', '_') + '.png'
                filepath = os.path.join(results_dir, filename)
                fig.savefig(filepath, dpi=300)
                plt.close(fig)



        if False:
            for fold in tqdm(os.listdir(root), leave=False):
                if f'{inc}hr' in fold:
                    all_data = all_data|read_pickle(root+'/'+fold+'/all_data.pickle')
            a = [i for i in all_data.keys() if i.startswith('1')][17]
            da = all_data[a]
            ra = [max(row) for row in da]
            e = [i for i in all_data.keys() if i.startswith('0')][10]
            de = all_data[e]
            re = [max(row) for row in de]
            p = [i for i in all_data.keys() if i.startswith('2')][10]
            dp = all_data[p]
            rp = [max(row) for row in dp]
            data = {a:da,e:de,p:dp}
            rings = {a:ra,e:re, p:rp}
            get_points = lambda arr: np.array([[i,np.argmax(row),max(row)] for i,row in enumerate(arr)])
            points = {i:get_points(di) for i,di in data.items()}
            path = 'MY PAPERS/Interferometry AST/Figures/Fig2_ID/'
            for i in data.keys():
                continue
                plot_3d(data[i],cmap='terrain',consistent_lims=False,zlims=[-.1, 8000],lims=[0,1500],count=150,view=[30,-40])
                plt.gca().axis('off')
                ax = plt.gca()
                x,y,z = points[i].T
                ax.plot3D(x,y,z+np.ones_like(z),color='red',alpha=1,zorder=10)
                fig = plt.gcf()
                #fig.savefig(path+i+'.svg')
                plt.figure('rings');plt.plot(points[i].T[2])


        
    #print(all_best_feats)

    ##
    # Combine the two dataframes in all_dfs on the 'Spot' column

    combined_df = pd.merge(
        all_dfs[0],
        all_dfs[1],
        on='Spot',
        how='inner',
        suffixes=('', '_drop')  # Keep original column names in df[0]
    )
    # Drop the "_drop" columns from df[1]
    combined_df = combined_df[[col for col in combined_df.columns if not col.endswith('_drop')]]

    combined_df['logCFU'] = combined_df['Plated CFU'].apply(np.log10)
    plt.figure()
    sns.histplot(combined_df, x='logCFU', hue='Genera',bins=50)
    plt.suptitle('Plated CFU Distribution')

    root = f'/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/ID/'
    # Save all open figures
    results_dir = root + f'{truth_feature}/results'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(root + f'{truth_feature}/results/combined_cfu_distribution.png', dpi=300)

    all_classes, counts = np.unique(combined_df[truth_feature], return_counts=True)
    classes = all_classes[counts >= min_count]

    best_accs['Features']['Together'] = []
    best_accs['UMAP']['Together'] = []
    best_accs['Features']['Combined'] = []
    best_accs['UMAP']['Combined'] = []

    all_used_features = []
    for n_feats in tqdm(n_features_list,leave=True):

        these_best_feats = all_best_feats[n_feats]
        ml = ML(combined_df, these_best_feats,truth_feature = truth_feature)
        ml.svm_loocv(include_fails=include_fails)
        ml.svm_loocv.confusion_matrices()
        plt.suptitle(f'Combined Time Data-{n_feats} Features')
        best_accs['Features']['Together'].append(ml.svm_loocv.best_acc)
        umap_acc = perform_umap(combined_df, these_best_feats, truth_feature, classes)
        best_accs['UMAP']['Together'].append(umap_acc)

        for i in plt.get_fignums():
            fig = plt.figure(i)
            title = fig._suptitle.get_text() if fig._suptitle else f"Figure_{i}"
            filename = title.replace('\n', '').replace(' ', '_') + '.png'
            filepath = os.path.join(results_dir, 'all-combined')
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, filename)
            fig.savefig(filepath, dpi=300)
            plt.close(fig)


        best_feats, best_acc= perform_rfe_svm(combined_df, all_features, truth_feature, results_dir+'/', n_features_to_select=n_feats, n_splits=n_splits)
        plt.suptitle(f'{inc}hr Data\n{n_features} Features')
        best_accs['Features']['Combined'].append(best_acc)

        umap_acc = perform_umap(combined_df, best_feats, truth_feature, classes)
        plt.suptitle(f'{inc}hr Data - UMAP\n{n_features} Features')
        best_accs['UMAP']['Combined'].append(umap_acc)


        for i in plt.get_fignums():
            fig = plt.figure(i)
            title = fig._suptitle.get_text() if fig._suptitle else f"Figure_{i}"
            filename = title.replace('\n', '').replace(' ', '_') + '.png'
            filepath = os.path.join(results_dir, 'all-together')
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, filename)
            fig.savefig(filepath, dpi=300)
            plt.close(fig)

        # Calculate intersection and differences between all_best_feats and best_feats
        intersection = set(all_best_feats[n_feats]).intersection(best_feats)

        # Print the results
        #print(f"Intersection ({len(intersection)}):", intersection)
        all_used_features.extend(these_best_feats)

    i=0
    all_used_features = list(set(all_used_features))
    for feat in all_used_features:
        if 'power' not in feat:
            print(feat)
            i += 1
    print('\n\n\n\n')
    print(f"Total features used: {len(all_used_features)}")
    print(f"Total non-power features used: {i}")

    plt.figure(figsize=(10, 6))
    feature_types = list(best_accs.keys())
    data_types = list(best_accs[feature_types[0]].keys())
    for feature_type, acc_dict in best_accs.items():
        for data_type, acc_values in acc_dict.items():
            if feature_type.lower() == 'umap':
                continue
            plt.plot(
                n_features_list, acc_values, 
                label=f"{feature_type} - {data_type}", 
                linestyle=['-','--'][feature_types.index(feature_type)], 
                color=class_colors_list[data_types.index(data_type)]
            )
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Features')
    plt.legend()
    plt.grid(True)

    figroot = '/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/ID/'
    plt.savefig(figroot + f'acc_vs_n-features-{truth_feature}.png', dpi=300)
    
    # Save as Python figure object
    with open(os.path.join(root, f'Accuracy_vs_Number_of_Features-{truth_feature}.fig.pickle'), 'wb') as fig_file:
        pickle.dump(plt.gcf(), fig_file)

    plt.show()