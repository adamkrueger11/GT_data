'''More classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py'''
import numpy as np 
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def rescale_features(features,names,scaling=None):
    scaled_df = features.copy()
    if scaling is None:
        means = np.mean(features[names].values,axis=0)
        stds  = np.std(features[names].values,axis=0)
    else:
        scaling = np.array(scaling)
        means = scaling[0]
        stds  = scaling[1]
    dat= np.array(features[names])
    dat = (dat-means.reshape(1,-1))/stds.reshape(1,-1)
    scaled_df[names] = dat
    scaling = np.array([means,stds])
    return scaled_df, scaling
def define_subplot_size(num):
    area = np.array([[i*j if j<=i and i*j>=num else 1e7 for i in range(1,num+1)] for j in range(1,num+1)])
    perim= np.array([[abs(i-j) if j<=i else 1e7 for i in range(1,num+1)] for j in range(1,num+1)])
    dim = [1e7,0];shift=0
    while np.ptp(dim)>np.sqrt(num):
        a_dim = np.array(np.where((area-num)<=np.min(area-num)+shift)).T
        p_dim = np.argmin([perim[tuple(a)] for a in a_dim])
        dim = a_dim[p_dim]
        shift+=1
    return np.array(dim)+np.ones_like(dim)
def plot_confusion(mat,labels=['R','H','S'],ax=None,stats=True,plot=True):
    if type(mat)==str:
        mat = np.loadtxt(mat,delimiter=',',dtype=int)
    else:
        mat = np.array(mat)
    if all([isinstance(i,list) for i in labels]):
        x_labels,y_labels = labels 
        print(x_labels,y_labels)
    else:
        x_labels,y_labels = labels,labels
    tot = np.sum(mat,axis=1); tot = np.where(tot==0,1,tot)
    percents = (mat.T/tot).T
    percents = np.concatenate((percents,[[1,0]]*len(mat)),axis=1)
    if plot:
        if ax is None: 
            fig = plt.figure()
            ax = plt.gca()
        ax.imshow(percents,cmap='coolwarm')
        ax.set_xticks(ticks=np.arange(len(x_labels)),labels=[x[0] for x in x_labels])
        ax.set_yticks(ticks=np.arange(len(y_labels)),labels=y_labels)
        ax.set_ylabel('Real Classification')
        ax.set_xlabel('Model Classification')
        ax.set_title('Confusion Matrix')
        fig = plt.gcf()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        height = bbox.height/mat.shape[0]*7*plt.rcParams['font.size']
        size = int(np.ceil(height/4))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j,i,str(mat[i][j])[:4],color='white',size=size,ha='center',va='center')
        ax.set_xlim([-0.5,mat.shape[1]-0.5])
        ax.set_ylim([mat.shape[0]-0.5,-0.5])
    stat_dict,these_stats = {},{}
    stat_dict['Accuracy'] = np.trace(mat)/np.sum(mat)
    stat_dict['Sensitivity'] = mat[0][0]/np.sum(mat[0])
    stat_dict['Specificity'] = np.sum(mat[1:].T[1:])/np.sum(mat[1:]) if np.sum(mat[1:])>0 else np.nan 
    all_stats = [stat.lower() for stat in stat_dict.keys()]
    if type(stats)==bool and stats:
        stats = 0
    elif type(stats)==int:
        negs = np.delete(np.arange(len(mat)),stats)
        acc = np.trace(mat)/np.sum(mat)
        sens = mat[stats][stats]/np.sum(mat[stats])
        spec = np.sum(mat[negs].T[negs])/np.sum(mat[negs])
    elif isinstance(stats,(str,list,np.ndarray)):
        if isinstance(stats,str): stats = [stats]
        
# =============================================================================
#         s = np.array(list(stat_dict.keys()))
#         s = s[[]]
#         these_stats[]
# =============================================================================
            
# =============================================================================
#     if stats==0 and plot:
#         ax.set_title('        Accuracy: '+str(np.round(acc*100,int(np.floor(np.sum(mat)/100)))) +
#                   '\nSensitivity ({}): '.format(labels[stats])+str(np.round(sens*100,int(np.floor(np.sum(mat)/100)))) +
#                   '\nSpecificity ({}): '.format(labels[stats])+str(np.round(spec*100,int(np.floor(np.sum(mat)/100)))))
# =============================================================================
    return stat_dict
def plot_contour(data,location = 111,xy_scale=1,xy_units=None,switch_xy=False,new_fig = True,clear = False, cmap = 'coolwarm',vlims = [None,None], cbar=True, cbar_lims=[None,None], cbar_label='', axis=True, alpha = 1,ax = False):
    if new_fig and type(ax)==bool:
        plt.figure()
    if cmap == None:
        cmap = cm.coolwarm
    if type(location)==str:
        num,this_one = np.int_(location.split('_'))
        loc = define_subplot_size(int(num))
        if switch_xy: #switch x and y
            this_one = ((this_one-1)%loc[0])*loc[1] + int((this_one-1)/loc[0]) +1
        location = (loc[0],loc[1],this_one)
    if clear:
        axis=False
        cbar=False
    if type(ax)==bool:
        try:
            ax = plt.subplot(*location)
        except:
            ax = plt.subplot(location)
    else:
        plt.sca(ax)
    if location == 111 and axis==False:
        fig = plt.gcf()
        sizes = np.shape(data)
        fig.set_size_inches(10. * sizes[1] / sizes[0],10.,forward=False)
    plt.imshow(data, cmap = cmap, vmin = vlims[0], vmax = vlims[1],alpha = alpha, interpolation=None, extent=np.array([0,data.shape[1],0,data.shape[0]])*xy_scale);
    
    if not axis:
        ax.axis('off')
    axis_label = 'Pixels' if xy_scale==1 else xy_units if xy_units is not None else 'No units provided'
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    if cbar or len(cbar_label)>0:
        if any([i is not None for i in cbar_lims]):
            cbar_lims = [0 if i is None else i for i in cbar_lims]
            colormap = plt.get_cmap(cmap)
            norm = Normalize(vmin=vlims[0], vmax=vlims[1])
    
            new_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                'truncated_'+cmap, colormap(norm(np.arange(cbar_lims[0], cbar_lims[1]))), N=100)
            norm2 = Normalize(vmin=cbar_lims[0],vmax=cbar_lims[1])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=new_cmap), label=cbar_label)
        else: cbar = plt.colorbar(ax=ax,label=cbar_label)
        return cbar
    plt.pause(0.01)


class ML:
    def __init__(self,train_df,feature_names,truth_feature = 'Class',scale=True):
        self.features = feature_names 
        self.rawdata     = train_df
        self.truth_feature = truth_feature
        self.truths   = train_df[truth_feature]
        self.classes  = np.unique(self.truths).tolist()
        
        if scale:
            scaled_df, scaling = rescale_features(train_df,feature_names)
        else:
            scaled_df = train_df.copy()
            scaling = [[0 for _ in feature_names],[1 for _ in feature_names]]
        self.data = scaled_df
        self.scaling       = scaling
        self.scaling_means = scaling[0]
        self.scaling_stds  = scaling[1]
    
    def svm_loocv(self,kernels=['linear'],use_replicates = True,include_fails=True):
        self.svm_loocv = ML_svm(self.data,self.features,self.truth_feature,kernels = kernels,use_replicates=use_replicates,include_fails=include_fails)
        self.svm = self.svm_loocv.make_model(kernel=kernels[0])
        
    def forest_loocv(self,use_replicates=True):
        self.forest_loocv = ML_randomforest(self.data, self.features,self.truth_feature,use_replicates=use_replicates)
        self.forest = self.forest_loocv.make_model()
        
    def svm_test(self,test_df,kernel='linear'):
        if not hasattr(self, 'svm'):
            self.svm_loocv = ML_svm(self.data,self.features,self.truth_feature,kernels=['linear'],use_replicates=False)
            self.svm_loocv.make_model(kernel=kernel)
            self.svm = self.svm_loocv.model
        scaled_df,_ = rescale_features(test_df,self.features,scaling=self.scaling)
        
        X = scaled_df[self.features].values
        Y = [self.svm_loocv.classes.index(i) for i in scaled_df[self.truth_feature].values]
        self.__svmx__ = X
        self.__svmy__ = Y
        self.svm_result = self.svm.predict(X)
    
    def forest_test(self,test_df):
        if not hasattr(self,'forest'):
            self.forest_loocv = ML_randomforest(self.data, self.features,self.truth_feature,use_replicates=False)
            self.forest_loocv.make_model()
            self.forest = self.forest_loocv.model 
        scaled_df,_ = rescale_features(test_df,self.features,scaling=self.scaling) 
        
        X = scaled_df[self.features].values 
        Y = [self.forest_loocv.classes.index(i) for i in scaled_df[self.truth_feature].values]
        self.__forestx__ = X
        self.__foresty__ = Y 
        self.forest_result = self.forest.predict(X) 
        
    def make_confusion(self,test='svm',all_must_agree = False):
        if not hasattr(self,test):
            getattr(self,test)
        result = getattr(self,test+'_result')
        truth  = getattr(self,f'__{test}y__')
        classes = getattr(self,test+'_loocv').classes
        mat = np.zeros(tuple(np.int_(np.ones(2)*len(classes))))
        for r,t in zip(result,truth):
            mat[t,r]+=1
        if all_must_agree and (np.trace(mat)!=np.sum(mat)):
            mat *= 0
        #plot_confusion(mat,labels=classes)
        return mat, np.trace(mat)/np.sum(mat)
        
    #def forest_test(self):

class ML_svm:
    def __init__(self,test_df,feature_names,truth_feature='Class',kernels=['linear'],use_replicates=True,include_fails=True,quiet=True):
        #kernels= ['linear']#,'poly','rbf','sigmoid']
        u_classes = np.unique(test_df[truth_feature]).tolist()
        X = np.array(list(test_df[feature_names].values))
        Y = np.array([u_classes.index(i) for i in test_df[truth_feature]])
        self.__x__ = X 
        self.__y__ = Y
        if use_replicates:
            rep_col = 'Replicates'
        else:
            rep_col = 'Filename'
        replicates = np.unique(test_df[rep_col].values)
        wrongs,fails,details_wrong,details_fail = [],[],[],[]
        mats={}
        test_df = test_df.copy()
        for n_k,kernel in enumerate(kernels):
            test_df.drop('model_SVM-'+kernel,axis='columns',inplace=True,errors='ignore')
            test_df['model_SVM-'+kernel] = [None for i in range(test_df.shape[0])]
            mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
            wrong,fail,detail,fail_details = [],[],{},{}
            for n,(rep) in enumerate(tqdm(replicates,disable=quiet)):
                to_keep = test_df[rep_col].values!=rep
                train_x = X[to_keep]
                train_y = Y[to_keep]
                test_x = X[~to_keep]
                test_y = Y[~to_keep]
                clf = svm.SVC(kernel = kernel,class_weight='balanced')
                clf.fit(train_x,train_y)
                result=clf.predict(test_x)
                for m,(true,guess) in enumerate(zip(test_y,result)):
                    test_name = test_df['Filename'].values[~to_keep][m]            
                    test_df.loc[test_df['Filename']==test_name,'model_SVM-'+kernel]= u_classes[guess]#==true
                    if use_replicates and len(set(result))>1: #np.sum(result)%len(result)!=0:
                        fail.append(test_name)
                        fail_details = fail_details | {test_name: {'Truth':true,'Guess':guess}}
                        if not include_fails:
                            continue
                    elif use_replicates and len(set(result))==1 and all(result!=test_y):
                        detail = detail | {test_name: {'Truth':true,'Guess':guess}}
                    mat[true][guess]+= 1 if include_fails else 0.5
                    if true!=guess: wrong.append(test_df['Filename'].values[~to_keep][m])
            mat = np.int_(mat) #convert to int
            del clf
            wrongs.append(wrong);fails.append(fail);details_wrong.append(detail);details_fail.append(fail_details)
            mats = mats|{kernel:mat}
        accs = [np.trace(mat)/np.sum(mat)*100 for mat in mats.values()] 
        
        self.fails   = fails
        self.classes = u_classes
        self.wrongs  = wrongs
        self.detail_wrongs = details_wrong
        self.detail_fails  = details_fail
        self.df      = test_df 
        self.mat_dict    = mats
        self.best_kernel = np.array(list(mats.keys()))[np.argmax(accs)]
        self.best_acc    = max(accs)
        self.truth_feature = truth_feature
        
    def confusion_matrices(self):
        u_classes = self.classes
        mats = self.mat_dict
        fig,axs = plt.subplots(*define_subplot_size(len(mats)))
        
        for (kernel,mat),ax in zip(mats.items(),np.array(axs).flatten()):
            plot_confusion(mat,labels=u_classes,ax=ax)
            acc = np.trace(mat)/np.sum(mat) *100
            ax.set_title(kernel+'\n'+str(acc)[:4]+'%')
        
        plt.suptitle('SVM')
        fig.tight_layout()
    
    def make_model(self,kernel='linear',training_data_mask = None):
        if training_data_mask is None:
            training_data_mask = np.arange(len(self.__x__))
        X = self.__x__ [training_data_mask]
        Y = self.__y__ [training_data_mask] 
        classifier = svm.SVC(kernel=kernel,class_weight='balanced')
        classifier.fit(X,Y)
        self.model = classifier 
        self.validation_acc = np.sum(classifier.predict(X)==Y)/len(Y)
        
    def plot_wrongs(self,data):
        dic = self.detail_wrongs[0]
        classes,counts = np.unique([d['Truth'] for d in dic.values()],return_counts=True)
        all_axs = {}
        for c,n in zip(classes,counts):
            this_dic = {i:d for i,d in dic.items() if d['Truth']==c}
            rows,cols = define_subplot_size(n)
            cols = cols + cols%2
            rows = int(np.ceil(n/cols))
            fig,axs = plt.subplots(rows,cols)
            truth = self.classes[c]
            plt.suptitle(f'True {truth} - wrong tests')
            for n,(name,results) in enumerate(this_dic.items()):
                guess = self.classes[results['Guess']]
                ax = axs.ravel()[n]
                color = 'green' if truth==guess else 'red'
                rect = plt.Rectangle( (-.1,-.1), 1.2, 1.2, transform = ax.transAxes,alpha=0.5,color=color,zorder=-1)
                fig.patches.append(rect)
                plot_contour(data[name],ax = ax,vlims=[-100,1e3],cbar=False,axis=False)
                if guess!=truth:
                    ax.text(0.5,1,guess,ha='center',va='bottom',color='white',transform=ax.transAxes)
                short_name = '_'.join(name.split('_')[:2]+[name[-6:]])
                ax.text(1,0.5,short_name,rotation = -90,ha='right',va='center',color='white',transform=ax.transAxes)
                ax.set_gid({'Truth':truth,'Guess':guess,'Filename':name})
            for m in range(n+1,rows*cols):
                ax = axs.ravel()[m]
                ax.axis('off')
            all_axs[fig] = axs 
        return all_axs
    
    def plot_fails(self,data):
        dic = self.detail_fails[0]
        classes,counts = np.unique([d['Truth'] for d in dic.values()],return_counts=True)
        all_axs = {}
        for c,n_tot in zip(classes,counts):
            this_dic = {i:d for i,d in dic.items() if d['Truth']==c}
            rows,cols = define_subplot_size(n_tot)
            cols = cols + cols%2
            rows = int(np.ceil(n_tot/cols))
            fig,axs = plt.subplots(rows,cols)
            truth = self.classes[c]
            plt.suptitle(f'True {truth} - failed tests')
            for n,(name,results) in enumerate(this_dic.items()):
                guess = self.classes[results['Guess']]
                ax = axs.ravel()[n]
                color = 'green' if truth==guess else 'red'
                rect = plt.Rectangle( (-.1,-.1), 1.2, 1.2, transform = ax.transAxes,alpha=0.5,color=color,zorder=-1)
                fig.patches.append(rect)
                plot_contour(data[name],ax = ax,vlims=[-100,1e3],cbar=False,axis=False)
                if guess!=truth:
                    ax.text(0.5,1,guess,ha='center',va='bottom',color='white',transform=ax.transAxes)
                short_name = '_'.join(name.split('_')[:2]+[name[-6:]])
                ax.text(1,0.5,short_name,rotation = -90,ha='right',va='center',color='white',transform=ax.transAxes)
                ax.set_gid({'Truth':truth,'Guess':guess,'Filename':name})
            for m in range(n+1,rows*cols):
                ax = axs.ravel()[m]
                ax.axis('off')
            all_axs[fig] = axs
        return all_axs
    
    def plot_test(self,data):
        df = self.df
        classes = self.classes
        all_axs = {}
        for n,c in enumerate(classes):
            rows,cols = define_subplot_size(2)
            fig,axs = plt.subplots(rows,cols)
            truth = c
            plt.suptitle(f'Testing {truth}')
            files = df.loc[df[self.truth_feature]==c]['Filename'].values[:2]
            for m,filename in enumerate(files):
                ax = axs.ravel()[m]
                plot_contour(data[filename],ax = ax,vlims=[-100,1e3],cbar=False,axis=False)
                ax.set_gid({'Truth':classes[n],'Guess':classes[(n-m)%len(classes)],'Filename':filename})
            for m in range(n+1,rows*cols):
                ax = axs.ravel()[m]
                ax.axis('off')
            all_axs[fig] = axs 
        return all_axs

from sklearn.neural_network import MLPClassifier
class ML_MLP:
    def __init__(self,test_df,feature_names,truth_feature='Class',solvers=['adam','lbfgs', 'sgd'],hidden_layer_sizes=(6,9),use_replicates=True,quiet=True):
        #kernels= ['linear']#,'poly','rbf','sigmoid']
        u_classes = np.unique(test_df[truth_feature]).tolist()
        X = np.array(list(test_df[feature_names].values))
        Y = np.array([u_classes.index(i) for i in test_df[truth_feature]])
        if isinstance(solvers,str): solvers = [solvers]
        self.__x__ = X 
        self.__y__ = Y
        self.layers = hidden_layer_sizes
        self.solver = solvers[0]
        if use_replicates:
            rep_col = 'Replicates'
        else:
            rep_col = 'Full names'
        replicates = np.unique(test_df[rep_col].values)
        wrongs,fails = [],[]
        mats={}
        for n_k,solver in enumerate(solvers):
            test_df.drop('model_MLP-'+solver,axis='columns',inplace=True,errors='ignore')
            test_df['model_MLP-'+solver] = [None for i in range(test_df.shape[0])]
            mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
            wrong,fail = [],[]
            for n,(rep) in enumerate(tqdm(replicates,disable=quiet)):
                to_keep = test_df[rep_col].values!=rep
                train_x = X[to_keep]
                train_y = Y[to_keep]
                test_x = X[~to_keep]
                test_y = Y[~to_keep]
                mlp = MLPClassifier(solver=solver,alpha=1e-5,hidden_layer_sizes=self.layers)
                mlp.fit(train_x,train_y)
                result=mlp.predict(test_x)
                for m,(true,guess) in enumerate(zip(test_y,result)):
                    fname = test_df['Replicates'].values[~to_keep][m]            
                    test_df.loc[test_df['Replicates']==fname,'model_MLP-'+solver]= u_classes[guess]#==true
                    if use_replicates and np.sum(result)%len(result)!=0:
                        fail.append(fname)
                        continue
                    mat[true][guess]+=1
                    if true!=guess: wrong.append(test_df['Full names'].values[~to_keep][m])
            del mlp
            wrongs.append(wrong);fails.append(fail)
            mats = mats|{solver:mat}
        accs = [np.trace(mat)/np.sum(mat)*100 for mat in mats.values()] 
        
        self.classes = u_classes
        self.wrongs = wrongs
        self.df = test_df 
        self.mat_dict = mats
        self.best_kernel = np.array(list(mats.keys()))[np.argmax(accs)]
        self.best_acc = max(accs)
        
    def confusion_matrices(self):
        u_classes = self.classes
        mats = self.mat_dict
        fig,axs = plt.subplots(*define_subplot_size(len(mats)))
        
        for (kernel,mat),ax in zip(mats.items(),np.array(axs).flatten()):
            plot_confusion(mat,labels=u_classes,ax=ax)
            acc = np.trace(mat)/np.sum(mat) *100
            ax.set_title(kernel+'\n'+str(acc)[:4]+'%')
        
        plt.suptitle('MLP')
        fig.tight_layout()
    
    def make_model(self,solver=None,training_data_mask = None):
        if solver is None: solver = self.solver
        if training_data_mask is None:
            training_data_mask = np.arange(len(self.__x__))
        X = self.__x__ [training_data_mask]
        Y = self.__y__ [training_data_mask] 
        classifier = MLPClassifier(solver=solver,alpha=1e-5,hidden_layer_sizes=self.layers)
        classifier.fit(X,Y)
        self.model = classifier 
        self.validation_acc = np.sum(classifier.predict(X)==Y)/len(Y)
# =============================================================================
# 
# test_df.drop('model_MLP',axis='columns',inplace=True,errors='ignore')
# test_df['model_MLP'] = [None for i in range(test_df.shape[0])]
# X = np.array(list(test_df[feature_names].values))
# Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
# replicates = np.unique(test_df['Replicates'].values)
# mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
# importances = []
# for n,(rep) in enumerate(tqdm(replicates)):
#     to_keep = test_df['Replicates'].values!=rep
#     train_x = X[to_keep]
#     train_y = Y[to_keep]
#     test_x = X[~to_keep]
#     test_y = Y[~to_keep]
#     mlp = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(6,9))
#     forest.fit(train_x,train_y)
#     result = forest.predict(test_x)
#     importances.append(forest.feature_importances_)
#     for m,(true,guess) in enumerate(zip(test_y,result)):
#         fname = test_df['Filename'].values[~to_keep][m]            
#         test_df.loc[test_df['Filename']==fname,'model_MLP']= u_classes[guess]#==true
#         mat[true][guess]+=1
#     del forest
# plot_confusion(mat,u_classes)
# title = plt.gca().get_title();plt.title(title+'\nMLP')
# =============================================================================

class ML_randomforest:
    def __init__(self,test_df,feature_names,truth_feature='Class',use_replicates=True):
        #fig,axs = plt.subplots()
        u_classes = np.unique(test_df[truth_feature]).tolist()
        X = np.array(list(test_df[feature_names].values))
        Y = np.array([u_classes.index(i) for i in test_df[truth_feature]])
        self.__x__ = X 
        self.__y__ = Y
        if use_replicates:
            rep_col = 'Replicates'
        else:
            rep_col = 'Full names'
        replicates = np.unique(test_df[rep_col].values)
        wrongs,fails = [],[]
        mats={}
        
        test_df.drop('model_RandomForest',axis='columns',inplace=True,errors='ignore')
        test_df['model_RandomForest'] = [None for i in range(test_df.shape[0])]
        mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
        wrong,fail,importances = [],[],[]
        for n,(rep) in enumerate(tqdm(replicates)):
            to_keep = test_df[rep_col].values!=rep
            train_x = X[to_keep]
            train_y = Y[to_keep]
            test_x = X[~to_keep]
            test_y = Y[~to_keep]
            forest = RF()
            forest.fit(train_x,train_y)
            result = forest.predict(test_x)
            importances.append(forest.feature_importances_)
            for m,(true,guess) in enumerate(zip(test_y,result)):
                fname = test_df['Replicates'].values[~to_keep][m]            
                test_df.loc[test_df['Replicates']==fname,'model_RandomForest']= u_classes[guess]#==true
# =============================================================================
#                 if use_replicates and np.sum(result)%len(result)!=0:
#                     fail.append(fname)
#                     continue
# =============================================================================
                mat[true][guess]+=1
                if true!=guess: wrong.append(fname)
            del forest
        print('Random Forest'+str(np.round(np.trace(mat)/np.sum(mat)*100,1))+'%')
        wrongs.append(wrong);fails.append(fail)
        mats = mats|{'RandomForest':mat}
        
        self.classes = u_classes
        self.features = np.array(feature_names)
        self.wrongs = wrongs 
        self.importances = importances 
        self.df = test_df 
        self.mat_dict = mats
        self.best_acc = max([np.trace(mat)/np.sum(mat)*100 for mat in mats.values()])
        
    def confusion_matrices(self):
        u_classes = self.classes
        mats = self.mat_dict
        mat = mats['RandomForest']
        plot_confusion(mat,labels=u_classes)
        plt.title('Random Forest')
        plt.gcf().tight_layout()
        
    def plot_importances(self):
        imps = self.importances 
        means = np.mean(imps,axis=0)
        args = np.argsort(means)
        stds  = np.std( imps,axis=0)
        plt.figure();plt.errorbar(self.features[args],means[args],yerr=stds[args],fmt='o')
        plt.xticks(ha='right',rotation=45)
        plt.gcf().tight_layout()
        plt.ylabel('Random Forest LOOCV\nimportance averages')
        return means
    
    def make_model(self,training_data_mask = None):
        if training_data_mask is None:
            training_data_mask = np.arange(len(self.__x__))
        X = self.__x__ [training_data_mask]
        Y = self.__y__ [training_data_mask] 
        classifier = RF()
        classifier.fit(X,Y)
        self.model = classifier 
        self.validation_acc = np.sum(classifier.predict(X)==Y)/len(Y)