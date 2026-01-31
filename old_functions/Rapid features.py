# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:10:20 2023

@author: ajkru
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn 
import os
import pandas as pd
import seaborn as sns
from scipy import ndimage
from myfunctions import lmat,define_subplot_size,plot_all,remove_outliers,power_spectrum,radial_avg,convert_data,sym_reg,interpol,make_confusion,plot_confusion,ring_width,get_features,plot_contour
from myfunctions_full import cartesian
from tqdm import tqdm
from datetime import datetime
start = datetime.now()
from Find_best_algorithm import find_best_combination,most_important_algs

to_rep = lambda x: '_'.join(x.split('_')[:4:2])
class_from_rep = lambda rep,df: pd.unique(df.loc[df['Replicates']==rep]['Class'])[0]
all_genera = ['Enterobacter','Escherichia','Klebsiella','Acinetobacter','Pseudomonas']
drugs,families,associations,associations_drugs=[],[],{},{}

if bool(1): #enterobacteriaceae
    drugs.append(['TZP','TET','CIP','SXT','FOF','TOB'])
    families.append('Enterobacteriaceae')
    associations['Enterobacteriaceae']=drugs[-1]
    associations_drugs = associations_drugs|{d:families[-1] for d in drugs[-1]}
if bool(0): #moraxellaceae
    drugs.append(['FDC','SAM'])
    families.append('Moraxellaceae')
    associations['Moraxellaceae']=drugs[-1]
    associations_drugs = associations_drugs|{d:families[-1] for d in drugs[-1]}
if bool(1): #Pseudomonas
    drugs.append(['CZA','LEV','TOB'])
    families.append('Pseudomonas')
    associations['Pseudomonas']=drugs[-1]
    associations_drugs = associations_drugs|{d:families[-1] for d in drugs[-1]}
families_to_test = [[family] for family in families] + [families] if len(families)>1 else [families]
#drugs = [['TOB']]
#families = [associations_drugs['CZA']]
test_individually   = True
test_together       = True
only_together       = False
classifying_feature = 'Class' #'Family' or 'Genera' or 'Class'
obj                 = 5
plot                = False
pairplots           = False
fast                = True  #fast is to just read excel of features
remove_class        = []#'Escherichia','Enterobacter','Pseudomonas']
remove_class_class  = 'Species'
test = True
main_root = 'Weiss Lab Data/'

if classifying_feature in ['Species','Genera','Family']: drugs,only_together = [['end','mnd','pnd']],True
all_drugs = pd.unique([d for drug in drugs for d in drug]).tolist()
roots = [main_root+'Rapid R '+fam+'/'+f+'/' for fam in families for f in os.listdir(main_root+'Rapid R '+fam) if '_Rapid R' in f and '-4hr' in f and any([d in f for d in all_drugs])]

all_summaries = {}
for vert_resolution in [1]:#[250,500,1000,5e3,10e3]:
    #big_data = len(roots)>4
    big_data=False
    #if big_data:
    #    plot=False
    fftval = 18
    plt.close('all');errs=[]
    dfs={};full_df = pd.DataFrame();scaled_df = pd.DataFrame()
    feature_names = None #['Home Height','Home CoV','Ring w_sat','Ring Hurst','ringH/homeH','FFT-{}um'.format(fftval)] 
    all_feature_names = get_features(None,obj=obj,fftval=fftval)
    feature_names = all_feature_names if feature_names is None else feature_names
    bads = {}
    all_data={}
    all_counts={'R':0,'S':0,'N':0}
    for root in roots:
        date = root.split('/')[2][:6]
        data = {};get_data = True
        drug = root.split('-')[0][-3:]
        family = root.split('/')[1].split(' ')[2]
        print('Gathering data for '+family +'+'+ drug + ' on '+date)
        try:
            if fast:# and os.path.exists(root+'features.xlsx') and os.path.exists(root+'cleaned_data.mat'):
                df = pd.read_excel(root+'features.xlsx',header=0)
                data = lmat(root+'cleaned_data.mat',{})
            else:
                int('s')
        except:
            try:
                if fast:# and os.path.exists(root+'cleaned_data.mat'):
                    data = lmat(root+'cleaned_data.mat',{})
                    get_data = False
                else:
                    int('s')
            except:
                info = [pd.read_excel(root+file,dtype=str) for file in os.listdir(root) if file.endswith('.xlsx') and 'feat' not in file][0]
                species_dict={}
                for file in tqdm(os.listdir(root)):
                    if '.datx' not in file: continue
                    if 'bad' in file.lower(): continue
                    filename = root+file
                    if not os.path.exists(root+'cleaned_data.mat') or get_data:
                        raw,latres = convert_data(filename,resolution=True)
                        
                        raw = np.round(raw/vert_resolution)*vert_resolution
                        
                        c = sym_reg(raw,1,normal=True)
                        I,J = cartesian(c)
                        if obj == 5:
                            mask = J>c.shape[1]-100
                        elif obj == 5.5:
                            mask = J>c.shape[1]-200
                        dd = np.where(mask,c,np.nan)
                        c = sym_reg(dd,1,normal=True,full_image=c)
                        if np.sum(c[250:650,:400]>0)/400**2<0.1:
                            os.rename(filename,root+'bad-'+file)
                            plot_contour(c)
                            #continue
                        data[file[:-6]+date+'_'+file[-6]+'_'+drug] = c
                        
                    file_info = file[:-5].split('_')
                    species_dict[file_info[0]] = info['Isolate'].loc[info['image_label']==file_info[0]].values[0].split(' ')[0]
                    label = '_'.join([file_info[0],date]+file_info[1:])
                
    
    
            print('    Calculating features ({} its)'.format(len(data.keys()))) 
            n_r = np.sum([i[-5]=='R' for i in data.keys()])
            n_s = len(data)-n_r
            counts = {'R':0,'S':0,'N':0}
            features = {i:[] for i in all_feature_names}
            phenotypes,sn,files,rep,genera,species = [],[],[],[],[],[]
            for n,(i,arr) in enumerate(tqdm(data.items())):
                if 'bad' in i: continue
                if np.sum(np.isnan(arr))>0:
                    arr = interpol(arr)
                    data[i] = arr
                features_dict = get_features(arr,obj=obj)
                for name,feat in features_dict.items():
                    features[name].append(feat)
                phenotypes.append(i[-5])
                sn.append(i.split('_')[0])
                rep.append('_'.join(i.split('_')[::2]+[i[-5]]))
                files.append(i)
                genera.append(species_dict[i.split('_')[0]])
                counts[i[-5]]+=1
                all_counts[i[-5]]+=1
            df = pd.DataFrame(features)
            df['Class'] = phenotypes
            df['Strain'] = sn
            df['Filename'] = files
            df['Replicates'] = rep
            df['Genera'] = genera
            df['Family'] = [family for i in genera]
            df['Drug'] = [drug for i in range(len(genera))]
            if vert_resolution==1:
                df.to_excel(root+'features.xlsx',index=False)
        full_df = pd.concat([full_df,df],axis=0)
        if vert_resolution==1:
            scipy.io.savemat(root+'cleaned_data.mat',data)
        all_data = all_data|data if not big_data else {}
    
    
    scaled_df = full_df.copy()
    data_gathered = datetime.now()
    count = {clas: np.sum(scaled_df[classifying_feature]==clas)/2 for clas in np.unique(scaled_df[classifying_feature].values)}
    classes = np.unique(scaled_df[classifying_feature]).tolist()
    for i in feature_names:
        dat = np.array(scaled_df[i])
        dat = (dat-np.mean(dat))/np.std(dat)
        scaled_df[i] = dat
    
    if not test:
        print('Purposefully committing error, no testing requested.')
        int('s')
    summary={};all_mats = {};test_df=[]; all_results={}; all_importances={}
    if only_together: families_to_test = [families_to_test[-1]]
    for family in families_to_test:
        if len(family)>1:
            family_identifier = 'All Combined'
        else:
            family_identifier = family[0]
        these_drugs = [np.unique(associations[fam]).tolist() for fam in family]
        all_these_drugs = np.unique([dr for fam in family for dr in associations[fam]]).tolist()
        drugs_to_test = [[d] for d in these_drugs] if test_individually and not test_together else these_drugs+[all_these_drugs]*(len(these_drugs)>1) if test_together and not test_individually else [[d] for d in all_these_drugs]+these_drugs+[all_these_drugs]*(len(these_drugs)>1)
        if only_together: drugs_to_test=[drugs_to_test[-1]]
        if classifying_feature!='Class': drugs_to_test = drugs.copy()
        for drug in drugs_to_test:
            if len(drug)>1:
                drug_identifier = 'Combined-'+','.join(drug)
            else:
                drug_identifier = 'Drug: '+drug[0]
            print('\n\nTesting on '+drug_identifier+' for '+family_identifier)
            if len(drugs_to_test)>1: plt.close('all')
            del test_df
            drug_mask = np.array([i in drug for i in scaled_df['Drug']])
            fam_mask  = np.array([i in family for i in scaled_df['Family']])
            test_df = scaled_df.loc[drug_mask & fam_mask].copy()
            if len(np.unique(test_df['Family']))==1 and len(family)>1: continue
            if len(remove_class)>0:
                test_df = test_df.loc[[i not in remove_class for i in test_df[remove_class_class]]].copy()
            classes = np.unique(test_df[classifying_feature]).tolist()
            model = sklearn.cluster.KMeans(len(classes))
            model.fit(np.array(test_df[feature_names]))
            test_df['Model_KMeans']=[i+'->'+str(j) for i,j in zip(test_df[classifying_feature],model.labels_)]
            u = pd.unique(test_df['Model_KMeans'])
            
            confusion_matrix,cipher = make_confusion(test_df,get_cipher=True,truth=classifying_feature)
            plot_confusion(confusion_matrix,labels=classes,stats=True)
            #not ideal because now x and o are same color..but good classification generalization
            colors = {c:i for i,c in zip([[0,0,1],[1,0,0],[1,0.5,0],[0.5,1,0],[0.5,0.5,0]],np.unique(test_df[classifying_feature]))}
            markers = [['o','X'][int(i.split('->')[0]!=cipher[i.split('->')[1]])] for i in u]
            palette = {i:tuple(colors[i.split('->')[0]]+[0.1 if m=='o' else 1]) for i,m in zip(u,markers)}
            
            R_df = test_df.loc[test_df['Class']=='R']
            if pairplots:
                gtgold = '#CC9933'; gtblue = '#003057';
                sns.pairplot(full_df,hue=classifying_feature,vars = feature_names,palette={'R':gtblue,'S':gtgold} if classifying_feature=='Class' else None)
                sns.pairplot(test_df,hue='Model_KMeans',palette=palette,markers=markers,vars=feature_names)
                sns.pairplot(R_df,vars = feature_names,hue='Species')
            
            u_classes,c_classes = list(np.unique(test_df[classifying_feature],return_counts=True))
            all_counts = {u:c for u,c in zip(u_classes,c_classes)}
            u_classes = u_classes.tolist()
            
            results = [cipher[i.split('->')[1]] for i in test_df['Model_KMeans']]#[cipher[i[-1]]==i.split('->')[0] for i in test_df['Model_KMeans']]
            
            test_df.drop('model_KMeans-unsupervised',axis='columns',inplace=True,errors='ignore')
            test_df['model_KMeans-unsupervised'] = results
            
            
            
            ###Change to generalized plotting
            if plot and len(drug)==1:
                for des in ['R','S']:
                    dic = {i:d for i,d in all_data.items() if des in i.split('_')[-2] and drug[0] in i}
                    figsize = np.flip(define_subplot_size(len(dic)))*2+np.array([4,2])
                    plt.figure(des+'-'+str(drug),figsize=figsize,dpi=200)
                    plot_all(dic,vlims=[0,15e3],new_fig=False,figsize=figsize,dpi=200)
                    bug = associations_drugs[drug[0]]
                    plt.savefig('Weiss Lab Data/Rapid R {}/Images/{} data-{}.png'.format(bug,des,drug[0]),dpi=200)
            
            
            
            
            
            
            mats = {'KMeans':confusion_matrix}
            ##SVM
            if True:
                from sklearn import svm
                kernels= ['linear','poly','rbf','sigmoid']
                fig,axs = plt.subplots(*define_subplot_size(len(kernels)))
                X = np.array(list(test_df[feature_names].values))
                Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
                replicates = np.unique(test_df['Replicates'].values)
                wrongs,fails = [],[]
                for n_k,kernel in enumerate(kernels):
                    test_df.drop('model_SVM-'+kernel,axis='columns',inplace=True,errors='ignore')
                    test_df['model_SVM-'+kernel] = [None for i in range(test_df.shape[0])]
                    mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
                    wrong,fail = [],[]
                    for n,(rep) in enumerate(tqdm(replicates)):
                        to_keep = test_df['Replicates'].values!=rep
                        train_x = X[to_keep]
                        train_y = Y[to_keep]
                        test_x = X[~to_keep]
                        test_y = Y[~to_keep]
                        clf = svm.SVC(kernel = kernel,class_weight='balanced')
                        clf.fit(train_x,train_y)
                        result=clf.predict(test_x)
                        for m,(true,guess) in enumerate(zip(test_y,result)):
                            fname = test_df['Filename'].values[~to_keep][m]            
                            test_df.loc[test_df['Filename']==fname,'model_SVM-'+kernel]= u_classes[guess]#==true
                            if np.sum(result)%2!=0:
                                fail.append(rep)
                            else:
                                mat[true][guess]+=1
                            if true!=guess: wrong.append(fname)
                    del clf
                    plot_confusion(mat,labels=u_classes,ax=np.array([axs]).flatten()[n_k])
                    title = np.array([axs]).flatten()[n_k].get_title();np.array([axs]).flatten()[n_k].set_title(title+'\nSVM-'+kernel)
                    print(kernel,np.round(np.trace(mat)/np.sum(mat)*100,1))
                    wrongs.append(wrong);fails.append(fail)
                    mats = mats|{kernel:mat}
                plt.suptitle('SVM')
                wrongs_dic = {}
                for wrong in wrongs:
                    for f in wrong:
                        if f not in wrongs_dic: wrongs_dic[f]=0
                        wrongs_dic[f]+=1
                fails_dic = {}
                for fail in fails:
                    for f in fail:
                        if f not in fails_dic: fails_dic[f]=0
                        fails_dic[f]+=1
                
            ##Random Forest
            if True:
                from sklearn.ensemble import RandomForestClassifier
                test_df.drop('model_Random Forest',axis='columns',inplace=True,errors='ignore')
                test_df['model_Random Forest'] = [None for i in range(test_df.shape[0])]
                X = np.array(list(test_df[feature_names].values))
                Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
                replicates = np.unique(test_df['Replicates'].values)
                mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
                importances = []
                for n,(rep) in enumerate(tqdm(replicates)):
                    to_keep = test_df['Replicates'].values!=rep
                    train_x = X[to_keep]
                    train_y = Y[to_keep]
                    test_x = X[~to_keep]
                    test_y = Y[~to_keep]
                    forest = RandomForestClassifier()
                    forest.fit(train_x,train_y)
                    result = forest.predict(test_x)
                    importances.append(forest.feature_importances_)
                    for m,(true,guess) in enumerate(zip(test_y,result)):
                        fname = test_df['Filename'].values[~to_keep][m]            
                        test_df.loc[test_df['Filename']==fname,'model_Random Forest']= u_classes[guess] #==true
                        mat[true][guess]+=1
                    del forest
                plot_confusion(mat,u_classes)
                title = plt.gca().get_title();plt.title(title+'\nRandom Forest')
            
            ##Multi-layer perceptron
            if False:
                from sklearn.neural_network import MLPClassifier
                test_df.drop('model_MLP',axis='columns',inplace=True,errors='ignore')
                test_df['model_MLP'] = [None for i in range(test_df.shape[0])]
                X = np.array(list(test_df[feature_names].values))
                Y = np.array([u_classes.index(i) for i in test_df[classifying_feature]])
                replicates = np.unique(test_df['Replicates'].values)
                mat = np.zeros((len(u_classes),len(u_classes)),dtype=int).tolist()
                importances = []
                for n,(rep) in enumerate(tqdm(replicates)):
                    to_keep = test_df['Replicates'].values!=rep
                    train_x = X[to_keep]
                    train_y = Y[to_keep]
                    test_x = X[~to_keep]
                    test_y = Y[~to_keep]
                    mlp = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(6,9))
                    forest.fit(train_x,train_y)
                    result = forest.predict(test_x)
                    importances.append(forest.feature_importances_)
                    for m,(true,guess) in enumerate(zip(test_y,result)):
                        fname = test_df['Filename'].values[~to_keep][m]            
                        test_df.loc[test_df['Filename']==fname,'model_MLP']= u_classes[guess]#==true
                        mat[true][guess]+=1
                    del forest
                plot_confusion(mat,u_classes)
                title = plt.gca().get_title();plt.title(title+'\nMLP')
            
            
            model_names = [col for col in test_df if 'model' in col]# and 'unsupervised' not in col]
            
            strain_info_columns = [i for i in test_df.columns if i not in feature_names and i not in model_names]
            
            results,wrong_reps,names,first_sorting = find_best_combination(test_df,model_names,classifying_feature)
            all_results[(str(family),str(drug))] = results
            best = {n:d for n,d in zip(names,results[0])}
            mat_best = best['Confusion Matrix']
            plot_confusion(np.int_(mat_best),labels = u_classes)
            mat = mat_best.copy()
            all_mats[str(drug)]=mat
            failed = {'R':0,'S':0,'N':0}
            if len(best['Failed replicates'])>0:
                for fail in np.array(best['Failed replicates'],dtype=object).T[0]:
                    failed[fail[-1]]+=1
            acc,sens,spec = plot_confusion(mat,['R','S'],plot=False).values()
            columns = ['Successful Tests','Number R','Number S','Success Rate',
                       'Accuracy','R Sensitivity','R Specificity','% VME','% ME',
                       '# Failed Tests','# Failed R','# Failed S','Best Model']
            summary_list = [int(np.sum(mat)),int(np.sum(mat[0])),int(np.sum(mat[1])),best['Success Rate'],
                       acc,sens,spec,best['Very Major Error'],best['Major Error'],
                       best['# fails'],failed['R'],failed['S'],best['Test names']
                       ]
            species_reps = {}
            for sp in all_genera:
                mask = test_df['Genera'].values==sp    
                species_reps[sp] = np.unique(test_df['Replicates'].loc[mask])
                try:
                    fails = len([rep for rep in species_reps[sp] if rep in np.array(best['Failed replicates'],dtype=object).T[0]])
                except:
                    fails = 0
                right,wrong = best['Correct replicates'],best['Incorrect replicates']
                n_species = len(species_reps[sp])
                n_r = len([r for r in species_reps[sp] if r[-1]=='R'])
                n_s = len([r for r in species_reps[sp] if r[-1]=='S'])
                right = [rep for rep in species_reps[sp] if rep in right]
                R_right = [r for r in right if r[-1]=='R']
                S_right = [r for r in right if r[-1]=='S']
                wrong = [rep for rep in species_reps[sp] if rep in wrong]
                R_wrong = [r for r in wrong if r[-1]=='R']
                S_wrong = [r for r in wrong if r[-1]=='S']
                if n_species==fails:
                    s_acc,s_vme,s_me = np.nan,np.nan,np.nan
                else:
                    s_acc = len(right)/(n_species-fails)
                    s_vme = len(R_wrong)/(n_species-fails)*100
                    s_me  = len(S_wrong)/(n_species-fails)*100
                #try:
                #    fails = len([rep for rep in species_reps[sp] if rep in np.array(results[best][-1],dtype=object).T[0]])
                #except:
                #    fails = 0
                columns = columns + ['{}-{}'.format(sp,i) for i in ['#','#R','#S','Acc','VME','ME','#Fails']]
                summary_list = summary_list + [n_species,n_r,n_s,s_acc,s_vme,s_me,fails]
                
            all_importances[(family_identifier,drug_identifier)] = np.array(importances)
            summary[(family_identifier,drug_identifier)] = summary_list
    finish = datetime.now()
    print('\n\nData Gathering:',data_gathered-start)
    print('LOOCV:',finish-data_gathered)
    print('Total time elapsed:',finish-start)
    summary_df = pd.DataFrame.from_dict(summary,orient='index',columns=columns)
    
    today = datetime.today().strftime('%Y%m%d')[2:]
    summary_df.sort_index(inplace=True)
    summary_df = summary_df.T
    summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns,names=['Family','Drug'])
    #sum_writer = pd.ExcelWriter('Weiss Lab Data/Rapid R results/{}_results.xlsx'.format(today),engine='xlsxwriter')
    #summary_df.to_excel(sum_writer,sheet_name=f'Summary-{vert_resolution}um')
    #workbook = sum_writer.book 
    #worksheet = sum_writer.sheets['Summary']
    #red_format = workbook.add_format({'bg_color':'#FFC7CE', 'font_color': '#9C0006'})
    #alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    #alphabet = alphabet + [i+j for i in alphabet for j in alphabet]
    #end_letter,end_number = alphabet[summary_df.shape[1]],columns.index('Accuracy')+4
    #worksheet.conditional_format('B{}:{}{}'.format(end_number,end_letter,end_number),{'type':'cell','criteria':'<','value':0.95,'format':red_format})
    #sum_writer.save()
    #sum_writer.close()
    all_summaries[vert_resolution] = summary_df.copy()
    
    numbers={'Total':{},'Resistant':{},'Susceptible':{}}
    for drug in all_drugs:
        df = scaled_df.loc[scaled_df['Drug']==drug]
        for phen in numbers:
            numbers[phen][drug]=[]
        for ge in all_genera:
            numbers['Total'][drug].append(len(np.unique(df['Replicates'].loc[df['Genera']==ge])))
            numbers['Resistant'][drug].append(len(np.unique(df['Replicates'].loc[(df['Genera']==ge)&(df['Class']=='R')])))
            numbers['Susceptible'][drug].append(len(np.unique(df['Replicates'].loc[(df['Genera']==ge)&(df['Class']=='S')])))
    for phen in numbers:
        numbers[phen]['Total']=np.sum(list(numbers[phen].values()),axis=0)
        for drug in all_drugs:
            numbers[phen][drug].append(np.sum(numbers[phen][drug]))
    writer = pd.ExcelWriter('Weiss Lab Data/Rapid R results/{}_isolates.xlsx'.format(today), engine='openpyxl')
    for df_name,df in numbers.items():
        pd.DataFrame.from_dict(df,orient='index',columns=all_genera+['Total']).to_excel(writer,sheet_name=df_name)
    writer.close()
    best = {name:res for name,res in zip(names,results[0])}
    other = {name:res for name,res in zip(names,sorted(results,key=lambda x:-x[names.index('Accuracy' if 'Accuracy'!=first_sorting else 'Success Rate')])[0])}
    colors = plt.get_cmap('tab10')
    plt.figure() 
    cc1,_,_ = plt.hist(results.T[names.index('Success Rate')],bins = np.arange(0,1,.05),label='Success Rate',alpha=0.8,color=colors(0),density=True)
    cc2,_,_ = plt.hist(results.T[names.index('Accuracy')],bins = np.arange(0,1,.05),label='Accuracy',alpha=0.8,color=colors(1),density=True)
    mm = np.max(list(cc1)+list(cc2))
    plt.plot(np.ones(2)*best['Success Rate'],[0,mm],c=colors(0))
    plt.plot(np.ones(2)*best['Accuracy'],[0,mm],c=colors(1))
    
    plt.plot(np.ones(2)*other['Success Rate'],[0,mm],'k--')
    plt.plot(np.ones(2)*other['Accuracy'],[0,mm],'k--')
    
    plt.legend()

# =============================================================================
# drug = 'FOF'
# date = '240224'
# root = 'Weiss Lab Data/{}_Rapid R {}-4hr/'.format(date,drug)
# df = pd.read_excel(root + 'strain_info.xlsx',dtype=str)
# raw_root = root+'Not filled/'
# for file in os.listdir(raw_root):
#     if drug not in file or 'bad' in file: continue
#     i = file.split('_')[0]
#     rep = 'b' in i.lower()
#     sn = i[:-1] if rep else i
#     num= '2' if rep else '1'
#     num = num if len(file.split('_'))==2 else '3'
#     phen = df['phenotype'].loc[df['image_label']==sn].values[0][0]
#     new = 'bad_'*(phen not in ['R','S']) +'_'.join([sn,num,phen])
#     os.rename(raw_root+file,raw_root+new+'.datx')
# =============================================================================

# =============================================================================
# drug = 'ND'
# family = 'Pseudomonas'
# date = '240305'
# root = 'Weiss Lab Data/Rapid R {}/{}_Rapid R {}-4hr/'.format(family,date,drug)
# df = pd.read_excel(root + 'strain_info.xlsx',dtype=str)
# raw_root = root+'Not filled/'
# for file in os.listdir(raw_root):
#     #if drug not in file or 'bad' in file: continue
#     i = file.split('_')[0]
#     rep = 'b' in i.lower()
#     sn = i[:-1] if rep else i
#     num= '2' if rep else '1'
#     num = num #if len(file.split('_'))==2 else '3'
#     phen = df['phenotype'].loc[df['image_label']==sn].values[0][0]
#     new = 'bad_'*(phen not in ['R','S','N']) +'_'.join([sn,num,phen])
#     os.rename(raw_root+file,raw_root+new+'.datx')
#     os.rename(root+file,root+new+'.datx')
# =============================================================================

# =============================================================================
# for file in os.listdir(raw_root):
#     if not file.startswith('26'): continue
#     raw = convert_data(raw_root+file)
#     plot_3d(raw);plt.title(file)
# =============================================================================


