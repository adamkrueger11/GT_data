## This file will contain all of the necessary cleaning functions and plotting

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import os
import time
import scipy
import scipy.stats
import scipy.io
import scipy.interpolate
import math
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib
import gzip
import shutil
from random import sample
import lzma 
from skimage import morphology as mor
from scipy import ndimage
import skimage
from matplotlib.patches import Ellipse,Rectangle,Path,PathPatch
import matplotlib.transforms as transforms
import h5py
import pandas as pd
from tqdm import tqdm
import pickle
import itertools

plt.rcParams['font.size']=20
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['text.color'] = 'black'
plt.rcParams['lines.color']='black'

def plotfiles(top,include=[],exclude=[]):
    '''Plot a whole collection of datx files from the "top" directory.
        include: list of file identifying strings
        exclude: list of file identifying strings'''
    for root,_,files in os.walk(top):
        for file in files:
            if any([i in file for i in exclude]) or all([i not in file for i in include]): continue
            if file.endswith('.datx'):
                print(root+'/'+file)
                data=clean(root+'/'+file,just_full=True)
                plot_3d(data)
                plt.title(file)



def datx2py(file_name):
    """Loads a .datx into Python, credit goes to gkaplan.
    https://gist.github.com/g-s-k/ccffb1e84df065a690e554f4b40cfd3a"""
    def _group2dict(obj):
        return {k: _decode_h5(v) for k, v in zip(obj.keys(), obj.values())}
    def _struct2dict(obj):
        names = obj.dtype.names
        return [dict(zip(names, _decode_h5(record))) for record in obj]
    def _decode_h5(obj):
        if isinstance(obj, h5py.Group):
            d = _group2dict(obj)
            if len(obj.attrs):
                d['attrs'] = _decode_h5(obj.attrs)
            return d
        elif isinstance(obj, h5py.AttributeManager):
            return _group2dict(obj)
        elif isinstance(obj, h5py.Dataset):
            d = {'attrs': _decode_h5(obj.attrs)}
            try:
                d['vals'] = obj[()]
            except (OSError, TypeError):
                pass
            return d
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.number) and obj.shape == (1,):
                return obj[0]
            elif obj.dtype == 'object':
                return _decode_h5([_decode_h5(o) for o in obj])
            elif np.issubdtype(obj.dtype, np.void):
                return _decode_h5(_struct2dict(obj))
            else:
                return obj
        elif isinstance(obj, np.void):
            return _decode_h5([_decode_h5(o) for o in obj])
        elif isinstance(obj, bytes):
            return obj.decode()
        elif isinstance(obj, list) or isinstance(obj, tuple):
            if len(obj) == 1:
                return obj[0]
            else:
                return obj
        else:
            return obj
    with h5py.File(file_name, 'r') as f:
        h5data = _decode_h5(f)
    return h5data

def convert_data(file_name,resolution = False,get=['Heights'],remove=True):
    '''Convert datx file to numpy array
        resolution: if True, return tuple of data and lateral resolution
        get: list of attributes to receive. 'Heights' and 'Intensity' are the options. can be both. if both, will provide dictionary. '''
    values = {}
    if '.' not in file_name: file_name+='.datx'
    full_dict = datx2py(file_name)
    if 'Heights' in get:
        try:
            lat_res = full_dict['Measurement']['Attributes']['attrs']['Data Context.Lateral Resolution:Value']
            data = np.array(full_dict['Measurement']['Surface']['vals'])
        except:
            data = np.array(full_dict['Processed Data: ']['PM-Micro']['AP_DS:Regions']['SequenceMatrix1']['Surface']['vals'])
        if remove:         
            data = np.where(data>10**100,np.nan,data)
            median = np.nanmedian(data)
            data /= median
                    
            new = np.array([[i if abs(i)<1000 else np.nan for i in j] for j in data])*median
                
            cutoff = np.nanmean(new)-5*np.nanstd(new)
            new = np.where(new > cutoff,new,np.nan)
        else:
            new = np.where(np.log10(np.abs(data))>300,np.nan,data)
        values['Heights'] = new
    if 'Intensity' in get:
        values['Intensity'] = np.array(full_dict['Measurement']['Intensity']['vals'])
    if resolution and len(get)==1:
        return new,lat_res
    if len(values)==1:
        return values[get[0]]
    else:
        if resolution: values['latres']=lat_res
        return values

def get_intensity(file_name):
    full_dict = datx2py(file_name)
    data = np.array(full_dict['Measurement']['Intensity']['vals'])    
    return data

def get_clean_data(mat_file,just_names = False):
    names = ['Homeland','No ring','Ring','Agar','Full','Ring coordinates','Ring width']
    if just_names:
        return names
    data_dict = scipy.io.loadmat(mat_file); data={}
    for name in names:
        data[name] = data_dict[name]
        if data[name].size==1: data[name]=data[name][0][0]
    return data


def get_arrangement(date):
    site_arrangements = {'211201': ['baseline1','01R1','.1R1','00R1','01R2','.1R2','00R2','baseline2'],
                         '211214': ['baseline1','01R1','.1R1','00R1','baseline2','01R2','.1R2','00R2','baseline3','01R3','.1R3','00R3'],
                         '220113': ['baseline1','00R1','00H1','.1R1','.1H1','01R1','01H1','baseline2','00R2','00H2','.1R2','.1H2','01R2','01H2','baseline3','00R3','00H3','.1R3','.1H3','01R3','01H3'],
                         '220120': ['baseline1','00R1','00H1','.1R1','.1H1','01R1','01H1','baseline2','00R2','00H2','.1R2','.1H2','01R2','01H2','baseline3','00R3','00H3','.1R3','.1H3','01R3','01H3'],
                         '220203': ['01R1','.01R1','00R1','01R2','.01R2','00R2','01R3','.01R3','00R3'],
                         '220204': ['01R1','.001R1','00R1','01R2','.001R2','00R2','01R3','.001R3','00R3','.001R4'],
                         '220205': ['01R1','.001R1','00R1','01R2','.001R2','00R2','01R3','.001R3','00R3','.001R4'],
                         '220210': ['01R1','.01R1','00R1','01R2','.01R2','00R2','01R3','.01R3','00R3','.01R4'],
                         '220211': ['01R1','.01R1','00R1','01R2','.01R2','00R2','01R3','.01R3','00R3','.01R4'],
                         '2202151': ['01R1','.01R1','00R1','01R2','.01R2','00R2','01R3','.01R3','00R3','.01R4'],
                         '2202152': ['01R1','.01R1','00R1','01R2','.01R2','00R2','01R3','.01R3','00R3','.01R4'],
                         '2202153': ['.01R1','00R1','.01R2','00R2','.01R3','00R3','.01R4','00R4','.01R5','00R5'],
                         '220308': ['01R1','.01R1','01R1','.01R1'],
                         '220315': ['01R1','00R1','.1R1','.01R1','01R2','00R2','.1R2','.01R2','01R3','00R3','.1R3','.01R3'],
                         'proces': [],'220419':[],
                         '220817': ['R1','S1','H1','R2','S2','H2','R3','S3','H3']
                         }
    return site_arrangements[date]
    
    
    
def timeit(func,data,reps = 1):
    t1 = time.time()
    for i in range(reps):
        func(data)
    return (time.time() - t1)/reps

def get_xy(img):
    if img.shape[1]==1:
        X = np.arange(len(img)).reshape(len(img),1)
        Y = np.array([])
        return X, Y
    else:
        X = np.arange(len(img)).reshape(len(img),1)
        Y = np.arange(len(img[0])).reshape(1,len(img[0]))
        return X,Y

def convert_obj(full_data):
    mf,nf = np.int_(np.floor(np.array(full_data.shape)/5))
    new=np.ones((mf,nf))*np.nan
    i=0
    while i<mf-1:
        j=0
        while j<nf-1:
            new[i,j]=np.nanmean(full_data[5*i:5*(i+1),5*j:5*(j+1)])
            j+=1
        i+=1
    return new


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
    #plt.pause(0.01)

def plot_all(dic,close=False,vlims=[None,None],new_fig=True,individual=False,figsize=None,dpi=None,fullname=True,titles=True,axis=False):
    if all(isinstance(value,dict) for value in dic.values()):
        full_dic = dic.copy()
        sup_title = True
    else:
        full_dic = {'All data':dic}
        sup_title = False
    if close: plt.close('all')
    axes = {}
    for name,dic in full_dic.items():
        if len(dic)==0: continue
        if new_fig:
            fig = plt.figure(num=str(name),figsize=figsize,dpi=dpi)
        else:
            fig = plt.gcf()
        if individual:
            for i,d in dic.items():
                plot_contour(d,vlims=vlims);
                if titles: plt.title(i)
                plt.gca().set_gid(i)
        else:
            for n,(i,d) in enumerate(dic.items(),1):
                plot_contour(d,vlims=vlims,new_fig=False,location='{}_{}'.format(len(dic),n),axis=axis,cbar=vlims[1] is None)
                if titles: plt.title('_'.join(i.split('_')[::2]) if not fullname else i)
                if axis:
                    plt.xlabel(None);plt.ylabel(None)
                    plt.yticks([]);plt.xticks([])
                plt.gca().set_gid(i)
            if vlims[0] is not None:
                fig = plt.gcf()
                mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vlims[0], vmax=vlims[1], clip=False),cmap=plasma)
                fig.colorbar(mappable, ax=fig.axes)
        if sup_title:
            plt.gcf().suptitle(name)
        axes[fig] = np.array(plt.gcf().axes)
    return axes

def plot(image,ax_on_off = 'off'):
    fig, ax = plt.subplots()
    plt.imshow(image,cmap = 'gray')
    ax.axis(ax_on_off)
    plt.pause(0.01)

def write_excel(dic, path):
    if type(dic) != dict:
        dic = {'Sheet1':dic}
    writer = pd.ExcelWriter(path,date_format=None,mode='w')
    for df_name, df in dic.items():
        df.to_excel(writer, sheet_name=df_name)
    writer.close()

def plot_profiles(orig, back = [], clean = [], fit = [],limits = [], save = False,name = 'Profile',image_type = '.jpg'):
    plot_clean = True; plot_fit = True; plot_no_ring = True; lim = True
    if len(list(clean))==0: plot_clean = False
    if len(list(fit))==0:   plot_fit = False
    if len(list(back))==0:  plot_no_ring = False
    if len(limits) == 0:    lim  = False
    
    
    num_plots = np.sum([1,plot_clean])  
    sizes = [200,500,800]
    
    


    for line in range(len(sizes)):
        
        fig = plt.figure(line)
        ax3 = fig.add_subplot(111, frameon=False)
        ax3.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax3.set_xlabel("Pixels")
        ax3.set_ylabel("Height")
        
        ax1 = fig.add_subplot(np.sum(num_plots),1,1)
        ax1.plot(orig[sizes[line]], label = 'Original'); 
        if lim:
            ax1.set_xlim(limits[sizes[line]])
            
        if plot_no_ring:
            ax1.plot(back[sizes[line]], label = 'No Coffee Ring')
        if plot_clean:
            ax2 = fig.add_subplot(2,1,2,sharex = ax1)
            ax2.plot(clean[sizes[line]],label = 'Cleaned'); 
            ax2.legend()
            if lim: 
                ax2.set_xlim(limits[sizes[line]])
        if plot_fit:
            ax1.plot(fit[sizes[line]], label = 'Background', alpha = 0.8)
        ax1.legend()
        
        fig.suptitle('Profile at y-pixel = {}\n{}'.format(sizes[line],name))
        if save:
            new_file_name = name + '_{}'.format(sizes[line])+image_type
            if os.path.exists(new_file_name):
                os.remove(new_file_name)
            plt.savefig(new_file_name, format = image_type[1:])           
def get_elbow(data, x=None, xy = True, elbow_point = True):
    y = (data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data)); 
    x = np.arange(0,1+10**(-5),1/len(data[:-1]));  
    flip=False
    if y[0]>y[-1]:
        x = np.flip(x)
        flip=True
    d = x-y
    if np.nanmax(d)!=0:# and y[1]<x[1] * 0.9:
        elbow=np.argmax(d)
# =============================================================================
#         dd= np.where(abs(d-np.nanmax(d))/np.nanmax(d)<0.01,d,np.nan)
#         elbow = d.tolist().index(np.nanmax(d))
#         elbow = dd.tolist().index(np.nanmin(dd))
# =============================================================================
    else:
        elbow = 0
    elbow = np.sum(y>y[elbow]*0.99)#(len(y)-1)/len(y))
    if not flip:
        elbow = len(y)-elbow
        
    if xy and not elbow_point:
        return x,y,d
    elif elbow_point and not xy:
        return d,elbow
    elif xy and elbow_point:
        return x,y,d,elbow
    else:
        return d


def get_data(folder,data={}):
    for file in os.listdir(folder):
        data[file[:-5]],latres = convert_data(folder+'/'+file,resolution=True,remove=False)
    return data,latres



def find_ring(data,direction = False):
    '''Identifies a broad region that could contain the coffee ring. Also identifies if the ring is horizontal or vertical (other functions require it being vertical)
            data: 2d array that potentially contains a coffee ring
            direction: bool - Return the direction of the coffee ring
        Returns:
            if direction: string with orientation of coffee ring
            ring: a list of indices along which there is likely to be part of the ring'''
    
    repeat = True; repititions=0
    while repeat:
        repititions += 1
        tallest = ~np.isnan(np.where(data>np.nanmean(data)+np.nanstd(data),data,np.nan))
        #tallest = ~np.isnan(np.where(tall>np.nanmean(tall)-np.nanstd(tall),tall,np.nan))
        global counts,all_counts,maxl
        all_counts={}
        maxl={}
        ind1 = np.array([np.arange(len(data[0])) for row in data])
        ind2 = np.array([np.arange(len(data.T[0])) for row in data.T])
        for ind,i,di in zip([ind1,ind2],[tallest,tallest.T],['no transpose','transpose required']):
            all_counts[di] = np.bincount(ind[i]);
            maxl[di] = np.max(all_counts[di]/len(i.T[0]))
        
        counts,d = [[all_counts[d].tolist(),d] for d in maxl if maxl[d]==max(maxl.values())][0]
        tran_bool = d
        pos = []
        for d in [-1,1]:
            i = counts.index(max(counts))
            a = counts[i]
            while a >0 and i in range(len(counts)-1):
                i += d
                a = counts[i]
            pos.append(i)
                
        ring = np.arange(max([pos[0]-10,0]),min([pos[1]+10+1,data.shape[1]])).tolist()
        
        if repititions >= 3:
            repeat = False
        if len(ring)>7*min(data.shape)/8:
            data = sym_reg(data,2)
        else:
            repeat = False

    if direction:
        return tran_bool
    return ring



def remove_outliers(data,N=10,fill=np.nan):
    global small_counts,x,h
    
    counts,bins = np.histogram(data[~np.isnan(data)],bins=N)
    for n,c in enumerate(counts):
        if c<10**(-3) * data.size and c !=0:
            data = np.where(data<bins[n+1],np.nan,data)
        elif c != 0:
            break
    for n2,c in enumerate(np.flip(counts[n+1:])):
        if c<10**(-3) * data.size and c != 0:
            data = np.where(data>bins[-(n2+2)],np.nan,data)
        elif c!=0:
            break
    try:
        np.isnan(fill)
    except:
        if fill=='mean':
            data = np.where(np.isnan(data),np.nanmean(data),data)
        if fill == 'median':
            data = np.where(np.isnan(data),np.nanmedian(data),data)
    return data

def remove_ring_old(data, to_end = False, width = False, return_ring = False):
    global tall, buffer, index,ring_width,no_ring, tallest,ring_loc, ring_pos,end,ring
    orig_data = data.copy()
    r = find_ring(data)
    if len(r)==0:
        im = data.copy()
    else: 
        im = np.concatenate((data[:,:r[0]],np.nan*np.ones((len(data),len(r))),data[:,r[-1]+1:]),axis = 1)
    data = sym_reg(im,2,full_image=data)
    data = remove_outliers(data)
    ring_loc = find_ring(data)
    if len(ring_loc)==0:
        print('This data has no significant coffee ring')
        if width:
            return data,[0,0],[[],[]]
        elif return_ring:
            return data,data
        else:
            return data
        

    ring_loc_min = int((3*min(ring_loc)-max(ring_loc))/2);   ring_loc_max = int((3*max(ring_loc)-min(ring_loc))/2)
    ring_min_init = ring_loc_min
    
    row_num = 0; no_ring = orig_data.copy(); ring = orig_data.copy(); ring_pos = []
    buffer = int(np.ceil(data[0].size/100))
    index = np.arange(data[0].size,dtype = int)
    for row in data:
        tall = np.where(row>np.nanmean(row)-2*np.nanstd(row)/np.sqrt(len(row)),index,np.nan)
        real_ring = np.where(tall>ring_loc_min,tall,np.nan)
        real_ring = np.where(real_ring<ring_loc_max,real_ring,np.nan)
        try:
            ring_min = int(np.nanmin(real_ring)); ring_max = int(np.nanmax(real_ring))
            
            ring_loc_min = max(ring_min -2*buffer,ring_min_init)
            
            
            while np.sum(np.isnan(row[ring_min-1-buffer:ring_min-1]))>0 and ring_min>0+1+buffer:
                ring_min-=1
            while np.sum(np.isnan(row[ring_max+1:ring_max+1+buffer]))>0 and ring_max<len(row)-1-buffer:
                ring_max+=1
            ring_width = int(ring_max-ring_min+2*buffer)
        except:
            continue
        if to_end:
            ring_range = np.arange(ring_min-buffer,len(row),dtype = int)
            no_ring[row_num][ring_range] = np.nan*np.ones(int(len(row)-ring_min+buffer))
        else:
            if ring_max+buffer>=data[0].size:
                end = data[0].size-1
                ring_width = int(data[0].size-1-(ring_min-buffer))
            else: end = ring_max+buffer
            ring_range = np.arange(ring_min-buffer,end,dtype = int)
            no_ring[row_num][ring_range] = np.nan*np.ones(ring_width)
            ring[row_num] = np.concatenate((np.nan*np.ones(min(ring_range)-1),ring[row_num][ring_range],np.nan*np.ones(len(row)-max(ring_range))))
        ring_pos.append([[row_num,ring_min],[row_num,ring_max]])
        row_num+=1

    if width:
        N = len(ring_pos); avg_ring = []
        for i in sample(range(0,N),int(np.ceil(N/4))):
            point = ring_pos[i][0]
            A = np.outer(np.ones(N),point)
            B = np.array(ring_pos)[:,1] 
            avg_ring.append(min(np.linalg.norm(A-B,axis = 1)))
        ring_mean = int(np.round(np.mean(avg_ring)))
        ring_med = int(np.round(np.median(avg_ring)))

        return no_ring, [ring_mean, ring_med], ring_pos
    elif return_ring:
        return no_ring, ring
    else:
        return no_ring





def remove_ring_old2(data,to_end = False,return_pos = False, width = False,return_ring = False):
    '''Find the Ring if it is significant and remove it (also provides various combinations of locations)
            data: 2d array or list that may have a coffee ring
            to_end: bool - remove everything outside the coffee ring (return just the homeland)
            return_pos: bool - return the boundaries of the ring
            width: bool - return the average width of the ring
            return_ring: bool - remove everything except the ring (return just the coffee ring)
        Returns:
            collection of lists/arrays corresponding to input booleans'''
    global ring_exists
    ring_exists = True
    data = remove_outliers(data)
    orig_data = data.copy()
    r = find_ring(data);
    if len(r)>data.shape[1]*0.5:
        r = find_ring(sym_reg(data,2))
        
            ## create 'im' that removes the approximate coffee ring
    if len(r)==0 or len(r)>=0.75 * data.shape[1]:
        im = data.copy()
    else: 
        im = np.concatenate((data[:,:r[0]],np.nan*np.ones((len(data),len(r))),data[:,r[-1]+1:]),axis = 1)
    
            ## fit a background to the data without the ring and subtract from the entire dataset
    data = sym_reg(im,2,full_image = data)
    
            ## The coffee ring is tall, so only look at the taller sections of the data
    cut = np.nanmean(data)# + np.nanstd(data)
    im = np.where(data>cut,1,0)
    closed = im + np.where(np.isnan(data),1,0)  ##create a binary mask of the tall data
    
            ## to find just the ring, we take the mask and erode to avoid having other high spots connected
            ## if the ring is too small, this could erode the ring, but then it is not a significant sized ring
    neighborhood = np.array([[0,0,1,0,0] if j!=3 else [1,1,1,1,1] for j in range(5)])
    for i in range(4):
        closed = mor.binary_erosion(closed,neighborhood)
    for i in range(4):
        closed = mor.binary_dilation(closed,neighborhood)
    closed = mor.closing(closed,np.ones((5,5)))
    buffer = int(np.ceil(data.shape[0]/100))

            ## find the remaining connected components (one should be the ring)
    labeled,num = ndimage.label(closed)
    
    means = [];ind=[]
    for i in range(num+1):
        mask = np.where(labeled==i,1,np.nan)
        mean = np.nanmean(data*mask)
        if np.isnan(mean): continue
        means.append(np.nanmean(data*mask))     ## finds the average height of connected component i
        ind.append(i)
    r = ind[np.argsort(means)[-1]]   # tallest connected component is the ring
    ring = np.where(labeled == r,1,0)
    if np.sum(ring)<np.sqrt(labeled.size)*10:
        ring_pos = [1]
    else:
        ring_mask = mor.closing(ring,np.ones((buffer,buffer)))
        ring_pos = skimage.measure.find_contours(ring_mask)
        
    if len(ring_pos)>2:
        long = np.argsort([len(i) for i in ring_pos])[-1]
        new = []
        for i in range(len(ring_pos)):
            if i==long: continue
            new.extend(ring_pos[i])
        ring_pos = [ring_pos[long],new]
    
    if len(ring_pos)==1:# and len(ring_pos[0])<1.25*len(data):
        print('No significant ring'); ring_exists=False
        ring_pos = np.array([np.array([np.arange(data.shape[0]),np.ones(data.shape[1])*data.shape[1]/2]).T,
                             np.array([np.arange(data.shape[0]),np.ones(data.shape[1])*data.shape[1]/2]).T])
        no_ring = orig_data.copy()
        ring = np.ones(data.shape)*np.nan
        if width:
            return no_ring, [np.nan,np.nan], ring_pos
        elif return_pos:
            return no_ring, ring_pos
        elif return_ring:
            return no_ring, ring
        else:
            return no_ring
        
    
    mean_ring = np.mean([np.mean(ring_pos[i],0)[1] for i in range(len(ring_pos))])
    d = np.sign(mean_ring - data.shape[1]/2)
    
    
    ring = np.where(ring_mask>0,orig_data,np.nan)
    no_ring = np.where(ring_mask==0,orig_data,np.nan)
    
    if to_end:
        labeled,num = ndimage.label(np.where(np.isnan(no_ring),0,1))
        region_info = skimage.measure.regionprops(labeled); labels = []
        for n,obj in enumerate(region_info):
            if d*obj.centroid[1]>d*mean_ring:
                labels.append(obj.label)
                
        for label in labels:
            labeled = np.where(labeled==label,np.nan,labeled)
        no_ring_mask = np.where(np.isnan(labeled),np.nan,1)
        no_ring = no_ring *no_ring_mask
    
    if width:
        N1 = len(ring_pos[0]); N2 = len(ring_pos[1]); avg_ring = []
        for i in sample(range(0,N1),int(np.ceil(N1/4))):
            point = ring_pos[0][i]
            A = np.outer(np.ones(N2),point)
            B = np.array(ring_pos[1])
            avg_ring.append(min(np.linalg.norm(A-B,axis = 1)))
        ring_mean = int(np.round(np.mean(avg_ring)))
        ring_med = int(np.round(np.median(avg_ring)))

        return no_ring, [ring_mean, ring_med], ring_pos
    elif return_pos:
        return no_ring, ring_pos
    elif return_ring:
        return no_ring, ring
    else:
        return no_ring
    

def remove_ring(data,ring=False,parts=False):
    '''Find the Ring if it is significant and remove it (also provides various combinations of locations)
            data: 2d array or list that may have a coffee ring
            to_end: bool - remove everything outside the coffee ring (return just the homeland)
            return_pos: bool - return the boundaries of the ring
            width: bool - return the average width of the ring
            return_ring: bool - remove everything except the ring (return just the coffee ring)
        Returns:
            collection of lists/arrays corresponding to input booleans'''
    just_ring = ring
    orig_data = data.copy()
    r = find_ring(data)
    if len(r)==0:
        im = data.copy()
    else:
        if len(r)>data.shape[1] * 3/4:
            data = sym_reg(data,2)
            r = find_ring(data)
        im = np.concatenate((data[:,:r[0]],np.nan*np.ones((len(data),len(r))),data[:,r[-1]+1:]),axis = 1)
    #data = sym_reg(im,2,full_image=data)
    data = remove_outliers(data)
    ring_loc = find_ring(data)
    if len(ring_loc)==0:
        print('This data has no significant coffee ring')
        return data
    
    no_ring = orig_data.copy(); ring = orig_data.copy(); agar = orig_data.copy(); ring_pos = []
    index = np.arange(data[0].size,dtype = int)
    last_left = min(ring_loc); buffer = int(np.ceil(data.shape[1]/100))
    for n,row in enumerate(data):
        if np.sum(~np.isnan(row))==0: continue
        if np.sum(~np.isnan(row[ring_loc]))<0.05*np.ptp(ring_loc):
            no_ring[n][ring_loc[0]:ring_loc[-1]] = np.zeros(np.ptp(ring_loc))
            agar[n][:ring_loc[-1]] = np.nan*np.ones(ring_loc[-1])
            continue
        num_nans = np.sum(np.isnan(row))
        max_ind = np.argsort(row)[-num_nans-1]
        
        short = np.where(row<=np.nanmedian(row),index,np.nan)
        if np.sum(~np.isnan(short[short<max_ind]))==0:
            left_ind = max_ind
        else:
            left_ind = int(np.nanmax(short[short<max_ind]))
# =============================================================================
#         if len(ring_pos)>0 and abs(left_ind-last_left)>buffer:
#             short2 = np.where(row[:max_ind]<=np.nanmedian(row[left_ind:max_ind]),index[:max_ind],np.nan)
#             left_ind = int(np.nanmax(short2))
# =============================================================================
            
        if np.sum(~np.isnan(short[max_ind:]))==0:
            right_ind = 2*max_ind - left_ind
        else:
            right_ind = int(np.nanmin(short[short>max_ind]))
            right_ind = int(np.nanmin(np.where(row[right_ind:]<=np.nanmedian(row[right_ind:]),np.arange(right_ind,len(row)),np.nan)))
            right_ind += buffer
        right_ind = min(data.shape[1]-1,right_ind)
        ring_range = right_ind-left_ind
        no_ring[n][left_ind:right_ind] = np.zeros(ring_range)
        agar[n][:right_ind] = np.nan*np.ones(right_ind)
        ring_pos.append([[n,left_ind],[n,right_ind]])
        last_left=left_ind
    ring_mask = np.where(no_ring==0,1,0)
    ring_mask = mor.dilation(ring_mask)
    ring_pos = skimage.measure.find_contours(ring_mask)
    agar_mask = np.where(np.isnan(agar),0,1)
    
    no_ring = np.where(np.logical_not(ring_mask),orig_data,np.nan)
    ring = np.where(ring_mask,orig_data,np.nan)
    home = np.where(ring_mask | agar_mask,np.nan,orig_data)
    
    if just_ring:
        return ring
    elif parts:
        N1 = len(ring_pos[0]); N2 = len(ring_pos[1]); avg_ring = []
        for i in sample(range(0,N1),int(np.ceil(N1/4))):
            point = ring_pos[0][i]
            A = np.outer(np.ones(N2),point)
            B = np.array(ring_pos[1])
            avg_ring.append(min(np.linalg.norm(A-B,axis = 1)))
        ring_mean = int(np.round(np.mean(avg_ring)))
        ring_med  = int(np.round(np.median(avg_ring)))
        ring_width = [ring_mean, ring_med]
        return home, ring, agar, ring_pos,ring_width
    else:
        return no_ring

    
    
    
    
def watershed_alg(image,r):
    distance = ndimage.distance_transform_edt(image)
    dist = np.where(distance>np.sqrt(distance.size),np.sqrt(distance.size),distance)
    distance = np.where(distance<0,0,dist)
    distance = ndimage.gaussian_filter(distance,sigma=1)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((r,r)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = skimage.segmentation.watershed(-distance, markers, mask=image)
    return labels
    
    
def interpol(img):
    x,y=get_lib_x(img)
    z=img.flatten()
    X,Y,Z = np.array([x,y,z]).T[~np.isnan(z)].T
    x,y = np.array([x,y]).T[np.isnan(z)].T 
    
    new = scipy.interpolate.griddata((X,Y),Z,(x,y),method='cubic')
    
    z[np.isnan(z)] = new
    
    final = z.reshape(img.shape)
    return final
    
    
    
    
    
    
    
def choose(n,r):    #binomial
    return int(math.factorial(n)/((math.factorial(r)*math.factorial(n-r))))
def get_lib_x(img):
    x = []
    XX,YY = get_xy(img)
    if YY.size == 0:
        x.append(XX.flatten())
        return np.array(x,dtype = float)
    
    X = np.outer(XX,np.ones(len(YY.T))).flatten()
    Y = np.outer(YY.T,np.ones(len(XX))).T.flatten()
    
    x.append(X)
    x.append(Y)
    return np.array(x,dtype = float)

def create_library(x,degree):
    
    num_of_vars = len(x)
    num_of_terms = choose(num_of_vars+degree,degree)
    theta=[[]for i in range(num_of_terms)]                      #this will be the library
    data_labels=['{}'.format(i) for i in range(num_of_vars)]    #this is the name of all fo the variables (0,1,2,3...)
    
    
    theta[0]=np.ones(len(x[0]))     #constant term
    
    terms=['' for i in range(num_of_terms)];                    #Will show the combination for each term
    
    i=1; d=0; prev_combos=0
    while i<num_of_terms:
        for j in range(prev_combos,choose(d+num_of_vars-1,d)+prev_combos):  #goes through combinations of same length (xx,xy,zz..length 2)
            if j==0: starting_var=0
            else: starting_var=data_labels.index(terms[j][-1])  #index makes sure we start with the 'next' variable in the combo
            
            for k in range(starting_var, num_of_vars):  #adds each new variable on to preexisting combination
                terms[i]=terms[j] + data_labels[k]
                theta[i]= x[k]*theta[j]
                i+=1
            #print()
        prev_combos+=choose(d+num_of_vars-1,d)
        d+=1        
    theta=np.array(theta)
    return theta,terms
def residual(library,coef,data):
    dif = np.matmul(library.T,coef)-data
    res=np.linalg.norm(dif[~np.isnan(dif)])/num_not_nans
    return res    
def sym_reg(image, degree, X = [], full_image = None, include_z=False,res_terms=False,normal=False,all_fits = False, surf = False, just_fit = False, just_res=False,coefs = False, terms_in_fit = False, secondary_trials = False):
    orig_shape = image.shape
    global num_of_vars,num_of_terms, num_not_nans
    num_of_vars=len(image.shape)    #number of variables to have data for
    if num_of_vars == 1: image = image.reshape(len(image),1)
    if full_image is None: 
        full_image = image.copy()
    else:
        orig_shape = full_image.shape
    num_of_terms = choose(num_of_vars + degree, degree)
    num_not_nans = np.sqrt(np.sum(~np.isnan(image)))
    if X == []:
        x = get_lib_x(image)
    else:
        x = np.array(X)
    data = image.flatten()
    not_nans = ~np.isnan(data);
    #x = x.T[not_nans].T; data = data[not_nans]
    if include_z:
        x = np.concatenate((x,data.reshape(1,len(data))))
    if num_of_vars == 1:
        x = x.reshape(1,np.array(x).size)
    global normalizations, best
    normalizations=np.ones(num_of_vars)
    for var in range(num_of_vars):
        normalizations[var]*=abs((np.nanmean(np.abs(x[var]))))
                
        x[var]/=normalizations[var]
        #data.T[var]/=normalizations[var]
    theta_full, terms = create_library(x,degree)
    theta = theta_full.T[not_nans].T; data = data[not_nans]
    denorms = np.array([np.prod(normalizations[np.int_(list(i))]) for i in terms])
            #finds initial regression coefficients
    
    xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(theta,theta.T)),theta),data)
    xi_best=xi.copy();terms_best=terms.copy()
    
    xi_final={x: [0]  for x in terms};  #initialize dictionary that will show final coefficient matrix with term labels
    xi_real=np.zeros(num_of_terms)
    times=[];
    #for var in range(num_of_vars):
    global deleted
    skip,deleted=[],[]
    magnitude = np.floor(np.log10(theta.shape[1])-4)
    if magnitude<0: magnitude = 0
    
    step_size = int(10**magnitude)
    theta_var=theta[:,::step_size]
    data = data[::step_size]
    terms_var=terms.copy()
    res_initial=residual(theta_var,xi,data)
    res = res_initial;res_best = res_initial
    global residues,all_terms, all_xi
    residues=[res_best]; all_terms = [terms.copy()]; all_xi = [xi] #[res_initial]
    if not normal:
        for term in range(num_of_terms-1):                  #Delete all except one term in regression
            trial_residues=[];
            for i in range(num_of_terms-term): #for each term to delete, run through all possible ones to minimize residual
                trial=np.delete(theta_var,i,0)          #trial library
                if np.linalg.det(np.matmul(trial,trial.T))==0:
                    trial_residues.append(res+10*res)
                    continue
                time1=time.time()
                xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(trial,trial.T)),trial),data)    #trial coefficient matrix
                time2=time.time()
                times.append(time2-time1)
                res=residual(trial,xi,data)
                if term==0 and i in skip:
                    #print(skip)
                    res*=1000
                trial_residues.append(res)
            #plt.semilogy(trial_residues)
            #plt.title('Fails: {}'.format(fails))
            #plt.close()
            term_to_del=trial_residues.index(min(trial_residues))   #find the best term to delete from the possibilities above
            deleted.append(terms_var[term_to_del])
            if term==0 and secondary_trials==True: skip.append(term_to_del)
                #delete term that mimized residual
            terms_var.pop(term_to_del)
            
            theta_var=np.delete(theta_var,term_to_del,0)
            #find coefficents and residual again
            xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(theta_var,theta_var.T)),theta_var),data)
            residues.append(min(trial_residues))
                #save the coefficient matrix with the best residual
            
            all_terms.append(terms_var.copy())
            
            all_xi.append(xi)
# =============================================================================
#         if residues[-1]<1.1 * res_best:         #WHAT IS TRIGGER POINT??
#             #print(residues[-1])
#             xi_best=xi.copy()
#             terms_best=terms_var.copy()
#             res_best=residues[-1]
#             best = term + 1
# =============================================================================

        #else: break
        global d
        d,elbow = get_elbow(residues,xy = False, elbow_point = True)
        terms_best = all_terms[elbow]
        xi_best = all_xi[elbow]
    else:
        terms_best = all_terms[0]
        xi_best = all_xi[0]
            #undo normalization of best-fit term coefficients
    for sparse_term in terms_best:
        fix_norm=1
        
        for term in sparse_term:
            fix_norm*=normalizations[int(term)]
        xi_final[sparse_term] = [xi_best[terms_best.index(sparse_term)]/fix_norm]
        xi_real[terms.index(sparse_term)] = xi_best[terms_best.index(sparse_term)]/fix_norm
    
    
    if full_image.shape == image.shape:
        theta = theta_full * denorms.reshape(-1,1)
    else:
        x = get_lib_x(full_image); 
        theta,_ = create_library(x,degree)
    #X,Y = get_xy(full_image)
    surface = theta.T@xi_real; surface = surface.reshape(orig_shape)
    if all_fits:
        fits=[]
        for one_xi,one_terms in zip(all_xi,all_terms):
            for sparse_term in one_terms:
                fix_norm=1
                
                for term in sparse_term:
                    fix_norm*=normalizations[int(term)]
                xi_final[sparse_term] = [one_xi[one_terms.index(sparse_term)]/fix_norm]
                xi_real[terms.index(sparse_term)] = one_xi[one_terms.index(sparse_term)]/fix_norm
            
            x = get_lib_x(full_image); theta,_ = create_library(x,degree)
            #X,Y = get_xy(full_image)
            surface = theta.T@xi_real; surface = surface.reshape(full_image.shape)
            fits.append(surface)
        return (fits,all_terms)
    full_image = full_image.reshape(orig_shape)
    if just_res:
        return residues
    if res_terms:
        return res_best, terms_best
    if just_fit:
        return surface
    elif coefs:
        return xi_real, terms_best
    elif terms_in_fit:
        return full_image - surface, terms_best
    elif surf:
        return full_image-surface, terms_best, surface
    else: return full_image - surface


def sym_reg_sparse(X,Y,Z, degree, full_image, surf = False, just_fit = False, terms_in_fit = False, secondary_trials = False):
    global num_of_vars,num_of_terms, num_not_nans
    num_of_vars = 2    #number of variables to have data for
    if len(X.shape)!=2:
        X = X.reshape(-1,1)
    if len(Y.shape)!=2:
        Y = Y.reshape(-1,1)
    if len(Z.shape)!=2:
        Z = Z.reshape(-1,1)
    num_of_terms = choose(num_of_vars + degree, degree)
    num_not_nans = np.sqrt(np.sum(~np.isnan(full_image)))
    
    x = np.concatenate((X,Y,np.ones(X.shape)),axis=1).T
    data = Z
    
    global normalizations, best
    normalizations=np.ones(num_of_vars)
    for var in range(num_of_vars):
        normalizations[var]*=abs((np.mean(np.abs(x[var]))))
                
        x[var]/=normalizations[var]
        #data.T[var]/=normalizations[var]
        
    theta, terms = create_library(x,degree)
            #finds initial regression coefficients
    xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(theta,theta.T)),theta),data)
    xi_best=xi.copy();terms_best=terms.copy()
    
    xi_final={x: [0]  for x in terms};  #initialize dictionary that will show final coefficient matrix with term labels
    xi_real=np.zeros(num_of_terms)
    times=[];
    #for var in range(num_of_vars):
    skip=[]
    magnitude = np.floor(np.log10(theta.shape[1])-4)
    if magnitude<0: magnitude = 0
    
    step_size = int(10**magnitude)
    step_size = 1
    theta_var=theta[:,::step_size]
    data = data[::step_size]
    terms_var=terms.copy()
    res_initial=residual(theta_var,xi,data)
    res = res_initial;res_best = res_initial
    global residues,all_terms, all_xi
    residues=[res_best]; all_terms = [terms.copy()]; all_xi = [xi] #[res_initial]
    for term in range(num_of_terms-1):                  #Delete all except one term in regression
        trial_residues=[];
        for i in range(num_of_terms-term): #for each term to delete, run through all possible ones to minimize residual
            trial=np.delete(theta_var,i,0)          #trial library
            if np.linalg.det(np.matmul(trial,trial.T))==0:
                trial_residues.append(res+10*res)
                continue
            time1=time.time()
            xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(trial,trial.T)),trial),data)    #trial coefficient matrix
            time2=time.time()
            times.append(time2-time1)
            res=residual(trial,xi,data)
            if term==0 and i in skip:
                #print(skip)
                res*=1000
            trial_residues.append(res)
        #plt.semilogy(trial_residues)
        #plt.title('Fails: {}'.format(fails))
        #plt.close()
        term_to_del=trial_residues.index(min(trial_residues))   #find the best term to delete from the possibilities above
       
        if term==0 and secondary_trials==True: skip.append(term_to_del)
            #delete term that mimized residual
        terms_var.pop(term_to_del)
        
        theta_var=np.delete(theta_var,term_to_del,0)
        #find coefficents and residual again
        xi=np.matmul(np.matmul(np.linalg.inv(np.matmul(theta_var,theta_var.T)),theta_var),data)
        residues.append(min(trial_residues))
            #save the coefficient matrix with the best residual
        
        all_terms.append(terms_var.copy())
        
        all_xi.append(xi)
# =============================================================================
#         if residues[-1]<1.1 * res_best:         #WHAT IS TRIGGER POINT??
#             #print(residues[-1])
#             xi_best=xi.copy()
#             terms_best=terms_var.copy()
#             res_best=residues[-1]
#             best = term + 1
# =============================================================================

        #else: break
    global d
    d,elbow = get_elbow(residues,xy = False, elbow_point = True)
    terms_best = all_terms[elbow]
    xi_best = all_xi[elbow]
            #undo normalization of best-fit term coefficients
    for sparse_term in terms_best:
        fix_norm=1
        
        for term in sparse_term:
            print(sparse_term)
            fix_norm*=normalizations[int(term)]
        xi_final[sparse_term] = [xi_best[terms_best.index(sparse_term)]/fix_norm]
        xi_real[terms.index(sparse_term)] = xi_best[terms_best.index(sparse_term)]/fix_norm
    
    x = get_lib_x(full_image); theta,_ = create_library(x)
    X,Y = get_xy(full_image)
    surface = theta.T@xi_real; surface = surface.reshape(len(X),len(Y.T))
    if just_fit:
        return surface
    elif terms_in_fit:
        return full_image - surface, terms_best
    elif surf:
        return full_image-surface, terms_best, surface
    else: return full_image - surface
    


def get_r2(orig, fit):
    numer = np.nansum((orig-fit)**2)
    denom = np.nansum((orig - np.nanmean(orig))**2)
    r2 = 1 - numer/denom
    return r2


def subparaboloid(img, full_image = np.array([1]), dx = 1, just_fit = False, fit = False, background_fit = False):
    """Substracts a paraboloid from the image, dx is the px distance in
    which the image is sampled for the fitting"""
    if full_image.shape == (1,): full_image = img.copy()
    X,Y = np.meshgrid(np.arange(0, img.shape[1], dx), np.arange(0, img.shape[0], dx))
    XX = X.flatten()
    YY = Y.flatten()  
    Z = img[YY, XX]
    idx = ~np.isnan(Z)   # Remove NaN values
    data = (np.vstack((XX[idx], YY[idx], Z[idx]))).transpose()
    X,Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    XX = X.flatten()
    YY = Y.flatten()
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    
    global C
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    X,Y = np.meshgrid(np.arange(full_image.shape[1]), np.arange(full_image.shape[0]))
    Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
    if just_fit == True:
        return Z
    elif fit == True:
        return full_image-Z, Z
    elif background_fit:
        return full_image-Z, img-Z, Z
    else:
        return full_image-Z
def fit_lows(image, full_image = np.array([1]), N = 5, distribution=False, percent = 0.1,degree=2,get_images=False,just_fit=False):
    '''input:
        image - 2d array
        full_image - if different than image
        N - can be 1 or 2 dimensions. Number of windows in each axis
        percent - fraction of pixels in each frame to use
       Return:
        fitted'''
    X,Y = cartesian(image)
    if full_image.shape==(1,): full_image = image.copy()
    if not hasattr(N,'__iter__'): N = [N,N]
    
    images = [np.array_split(arr,N[1],axis=1) for arr in np.array_split(image,N[0],axis=0)]
    images_dic = {(j,i):d for i,dd in enumerate(images) for j,d in enumerate(dd)}
    if get_images:
        return images_dic
    XX = [np.array_split(arr,N[1],axis=1) for arr in np.array_split(X,N[0],axis=0)]
    YY = [np.array_split(arr,N[1],axis=1) for arr in np.array_split(Y,N[0],axis=0)]
    
    xs,ys,zs = [],[],[]
    if distribution: plt.figure()
    for n1,(x,y,z) in enumerate(zip(XX,YY,images)):
        for n0,(x,y,z) in enumerate(zip(x,y,z)):
            if np.sum(~np.isnan(z))>0:
                flat = z.copy()#sym_reg(z,1,normal=True)
                args = np.unravel_index(np.argsort(flat.flatten())[int(.01*flat.size):int(percent*flat.size)],z.shape)
            else:
                args = []
            xs.extend(x[args])
            ys.extend(y[args])
            zs.extend(z[args])
            if distribution: plt.hist(z[args],bins=100,label=f'{n0}, {n1}')
    if distribution: plt.legend()
    X,Y,Z = np.array([xs,ys,zs],dtype=float)
    nan_mask = ~np.isnan(Z)
    X = X[nan_mask]
    Y = Y[nan_mask]
    Z = Z[nan_mask]
    theta,_ = create_library([X,Y],degree)
    xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta,theta.T)),theta),Z)
    theta_full,_ = create_library(get_lib_x(full_image),degree)
    fit = np.matmul(theta_full.T,xi).reshape(full_image.shape)
    if just_fit:
        return fit
    return full_image-fit

def get_zoomed(image, bins = [False], means = False,density = False, N = 4):
    data = image.copy()
    x_w = int(np.floor(data.shape[0]/N))
    y_w = int(np.floor(data.shape[1]/N))
    all_counts=[]; loc = []; mean=[];std=[]
    for i in range(N):
        for j in range(N):
            dat = data[i*x_w:(i+1)*x_w,j*y_w:(j+1)*y_w]
            dat,_ = fit_lows(dat)
            if np.sum(~np.isnan(dat))/dat.size< 0.2: continue
            if any(bins):
                counts,_ = np.histogram(dat[~np.isnan(dat)],bins = bins,density = density)
                all_counts.append(counts)
            loc.append([i,j])
            mean.append(np.nanmean(dat))
            std.append(np.nanstd(dat))
    if means:
        return all_counts,loc,mean,std
    return all_counts,loc

def rlength(curves):
    lens=[]
    for curve in curves:
        s = np.polyfit(np.arange(len(curve)),curve,2)
        fit = 0; s=np.flip(s); d=0
        for deg in range(len(s)):
            fit+=s[deg] * np.arange(len(curve))**deg
        for i in range(1,len(fit)):
            d+=np.sqrt( (fit[i] - fit[i-1])**2 + 1 )
        lens.append(d)
    return d

def rwidth(ring_pos):
    N1 = len(ring_pos[0]); N2 = len(ring_pos[1]); avg_ring = []
    for i in sample(range(0,N1),int(np.ceil(N1/4))):
        point = ring_pos[0][i]
        A = np.outer(np.ones(N2),point)
        B = np.array(ring_pos[1])
        avg_ring.append(min(np.linalg.norm(A-B,axis = 1)))
    ring_mean = int(np.round(np.mean(avg_ring)))
    return ring_mean

def rheight(ring_only):
    hs = []
    for row in ring_only:
        hs.append(np.nanmax(row))
    h = np.nanmedian(hs)
    return h
def volume_approx(ring_only,ring_pos):
    width = rwidth(ring_pos)
    ring_pos = np.array(ring_pos).T
    pos = ring_pos[1]
    lens = rlength(pos)
    height = rheight(ring_only)
    vol = np.mean(lens) * width * height
    return vol/10**6
    ring_pos = np.array(ring_pos).T
    pos = ring_pos[1]
    leftlen = rlength(pos[0])
    rightlen= rlength(pos[1])
    width = rwidth(ring_pos)
    height = height(ring_only)
    vol = (leftlen + rightlen)/2 * width * height
    return vol


def clean(image,processed=False,just_full = True):   
    '''Perform a collection of functions to clean the data.
            image: the data to be cleaned
            just_full: bool - return only the full image, otherwise will return all below
        Returns:
            a list containing the homeland, homeland and outside agar, only the coffee ring, just the agar, and the entire data set'''
    global ring_exists
    if type(image)==str:
        file = image
        if image.endswith('.csv'):
            image = get_data(image)
        elif image.endswith('.datx'):
            image = convert_data(image)
        else:
            print('Not a csv or datx..unsure what to do')
            return
        
    if np.sum(~np.isnan(image))==0:
        os.remove(file)
        return False
    
    if processed and just_full:
        return remove_outliers(image)
        
    if not processed:
        image = sym_reg(image,1)
    direction = find_ring(image,direction=True)
    if direction == 'transpose required':
        print('Performing a transpose')
        image = image.T
    
    
# =============================================================================
#     home_only,ring_width,ring_pos = remove_ring(image, to_end = True,width=True)
#     
#     
#     image_no_ring,ring_image = remove_ring(image, return_ring = True)
# =============================================================================
    home_only, ring_only, agar_only, ring_pos,ring_width = remove_ring(image,parts=True)
    image_no_ring = np.where(np.isnan(ring_only),image,np.nan)
   
    fit1 = sym_reg(agar_only,1,just_fit = True)
    no_ring = image_no_ring - fit1
    homeland  = home_only - fit1
    ring      = ring_only - fit1
    agar      = agar_only - fit1
    full_data = image     - fit1
    
    
    #no_ring, fit = fit_lows(no_ring,N=5)
        
    #homeland = home_only - fit1 -fit
    #full_data = image - fit1 - fit
    #ring = ring_image - fit1 - fit
    #agar -= fit1 + fit
    
    data_stuff = [homeland,no_ring,ring,agar,full_data]

    if just_full: return full_data
    return data_stuff, ring_width, ring_pos

def get_cid_from_array(array):
    string = array.tobytes()
    compressor = lzma.LZMACompressor()
    compressor.compress(string)
    compressed = compressor.flush()
    cid = len(compressed)/len(string)
    
    return cid

def get_meta(image):
    try:
        image2 = image[~np.isnan(image)]
        maximum = np.nanmax(image2)
        vol = np.sum(image2)
        avg = np.mean(image2)
        med = np.median(image2)
        stdmed = np.sqrt(np.sum((image2-np.median(image2))**2/image2.size))
        std = np.std(image2)
        coef_of_vari=std/avg
        cov_med = stdmed/med
        ran = max(image2)-min(image2)
        return [avg,std,abs(coef_of_vari),med,stdmed,abs(cov_med),ran,maximum,vol]
    except:
        return (np.nan*np.ones(9)).tolist()

def get_roundings(data, cutoffs = [50,100,200,300], print_cutoffs = False, meta=False):
    a=[];
    for i in cutoffs:
        rounded = np.around(data/i)*i
        if meta:
            a.extend(get_meta(rounded)[2:])
        else:
            a.append(get_cid_from_array(rounded))
        
    if print_cutoffs: 
        return a, cutoffs
    else: return a

def total_hist(X,x,h,bin_width):
    x_old = X.T[np.argsort(X[0])].T
    
    temp = []
    for num,i in enumerate(x):
        newh = h[num]
        if i in x_old[0]:
            index = int((i-x_old[0][0])/bin_width)
            x_old[1][index] += newh
        else:
            temp.append([i,newh])
    temp = np.array(temp).T
    if temp.size == 0: return X
    X2 = np.concatenate((x_old,temp),axis = 1)
    X2 = X2.T[np.argsort(X2[0])].T
    return X2

def get_file_num(top,site_arrangement=['R'],file_type = '.csv',baseline = True):
    file_num=0
    for root, folders, files in os.walk(top):
        for file in files:
            file_info = file.split('_')
            if file.endswith(file_type) and 'clean' not in file.lower() and 'timeseries' not in file.lower():# and '211119' in file:
                try:
                    site = (int(file_info[1])-1)%len(site_arrangement)
                    R_level = site_arrangement[site]
                    if 'b' in R_level and not baseline: continue
                except: pass
                file_num += 1
    return file_num


def confidence_ellipse(x,y,n_std=2.5,patch = False, ax = None, facecolor='none'):
    
    if ax == None and patch == True: fig,ax = plt.subplots()
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x,y)
    pearson = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset
    ell_radius_x = np.sqrt(1+pearson)
    ell_radius_y = np.sqrt(1-pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, alpha = 0.5)

    # Calculating the standart deviation of x from the square root of the variance
    # and multiplying with the given number of std deviation

    scale_x = np.sqrt(cov[0,0]) * n_std
    mean_x = np.mean(x)

    # Calculating the std deviation of y...
    scale_y = np.sqrt(cov[1,1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x,scale_y) \
        .translate(mean_x,mean_y)
    
    a = scale_x * ell_radius_x
    b = scale_y * ell_radius_y

    if patch:
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    else:
        return a, b, mean_x,mean_y
    
def remove_fourier_legs(binary_shift,full_image = False):
    XY = np.column_stack(np.where(binary_shift==1)).T
    dim = 0;
    while dim<2:
        X = XY[dim]
        counts,bins = np.histogram(X,np.arange(min(X),max(X)+1))
        for n,c in enumerate(counts):
            if c<=5 and c !=0:
                low_cut = bins[n+1]
            elif c != 0:
                low_cut = bins[0]
                break
        for n2,c in enumerate(np.flip(counts[n+1:])):
            if c<=5 and c != 0:
                high_cut = bins[-n2+2]
            elif c!=0:
                high_cut = bins[-1]
                break
        XY = np.array([x for x in XY.T if x[dim]>low_cut and x[dim]<high_cut]).T
    
        dim += 1
    
    if full_image:
        image = np.zeros(binary_shift.shape)
        for x,y in XY.T:
            image[x][y] = 1
        return image
    else:
        return XY[1],XY[0]
    
def make_binary_fourier(square_homeland,cut = 0.001,plot=False,mets=False):
    homeland =np.where(np.isnan(square_homeland),np.nanmean(square_homeland),square_homeland)
    ft = fft.fft2(homeland)
    shift = fft.fftshift(ft)
    if plot:
        s = np.log(np.abs(shift))
        sf= s.flatten()
        plot_contour(s,location = 121,cbar=False)
        plt.gca().axis('off')
        ax = plt.subplot(122)
        plt.hist(sf,100)
        i   = np.mean(sf)
        ii  = np.std(sf)
        iii =scipy.stats.skew(sf)
        iv=scipy.stats.kurtosis(sf)
        ax.text(1, 1,'  i:{:.2f}'.format(i),ha='right',va='top',transform = ax.transAxes)
        ax.text(1, 0.95,' ii:{:.2f}'.format(ii),ha='right',va='top',transform = ax.transAxes)
        ax.text(1, 0.9,'iii:{:.2f}'.format(iii),ha='right',va='top',transform = ax.transAxes)
        ax.text(1, 0.85,' iv:{:.2f}'.format(iv),ha='right',va='top',transform = ax.transAxes)
        if mets:
            return shift,[i,ii,iii,iv]
        else:
            return shift
    shift2 = np.where(np.abs(shift)<=cut*np.max(np.abs(shift)),0,1)
    
    return shift2

def cartesian(arr,central=False):
    [I,J] = [i.reshape(arr.shape) for i in get_lib_x(arr)]
    if central:
        I -= np.mean(I)
        J -= np.mean(J)
    return I,J

def polar(arr,ij=[None,None],ints = False):
    I,J = cartesian(arr)
    I -= ij[0] if not ij[0]==None else np.mean(I) if not ints else int(np.mean(I))
    J -= ij[1] if not ij[1]==None else np.mean(J) if not ints else int(np.mean(J))
    R = np.sqrt(I**2+J**2)
    th= np.where(R>0,np.arccos(-I/R),0)
    th[J<0]*=-1
    th[J<0]+=2*np.pi
    return R,th

def radial_avg(image,dk=5,get_x=False,mask=False,avg=['mean'],radii=[]):
    if len(image.shape)==1: image = image.reshape(1,-1)
    image = np.abs(image)
    # Get image parameters
    freqr = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    freqc = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    # Find radial distances
    [kx, ky] = np.meshgrid(freqc,freqr)
    k = np.sqrt(np.square(kx) + np.square(ky));
    R,th = polar(image)
    stop = np.max(R)
    #dif = 1/np.sqrt(np.sum(k<=stop))*dk
    #x = np.arange(0,stop,dif)
    #if get_x: return x
    if type(mask)==int or type(mask)==float:
        I,J = cartesian(image,True)
        image = np.where((I<mask) | (J<mask), np.nan,image)
    elif type(mask)==np.ndarray:
        image = np.where(mask,image,np.nan)
        
    x=[]
    rad = {'mean':[],'std':[],'energy':[]} 
    
    if len(radii)==0:
        r_vals = np.arange(0,stop,dk)
    else:
        r_vals = radii
    for r in r_vals:
        mask = (R>r-dk/2)&(R<=r+dk/2)
        x.append(np.nanmean(k[mask]))
        dat = image[mask]
        if 'mean' in [i.lower() for i in avg]:
            rad['mean'].append(np.nanmean(dat))
        if 'std' in [i.lower() for i in avg]:
            rad['std'].append(np.nanstd(dat))
    if 'energy' in [i.lower() for i in avg]:
        rad['energy'] = np.sum(np.abs(image)**2)
    rad = {i:np.array(d) if i!='energy' else d for i,d in rad.items()}
    x = np.array(x)
    if get_x:
        return x
    return rad,x



def power_spectrum(data,c = None,fmt='-',label = None,new_fig=False,plot=True,log=False,mask=False,dk=5,resolution=None,window=True,alpha=0.8,full=False,avg='mean'):
    window2d = np.ones_like(data)
    if len(data.shape)==2 and window:
        window2d = np.sqrt(np.abs(np.outer(np.hanning(data.shape[0]),np.hanning(data.shape[1]))))
    elif window:
        window2d = np.hanning(data.shape[0])
    data =np.where(np.isnan(data),np.nanmean(data),data)*window2d
    ft = fft.fftn(data)
    shift = fft.fftshift(ft)
    s = np.abs(shift)
    
    if not isinstance(avg, (list, tuple, np.ndarray)): avg = [avg]
    spect=[]
    spectra,x = radial_avg(s,mask=mask,dk=dk,avg=avg)
    for a in avg:
        spect.append(spectra[a])
    if len(spect)==1: spect = spect[0]
    lamb = x
    if not plot:
        if full:
            return x,spect,shift*np.conj(shift)
        else: return lamb, spect
    elif len(avg)>1:
        print('Not made to plot multiple version of spectra.')
    if new_fig: plt.figure()
    if log:
        plt.loglog(x,spect,ls=fmt,c=c,alpha = alpha,label=label,)
    else:
        plt.semilogy(x,spect,ls=fmt,c=c,alpha = alpha,label=label)
    plt.xlabel('Wavenumber (1/px)')
    if resolution != None:
        tick_labels = [0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2e3,5e3,1e4,2e4,5e4]
        def ktoL(k,resolution=resolution*(1e6 if np.log10(resolution)<-4 else 1)):
            return resolution/k
        def Ltok(l,resolution=resolution*(1e6 if np.log10(resolution)<-4 else 1)):
            return resolution/l
        from matplotlib.ticker import FormatStrFormatter
        ax=plt.gca().secondary_xaxis('top',functions=(ktoL,Ltok))
        ax.invert_xaxis()
        ax.set_xticks(tick_labels)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('Wavelength (um)')
    plt.pause(0.05)
    #x = 1/np.flip(x)
    if label!=None:
        plt.legend()
    if full:
        print('True')
        return x,spect,shift*np.conj(shift)
    return x,spect

def spatial_corr(x,spectrum,N=1000,dN=1):
    '''https://en.wikipedia.org/wiki/Correlation_function_(astronomy)'''
    e = []
    if x[0] == 0:
        k = np.array(x[1:])
        P = np.array(spectrum[1:]) 
    else:
        k = np.array(x)
        P = np.array(spectrum)
    r = np.arange(1,N+1,dN)
    for rr in r:
        inte = 1/(2*np.pi**2)
        inte *= np.sum(k**2 * P * np.sin(k*rr) / (k*rr))
        e.append(inte)
    r = np.arange(1,N+1,dN)
    return r,e

def my_fft(sq):
    I,J = [np.arange(sq.shape[i]) for i in range(2)]
    interp = scipy.interpolate.RectBivariateSpline(I,J,sq)
    points = np.random.random((5000,2))*I.size
    Kx,Ky = [(np.arange(0,1,0.005)-0.5) for i in range(2)]
    Fk = np.zeros((Kx.size,Ky.size),dtype=np.csingle)
    hvals = [interp.ev(x,y) for (x,y) in points]
    t1 = time.time()
    kmesh = np.meshgrid(Kx,Ky)
    a=np.sum([np.outer(kmesh[i],points.T[i]).reshape(kmesh[i].shape+tuple([points.shape[0]])).T for i in range(2)],axis=0)
    b = np.matmul(np.exp(1j*a).T,hvals).T
    t2 = time.time()
    for nx,kx in enumerate(Kx):
        for ny,ky in enumerate(Ky):
            Fk[nx,ny] = np.dot(hvals,np.ones(len(hvals))*np.exp(1j*(np.dot(np.array([kx,ky]),points.T))))
    plot_contour(sq, location=131,axis=False,cbar=False),plot_contour(np.log10(np.abs(Fk)),new_fig=False,axis=False,cbar=False,location=132)
    plot_contour(np.log10(np.abs(fft.fftshift(fft.fft2(sq)))),new_fig=False,location = 133,axis=False,cbar=False)
    print(t2-t1)
    print(time.time()-t2)
    return b,Fk

def get_color(l):
    col_dict = {'H':'black','R':'blue','S':'red'}
    if type(l)==str:
        return col_dict[l[0].upper()]
    l = [col_dict[i.lower()[0]] for i in l]
    return np.array(l)

def get_corners(arr,corners=['tl','tr','bl','br'],minR_factor=1,reverse=False):
    X = np.array([np.arange(arr.shape[1]) for i in range(arr.shape[0])])-arr.shape[1]/2
    Y = np.array([np.arange(arr.shape[0]) for i in range(arr.shape[1])]).T-arr.shape[0]/2
    bigR = np.sqrt(arr.shape[0]**2 + arr.shape[1]**2)/2
    minR = min(arr.shape)/2*minR_factor
    R = minR+(bigR - minR)/4
    
    rev = -1 if reverse else 1
    no = np.bool_(np.zeros(X.shape))
    border = np.where(((X**2+Y**2)*rev>R**2*rev)&(
                      (((X>0) & (Y>0)) if 'br' in corners else no)|
                      (((X>0) & (Y<0)) if 'tr' in corners else no)|
                      (((X<0) & (Y<0)) if 'tl' in corners else no)|
                      (((X<0) & (Y>0)) if 'bl' in corners else no)),arr,np.nan)
    return border

def get_square_home(home,coords,rect = False):
    if len(coords)==1: coords = coords[0]
    left = np.array(coords[0]).T
    right = np.array(coords[1]).T
    
    if left[1][0] < left[1][500] or left[1][-1] < left[1][500]:
        width = int(np.floor(min(left[1])))
        buffer = int(np.floor((home.shape[0] - width)/2))
        if rect:
            square = home[:,:width]
        else:
            square = home[buffer:home.shape[0]-buffer-1,:width]
    else:
        width = int(np.ceil(max(right[1])))
        buffer = int(np.floor((home.shape[0] - width)/2))
        if rect:
            square = home[:,width:]
        else:
            square = home[buffer:home.shape[0]-buffer-1,int(np.ceil(max(right[1]))):]
    square =np.where(np.isnan(square), np.nanmean(square), square)
    
    return left,right,square


def get_metrics(square_home,home,ring,full,print_names = False):
    names = ['100nm CID', '1nm CID', 'All_mean', 'All_stdev', 
               'Ring_biohist_peak_loc', 'Ring_Median', 'Ring_stdev',
               'Home_FTpoints', 'Home_ellipse','Home_size','Home_skewness','Home_kurtosis','NaNs']
    if print_names: return names
    metrics = {}
    
    cids,labs = get_roundings(full,cutoffs = [100,1],print_cutoffs=True)
    for n,l in enumerate(labs):
        metrics[str(l)+'nm CID'] = cids[n]
    
    
    metrics['All_mean'] = np.nanmean(full)
    metrics['All_stdev']= np.nanstd(full)
    
    counts, bins = np.histogram(ring[~np.isnan(ring)],np.arange(np.nanmin(ring),np.nanmax(ring)))
    sort = np.argsort(counts*bins[:-1])
    max_loc = bins[sort][-1]
    metrics['Ring_biohist_peak_loc'] = max_loc   
    
    metrics['Ring_Median'] = np.nanmedian(ring)
    metrics['Ring_stdev'] = np.nanstd(ring)
    
    
    ft = make_binary_fourier(square_home)
    X,Y = remove_fourier_legs(ft)
    a,b = confidence_ellipse(X,Y)
    
    metrics['Home_FTpoints'] = np.sum(ft)
    metrics['Home_ellipse'] = np.pi*a*b
    
    metrics['Home_size'] = np.sum(~np.isnan(home))
    metrics['Home_skewness']=scipy.stats.skew(home.flatten(),nan_policy = 'omit')
    metrics['Home_kurtosis']= scipy.stats.kurtosis(home.flatten(),nan_policy='omit')
    
    metrics['NaNs'] = np.sum(np.isnan(full))

    
    return list(metrics.keys()),list(metrics.values())
    
    
    
def est_finish(cleaned_count,total_files,start_clean_time,t0):
    global cur_t,avg_t,end_t
    print('Completed/Total: {}/{}'.format(cleaned_count,total_files))
    cur_t = time.time()
    avg_t = (cur_t-t0)/cleaned_count
    end_t = time.asctime(time.localtime(cur_t + avg_t*(total_files-cleaned_count)))
    print('Last file: {:.4}'.format(cur_t-start_clean_time))
    print('Average so far: {:.4}'.format(avg_t))
    print('Current Time:',time.asctime())
    print('Estimated Finishing Time:', end_t)
    print()

def zoom(data,ith,N=5):
    if ith>=N**2:
        print('Not valid location')
        return []
    
    x_w = int(np.floor(data.shape[0]/N))
    y_w = int(np.floor(data.shape[1]/N)); n=0
    for i in range(N):
        for j in range(N):
            if n == ith:
                return data[i*x_w:(i+1)*x_w,j*y_w:(j+1)*y_w],[[i*x_w,(i+1)*x_w],[j*y_w,(j+1)*y_w]]
            n+=1
    
def zoom_metrics(zoom,print_names = False):
    names = ['100nm CID', '1000nm CID', 'Mean', 'Median', 'Stdev', 'Skewness',
             'Biohist peak','Fourier points','NaNs']        #'Kurtosis',
    if print_names: return names
    metrics = {}
    
    cids,labs = get_roundings(zoom,cutoffs = [100,1000],print_cutoffs=True)
    for n,l in enumerate(labs):
        metrics[str(l)+'nm CID'] = cids[n]
    
    
    metrics['Mean']  = np.nanmean(zoom)
    metrics['Median']= np.nanmedian(zoom)
    metrics['Stdev'] = np.nanstd(zoom)
    metrics['Skewness']= scipy.stats.skew(zoom[~np.isnan(zoom)])
    #metrics['Kurtosis']= scipy.stats.kurtosis(zoom[~np.isnan(zoom)])
    
    counts, bins = np.histogram(zoom[~np.isnan(zoom)],np.arange(np.nanmin(zoom),np.nanmax(zoom)))
    sort = np.argsort(counts*bins[:-1])
    max_loc = bins[sort][-1]
    metrics['Biohist peak'] = max_loc   
    
    
    ft = make_binary_fourier(zoom)
    
    metrics['Fourier points'] = np.sum(ft)
    
    
    metrics['NaNs'] = np.sum(np.isnan(zoom))
    
    keys = list(metrics.keys())
    if not keys==names:
        print('Names and keys don\'t match')
        a = [i for i in names if i not in keys]
        b = [i for i in keys if i not in names]
        return a,b
    
    return list(metrics.keys()),list(metrics.values())

def get_zoomed_images(image,title = None, N = 4, histogram = False):
    data = image.copy()
    x_w = int(np.floor(data.shape[0]/N))
    y_w = int(np.floor(data.shape[1]/N))
    out = []; loc=[]
    if histogram:
        l = 121
    else: l = 111
    for i in range(N):
        for j in range(N):
            loc.append([[i*x_w,(i+1)*x_w],[j*y_w,(j+1)*y_w]])
            dat = data[i*x_w:(i+1)*x_w,j*y_w:(j+1)*y_w]
            #dat,_ = fit_lows(dat)
            
            plot_contour(dat,location=l,vlims=[0,1000])
            plt.title(title)
            if histogram:
                ax = plt.subplot(122)
            
                ax.hist(dat.flatten(),bins = np.arange(0,1000+1,20))
            plt.savefig('test.jpg')
            im = plt.imread('test.jpg')
            out.append(im)
    return out,loc

def show_zoom(data,locations,ith,subplot = 111):
    y,x = data.shape
    outer = [[0,0],[x,0],[x,y],[0,y],[0,0]]
    if type(ith)==int: ith = [ith]
    verts=[]
    for i in ith:
        loc = locations[i]
        y=loc[0];x=loc[1]
        inner = [[x[0],y[0]],[x[0],y[1]],[x[1],y[1]],[x[1],y[0]],[x[0],y[0]]]
        verts.extend(inner)
    verts = outer+verts
    codes = np.ones(len(verts))*Path.LINETO
    codes[::5] = np.ones(len(codes[::5]))*Path.MOVETO
    path = Path(verts,codes)
    patch = PathPatch(path,facecolor='black',alpha=0.5)
        
    plot_contour(data,location=subplot)
    ax = plt.gca()
    ax.add_patch(patch)

def approx_slopes(data, px_to_r = 1000/868):
    grad = np.gradient(data)
    slope = np.sqrt(grad[0]**2+grad[1]**2)
    reslope = slope/1000*px_to_r
    angles = np.where(~np.isnan(reslope),np.arctan(reslope),np.nan)*180/np.pi
    return angles

def make_confusion(df,get_r = False,model='Model_KMeans',truth='Class'):
    '''assumes it gets most correct'''
    truth = np.array(list(df[truth]))
    model = np.array(list(df[model]))
    cipher = model[truth=='R']
    count ={'0':0,'1':0}
    for i in cipher:
        count[i[-1]]+=1
    R = np.argmax(list(count.values()))
    S = [1,0][R]
    confusion = np.zeros((2,2))
    for t,m in zip(truth,model):
        confusion[int(t=='S'),0 if int(m[-1])==R else 1]+=1
    if get_r:
        return np.int_(confusion),{R:'R',S:'S'}
    else: return np.int_(confusion)
def make_confusion(df,get_cipher = False,model='Model_KMeans',truth='Class'):
    '''assumes it gets most correct'''
    truth = np.array(list(df[truth]))
    model = np.array(list(df[model]))
    labels,counts = np.unique(truth,return_counts=True)
    cipher = {str(i):None for i in range(len(labels))}
    index  = {i:n for n,i in enumerate(labels)}
    for label in np.flip(labels[np.argsort(counts)]):
        u,c =np.unique([i[-1] for i in model if label in i],return_counts=True)
        ind = np.flip(np.argsort(c))
        num=0
        try:
            while cipher[u[ind[num]]] is not None:
                num+=1
            cipher[u[ind[num]]]=label
        except:
            for i,d in cipher.items():      ####This is probably not general
                if d is None:
                    cipher[i] = label 
                    break
    confusion = np.zeros((len(labels),len(labels)))
    for t,m in zip(truth,model):
        m = cipher[m[-1]]
        confusion[index[t],index[m]]+=1
    if get_cipher:
        return np.int_(confusion),cipher
    else: return np.int_(confusion)
    
def make_confusion_from_results(df,model='model_SVM-linear',truth='Class'):
    if 'model' not in model: model = 'model_'+model
    truth = np.array(df[truth].values)
    model = np.array(df[model].values)
    size = 2
    mat = np.zeros((size,size))
    for t,m in zip(truth,model):
        i = ['R','S'].index(t)
        j = ['R','S'].index(m)
        #j = int(np.logical_not(bool(i)) ^ m)
        mat[i][j]+=1
    return mat
def plot_comparison(df,real,model,new_fig=False):
    if not new_fig:
        fig = plt.figure('Comparison Confusion Matrix')
        ax = plt.gca(); ax.remove(); ax = fig.add_subplot()
    else:
        fig,ax = plt.subplots()
    x_labels = np.unique(df[model])
    y_labels = np.unique(df[real])
    new_df = pd.DataFrame(np.zeros((len(y_labels),len(x_labels))),
                          columns = x_labels, index = y_labels)
    for g,m in zip(df[real],df[model]):
        new_df.at[g,m] += 1
    mat = np.int_(new_df.values)
    tot = np.sum(mat,axis=1); tot = np.where(tot==0,1,tot)
    percents = (mat.T/tot).T
    percents = np.concatenate((percents,[[1,0]]*len(mat)),axis=1)
    ax.imshow(percents,cmap='coolwarm')
    ax.set_xticks(ticks=np.arange(len(x_labels)),labels=x_labels)
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
                ax.text(j,i,str(int(mat[i][j])),color='white',size=size,ha='center',va='center')
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

def meta_from_excel(fold='',file=''):
    f=0
    if len(fold)>0:
        for root, folds, files in os.walk(fold):
            for file in files:
                if not file.endswith('.xlsx'): continue
                if not len(file.split('_'))==3: continue
                excel = root+'/'+file
                xl = np.array(pd.read_excel(excel),dtype=str)
                for n,i in enumerate(xl):
                    if i[0].startswith('File'):
                        break
    elif len(file)>0:
        xl = np.array(pd.read_excel(file),dtype=str)
        for n,i in enumerate(xl):
            if i[0].startswith('File'):
                break
            elif 'Interferometry_number' in i:
                f=list(i).index('Interferometry_number')
                break
    d = xl[n+1:,1:]
    data_categ = xl[n][1:]
    file_names = xl[n+1:].T[f]
    dfd = {f:{c:d[fn,cn] for cn,c in enumerate(data_categ)} for fn,f in enumerate(file_names)}
    return dfd

def designation(arr):
    new = []
    for i in arr:
        i = str(i).upper()
        if i.startswith('R'):
            new.append('R')
        elif i.startswith('S'):
            new.append('S')
        elif i.startswith('H'):
            new.append('HR')
        else:
            new.append(i)
        
    return new

def transfer(mat_file,size=[[None,None],[None,None]],hold_out=[],destination='Weiss Lab Data/For classification/Transfer'):
    data = scipy.io.loadmat(mat_file)
    counts={'R':0,'H':0,'S':0}
    shutil.rmtree(destination)
    os.mkdir(destination)
    n=0;t0=time.time()
    for i in data:
        if i.startswith('__'): continue
        if i[-1] in hold_out: continue
        print('Saving file:',i)
        start=time.time()
        n+=1
        d = data[i][size[0][0]:size[0][1],size[1][0]:size[1][1]]
        des = i.split('_')[-1]
        counts[des]+=1
        np.savetxt(destination+'/'+des+str(counts[des])+'.csv',d,delimiter=',')
        est_finish(n,len(data)-3,start,t0)


def get_inds(arr):
    X = np.array([np.arange(arr.shape[1]) for i in range(arr.shape[0])])
    Y = np.array([np.ones(arr.shape[1])*i for i in range(arr.shape[0])])
    return X,Y

def lmat(file,mat = {}):
    scipy.io.loadmat(file,mat)
    mat = {i: mat[i] for i in mat if not i.startswith('__')}
    return mat

def vol_dilutions(data_dict,latres=False,log=False,norm=True,sns = ['4','7','10','11'],ods =[0.1,0.2,0.3,0.4,0.5],close=True,best_fit=True):
    if close:
        plt.close('Volumes')
        plt.close('Avg Volumes')
    #plt.figure('Volumes')
    plt.figure('Avg Volumes')
    cmap = plt.get_cmap('Set1')
    vols = {sn:{od:[] for od in ods} for sn in sns}
    stds = vols.copy()
    this_vols=[]
    if type(latres)==bool:
        mult = 1
        axis = 'mm*px^2'
    else:
        mult = (latres*1000)**2
        axis = 'mm^3'
    for i,d in data_dict.items():
        #print(i)
        sn = i.split('_')[0]
        if sn not in vols: continue
        col = cmap(sns.index(sn))
        od = float(i.split('_')[2])
        vol = np.sum(d)/1000/1000*mult
        #stds[sn][int(float(od)*10)-1]= np.std(this_vols) if len(this_vols)>1 else np.sqrt(this_vols[0])
        vols[sn][od].append(vol)
    stds = {sn:{od:np.std(data) if len(data)>1 else np.sqrt(data[0])/100 for od,data in sn_dict.items()} for sn,sn_dict in vols.items()}
    vols = {sn:{od:np.mean(data) for od,data in sn_dict.items()} for sn,sn_dict in vols.items()}
    plt.ylabel('Approx Volume [%s]' %axis)
    plt.xlabel('OD (5ul)')
    plt.figure('Avg Volumes')
    for i,d in vols.items():
        if norm:
            ref = list(d.values())[0]
        else:
            ref = 1
        x = np.array(list(d.keys()))
        y = np.array(list(d.values()))/ref
        st= np.array(list(stds[i].values()))/ref
        plt.errorbar(x,y,st,fmt = '-o',color = cmap(sns.index(i)),alpha=0.9,label=i)
    if norm:
        plt.plot(x,x/x[0],'--k',alpha=0.7,label='Linear')
        plt.ylabel('Relative Approx. Volume')
    else:
        plt.ylabel('Approx Volume [%s]' %axis)
    plt.xlabel('OD (5ul)')
    plt.legend()
    fig = plt.figure('Avg Volumes')
    ax = plt.gca()
    if best_fit:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        height = bbox.height*fig.dpi
        m=0;fits={}
        for n,(s,v) in enumerate(vols.items()):
            if any([i==0 for i in stds[s]]): continue
            std = np.array(list(stds[s].values()))
            x = ods
            y = np.array(list(v.values()))
            slope,intercept = np.polyfit(np.log(x),np.log(y),1,w=1/np.log(std))
            yfit = np.exp(slope*np.log(x)+intercept)
            fig = plt.figure('Avg Volumes')
            ax = plt.gca()
            ax.plot(x,yfit,color=cmap(sns.index(s)),alpha=0.5)
            ax.text(1,0+15/height*n,s+': '+str(slope)[:4],ha='right',transform=ax.transAxes)
            m+=1
            fits[s]=[slope,intercept,y[0]]
        ax.text(0.95,15/height*(m+1),'Slopes',transform=ax.transAxes,ha='center')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    if best_fit:
        return vols,stds,fits
    else:
        return vols,stds
    

def power_from_full(dic,rad=800,desired_od=0.1):
    '''not a general function'''
    plt.close('all')
    ods = {'0.1':1,'0.2':1}
    homes = {}
    for i,d in dic.items():
        od = i.split('_')[2]
        per = '_'.join(i.split('_')[-2:])
        if str(desired_od)!= od: continue
        I,J = cartesian(d,True)
        dat = np.where((np.abs(I)<rad) & (np.abs(J)<rad),d,np.nan)
        bI = int(I[-1][-2]-rad)
        bJ = int(J[-1][-1]-rad)
        c1 = sym_reg(dat,2,normal=True)[bI:bI+2*rad,bJ:bJ+2*rad]
        plt.figure(od)
        plot_contour(c1,location=(6,6,ods[od]), new_fig=False,cbar=False,axis=False)
        plt.title(per)
        ods[od]+=1
        c1 = np.where(np.isnan(c1),np.nanmean(c1),c1)
        homes[i] = c1
        ft = 1+np.log10(np.abs(np.fft.fftshift(np.fft.fft2(c1))))
        plot_contour(ft,cmap='gray',cbar=False,axis=False,new_fig=False,location=(6,6,ods[od]))
        ods[od]+=1
        plt.figure('Power')
        col = {'R':'red','H':'blue','S':'black'}[i[-1]]
        x,spect = power_spectrum(c1,c=col,label=per,log=True)
        
def update_database(new_data,time='8hr',drug='TZP',od='0.2',save=False):
    root = 'my data/for classification/'
    cur_files = os.listdir(root)
    files = []
    mat = new_data
    for file in cur_files:
        if not file.endswith('.mat'): continue
        spl = file[:-4].split('_')
        if not (time==spl[0] and drug==spl[1] and od==spl[2]): continue 
        files.append([int(spl[3]),file])
    if len(files)>0:
        file = files[np.argsort(np.array(files).T[0])[-1]][1]
        mat = lmat(root+file,new_data)
    new_file = '_'.join([time,drug,od])+'_'+str(len(mat))+'.mat'
    if save:
        scipy.io.savemat(new_file,mat)
    return mat

def make_PAP_curves(file='my data/clinical isolates/Selected_PAP_data.xlsx'):
    df = pd.read_excel(file,sheet_name='Percent Survival',header=0,index_col=1)
    plt.close('all')
    plt.figure()
    plt.plot(df.columns[1:],np.ones(len(df.columns)-1)*10**(-6),'--k')
    for sn in df.index:
        print(sn)
        bkpt = df.loc[sn][1]
        line = '-' if bkpt>0.5 else '--' if bkpt>10**(-6) else ':'
        plt.loglog(df.columns[1:],np.array(list(df.loc[sn][1:])),line+'o',label=sn)
    plt.legend()
    plt.plot([1,1],plt.ylim(),'--k',alpha=0.3)
    plt.xlabel('Breakpoint AB (TZP)')
    plt.ylabel('Relative CFU')
    plt.set_cmap('tab20')
    
def reclassify(des_dic,data):
    '''des_dic can be a filename or the pandas dataframe
       data should be the dictionary of data'''
    if type(des_dic)==str:
        des_dic = pd.read_excel(des_dic,header=0)
    new_data = {}
    for i,d in data.items():
        strain_num = int(i.split('_')[0]) #re.split('(\d+)',file[:-5])[1] + '_CRE'*('CRE' in file)
        try:
            strain_rep = int(i.split('_')[1])
        except:
            strain_rep = 1
        strain_fold= i.split('-')[1].split('_')[0] if '-' in i else ''
        des = list(des_dic.loc[[sn==strain_num and r==strain_rep and f==strain_fold for sn,r,f in zip(des_dic['Strain Number'],des_dic['Replicate'],des_dic['Folder'])]]['Classification'])[0]
        new_i = i[:-1]+des 
        new_data[new_i] = d
    return new_data

def create_images(imgs,dt=20):
    newimgs=[]
    for n,img in enumerate(imgs):
        if n!=4: continue
        plot_contour(img,clear=True)
        plt.text(*np.flip(img.shape[:2]),str(dt*n)+' min\n'+str(dt*n/60)[:3]+' hr',ha='right',va='top',color='white',font='Times New Roman',size=22)
        rect = plt.Rectangle([0,0],img.shape[1]*n/(len(imgs)-1),img.shape[0]/20,color='white')
        plt.gca().add_patch(rect)
        for lab in [1,2,4,8,12,24,36,48]:
            if lab>dt*(len(imgs)-1)/60:continue
            x = lab*60/(dt*(len(imgs)-1))
            atmax = x>0.95
            plt.text(min([x,1])*img.shape[1],img.shape[0]/20,str(lab)+' hrs',ha='center' if not atmax else 'right',va='bottom',color='red',font='Times New Roman',size=18)
            plt.plot(x*img.shape[1]*np.ones(2),[0,img.shape[0]/20],'r')
        plt.savefig('test.png',bbox_inches='tight',pad_inches=0)
        newimgs.append(plt.imread('test.png'))
    return newimgs
import cv2#,rawpy
#def make_video(imgs,filename,fps=10):
#    if type(imgs[0])==str:
#        newimgs = []
#        for imgf in imgs:
#            raw = rawpy.imread(imgf)
#            img = raw.postprocess()
#            newimgs.append(img)
#        imgs = newimgs.copy()
#        imgs = create_images(imgs,20)
#    video = cv2.VideoWriter('.'.join(filename.split('.')[:-1])+f'-{fps}fps.mp4',0x7634706d,fps,tuple(np.array(imgs[0].shape)[[1,0]]))
#    for img in imgs:
#        video.write(img)
#    video.release()
#    cv2.destroyAllWindows()
    
    
    
from sklearn.cluster import KMeans
def lab_scores(scores,n_clusters=3):
    '''rtlc = ['aawaz','maryam','ray','pablo','emma','miles','adam'];
hr = ['pablo','ray','adam','miles','maryam','aawaz','emma'];
mb = ['ray','miles','adam','aawaz','pablo','maryam','emma'];
names = ['Miles','Pablo','Aawaz','Adam','Emma','Ray','Maryam']
scores = np.array([[i.index(name.lower()) for i in [rtlc,hr,mb]] for name in names])'''
    fig = plt.figure();ax = fig.add_subplot(projection='3d')
    model = KMeans(n_clusters = n_clusters).fit(scores)
    ax.scatter(*(scores.T),color=np.array(['red','blue','black'])[model.labels_])
    ax.set_xlabel('RTLC')
    ax.set_ylabel('HR')
    ax.set_zlabel('MB')
#plt.close('all')
#lab_scores(scores,2)
    

def curvature(img,latres=1):
    '''Curvature found in https://mathworld.wolfram.com/Curvature.html eq 17. (Gray 1997)
        img: 2d image array
        latres: k given in units of [latres]^-1'''
    dx,dy = np.gradient(img,latres)
    dxx,dxy = np.gradient(dx,latres)
    dyx,dyy = np.gradient(dy,latres)
    return (dxx*dy**2 - 2*dxy*dx*dy + dyy*dx**2)/(dx**2+dy**2)**(3/2)

def ring_width(arr,plot=False,vlims=[None,None],edges=False,all_data=False,just_ring=False,ax = None,**kwargs):
    arr = np.where(np.isnan(arr),0,arr)
    max_inds = np.argmax(arr[:,500:],axis=1) + 500*np.ones_like(arr[:,0])
    med_ind = np.median(max_inds)
    min_ind = int(med_ind - np.std(max_inds)*2)
    max_ind = int(med_ind + np.std(max_inds)*2)
    max_inds = np.argmax(arr.T[min_ind:max_ind],axis=0)+min_ind
    max_vals = np.diag(arr[:,max_inds])
    #if just_ring:
    #    return max_vals
    start_counter=1
    med = 0
    while True:
        start_counter+=1
        inds = max_inds.copy()
        test = np.array([row[i] for row,i in zip(arr,inds)])>max_vals/2
        while any(test):  
            inds = inds - np.int_(test)
            if any((inds>=arr.shape[1]-1)|(inds==0)):
                break
            test = np.array([row[i] for row,i in zip(arr,inds)])>max_vals/2
        start = inds.copy()
        #break
        med_start = int(np.median(start) - 3*np.std(start))
        if np.sum(max_inds<med_start)>0 and start_counter<10:
            max_inds = np.argmax(arr.T[med_start:],axis=0)+med_start
            max_vals = np.diag(arr[:,max_inds])
        else:
            break

    med_start = int(med_start) - 2 * np.std(start)
    end_counter=1
    while True:
        end_counter+=1
        inds = max_inds.copy()
        test = np.array([row[i] for row,i in zip(arr,inds)])>max_vals/2
        while any(test):        
            inds = inds + np.int_(test)
            if any((inds>=arr.shape[1]-1)|(inds==0)):
                break
            test = np.array([row[i] for row,i in zip(arr,inds)])>max_vals/2
        end = inds.copy()
        #break
        med_end = int(np.median(end) + 2 * np.std(end))
        if np.sum(max_inds>med_end)>0 and end_counter<10:
            max_inds = np.array([np.argmax(row[s:e],axis=0) for row,s,e in zip(arr.T,start,end)])+start
            max_vals = np.diag(arr[:,max_inds])
        else:
            break
    if False:
        plot_contour(arr)
        plt.gca().scatter(max_inds,np.arange(len(max_inds))[::-1],color='k',s=1)
        plt.gca().scatter(start,np.arange(len(start))[::-1],color='gray',s=1)
        plt.gca().scatter(end,np.arange(len(max_inds))[::-1],color='gray',s=1)
        plt.pause(1)
            
    widths = end - start
    if ax is not None:
        plot=True
    if plot:
        if ax is None:
            fig,ax = plt.subplots(1,1)
        plot_contour(arr,ax=ax,vlims=vlims,cbar=False)
        ax.plot(np.flip(max_inds)-np.ones_like(max_inds),np.arange(len(max_inds)),'k')
        ax.plot(np.flip(start),np.arange(len(start)),'gray')
        ax.plot(np.flip(end  ),np.arange(len(end  )),'gray')
    if edges:
        return start,end,widths
    elif all_data:
        return start,end,widths,max_inds
    return widths

def ring_width_radial(arr, plot=False, vlims=[None, None], edges=False, all_data=False, just_ring=False, ax=None, **kwargs):
    arr = np.where(np.isnan(arr), 0, arr)
    r, th = polar(arr)
    unique_angles = 2*np.pi * np.arange(0, 1000) / 1000
    max_vals = []
    max_inds, start_inds, end_inds = [],[],[]
    max_r, start_r, end_r = [],[],[]
    #
    for angle in tqdm(unique_angles,desc='Calc radial ring widths', leave=False):
        mask = np.abs(th - angle) < (2 * np.pi / len(unique_angles)) / 2
        radial_values = arr[mask]
        radial_indices = r[mask]
        y_indices, x_indices = np.where(mask)
        args = np.argsort(radial_indices)
        radial_values = radial_values[args]
        radial_indices = radial_indices[args]
        indices = np.array([x_indices[args], y_indices[args]]).T
        if len(radial_values) == 0:
            continue
        #
        max_idx = np.argmax(radial_values)
        max_val = radial_values[max_idx]
        max_vals.append(max_val)
        max_r.append(radial_indices[max_idx])
        max_inds.append(indices[max_idx])
        #
        # Find start index
        start_idx = max_idx
        while start_idx > 0 and radial_values[start_idx] > max_val / 2:
            start_idx -= 1
        start_r.append(radial_indices[start_idx])
        start_inds.append(indices[start_idx])
        #
        # Find end index
        end_idx = max_idx
        while end_idx < len(radial_values) - 1 and radial_values[end_idx] > max_val / 2:
            end_idx += 1
        end_r.append(radial_indices[end_idx])
        end_inds.append(indices[end_idx])
    start_inds = np.array(start_inds) - np.mean(start_inds, axis=0)
    end_inds = np.array(end_inds) - np.mean(end_inds, axis=0)
    #
    widths = np.array(end_r) - np.array(start_r)
    #
    if just_ring:
        return np.array(max_vals)
    #
    if ax is not None:
        plot = True
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        plot_contour(arr, ax=ax, vlims=vlims, cbar=False)
        print(np.array(max_inds).T)
        print(np.array(max_inds).shape)
        ax.plot(*np.array(max_inds).T, 'k')
        ax.plot(*np.array(start_inds).T, 'gray')
        ax.plot(*np.array(end_inds).T, 'gray')
        plt.figure()
        plt.plot(max_r, color= 'k')
        plt.plot(start_r, color='gray')
        plt.plot(end_r, color='gray')
    #
    if edges:
        return np.array(start_inds), np.array(end_inds), widths
    elif all_data:
        return np.array(start_r), np.array(end_r), widths, np.array(max_r)
    return widths



import sys
sys.path.append(os.path.dirname(__file__))
from roughness_PB import getwloc,w_data_extraction,linregress
def get_features(image,obj=5.5,fftval=18,these_features=None):
    latres = {5.5:1.5614175326405541,50:1.7334291543917991e-01,5:1.6417910447761194} #latres in micron/px
    latres = latres[obj]
    all_feats = ['Home Height','Home Var','Home CoV','Ring Height','Ring Width','Ring CoV','Ring w_sat','Ring Hurst','ringH/homeH','ringH*homeH']
    if fftval is not None:
        if not hasattr(fftval,'__iter__'): fftval=[fftval]
        for val in fftval:
            all_feats.append('Power-{}um'.format(str(val)[:4]))
    if image is None: return all_feats
    image = np.where(np.isnan(image),np.nanmedian(image),image)
    home = image[300:700,:400] if obj==5 else image[:,:700]
    ring = remove_outliers(np.array([np.max(row) for row in image]),N=100,fill='median')
    coefs = np.polyfit(np.arange(len(ring)),ring,1)
    fit = np.sum([np.arange(len(ring))**(1-d)*c for d,c in enumerate(coefs)],axis=0)
    flucs = ring-fit
    rh = np.median(ring); rv = np.var(flucs)
    rw = np.median(ring_width(image))
    med = np.median(home); var = np.var(home)
    
    if fftval is not None:
        x,spect=np.array(power_spectrum(home,plot=False))
        powerfeats = []
        for val in fftval:
            index = np.argmin(np.abs(1/x*latres-val))
            powerfeats.append(np.log(spect[index]))

    loc, wloc = getwloc(flucs, latres, rx=0.3)
    l_sat, w_sat, h = w_data_extraction(loc,wloc)
    hurst = linregress(np.log10(loc)[:15],np.log10(wloc)[:15]).slope

    feats = [med,var,var/med,rh,rw,rv/rh,w_sat,hurst,rh/med,rh*med,np.log(spect[index])]+powerfeats#,spect[0],m]
    feats = {i:d for i,d in zip(all_feats,feats)}
    return feats


def get_ring(arr, n=1000):
    dn = 2*np.pi/n
    r,th = polar(arr)
    ring,widths = [],[]
    for ang in np.arange(dn/2,2*np.pi,dn):
        mask = np.abs(th - ang)<dn/2
        ring.append(np.nanmax(arr[mask]))
        sector = arr[mask]
        slic   = r[mask]
        rpeak = slic[sector==ring[-1]]
        min_mask = r*mask*(r<rpeak)*(arr<ring[-1]/2)
        rmin = np.max(min_mask)
        max_mask = r*mask*(r>rpeak)*(arr<ring[-1]/2)
        max_mask = np.where(max_mask==0,np.max(r),max_mask)
        rmax = np.min(max_mask)
        w = rmax-rmin
        widths.append(w)
    return np.array(ring),np.array(widths)

def get_features_full(image,latres,fftval=18,homesize=400,ring_slices=1000,feature_names=['Home Height','Home Var','Home CoV','Ring Height','Ring Width','Ring CoV','Ring w_sat','Ring Hurst','ringH/homeH','ringH*homeH']):
    these_features = feature_names.copy()
    if fftval is not None:
        if not hasattr(fftval,'__iter__'): fftval=[fftval]
        for val in fftval:
            these_features.append('Power-{}um'.format(val))
    if image is None: return these_features
    image = np.where(np.isnan(image),np.nanmedian(image),image)
    home = image[300:700,300:700]
    ring,widths = get_ring(image,n=ring_slices)
                                            #make fft for coffee ring 
    coefs = np.polyfit(np.arange(len(ring)),ring,1)
    fit = np.sum([np.arange(len(ring))**(1-d)*c for d,c in enumerate(coefs)],axis=0)
    flucs = ring-fit
    rh = np.median(ring); rv = np.var(flucs)
    rw = np.median(widths)
    med = np.median(home); var = np.var(home)
    if fftval is not None:
        x,spect=np.array(power_spectrum(home,plot=False))
        powerfeats = []
        for val in fftval:
            index = np.argmin(np.abs(1/x*latres-val))
            powerfeats.append(np.log(spect[index]))

    loc, wloc = getwloc(flucs, latres, rx=0.3)
    l_sat, w_sat, h = w_data_extraction(loc,wloc)
    hurst = linregress(np.log10(loc)[:15],np.log10(wloc)[:15]).slope

    feats = [med,var,var/med,rh,rw,rv/rh,w_sat,hurst,rh/med,rh*med]+powerfeats
    feats = {i:d for i,d in zip(feature_names,feats)}
    return feats



def make_features(data_dic,fftvals=18,obj=5):
    all_feature_names = get_features(None,obj=obj,fftval=fftvals)
    features = {i:[] for i in all_feature_names}
    phenotypes,sn,files,rep= [],[],[],[]
    for n,(i,arr) in enumerate(tqdm(data_dic.items())):
        if 'bad' in i: continue
        if np.sum(np.isnan(arr))>0:
            arr = interpol(arr)
        features_dict = get_features(arr,obj=obj,fftval=fftvals)
        for name,feat in features_dict.items():
            features[name].append(feat)
        phenotypes.append(i[-5])
        sn.append(i.split('_')[0])
        rep.append('_'.join(i.split('_')[::2]+[i[-5]]))
        files.append(i)
    df = pd.DataFrame(features)
    df['Class'] = phenotypes
    df['Strain'] = sn
    df['Filename'] = files
    df['Replicates'] = rep
    return df,all_feature_names

def update_features(test_df,all_data,latres,obj = 5):
    homesize=400
    freq = np.fft.fftfreq(homesize)
    freq = freq[freq>0]
    start,end = max(freq),min(freq)
    fftvals = np.logspace(*np.log10(latres*1e6/np.array([np.sqrt(2)/2,end])),num=15)
    expanded_df,features = make_features(all_data,fftvals=fftvals,obj=obj)
    for col in ['Genera','Family','Drug']:
        expanded_df[col] = test_df[col].values
    return expanded_df


def save_pickle(filename,obj):
    with open(f'{filename}' + ('.pickle' if 'pickle' not in filename else ''),'wb') as handle:
        pickle.dump(obj,handle,protocol=pickle.HIGHEST_PROTOCOL)
        
def read_pickle(filename):
    with open(f'{filename}' + ('.pickle' if 'pickle' not in filename else ''),'rb') as handle:
        obj = pickle.load(handle)
    return obj

def get_features_ID(image,latres, region: str, obj=5,fftval=None,these_features=None):
    all_feats = ['Home Height','Home Var','Home CoV','Ring Height','Ring CoV','Ring w_sat','Ring Hurst','ringH/homeH','ringH*homeH','RWidth','RWidth Var','RWidth CoV']
    if fftval is not None:
        if not hasattr(fftval,'__iter__'): fftval=[fftval]
        for val in fftval:
            all_feats.append('Power-{}um'.format(str(val)[:4]))
    if image is None: return all_feats
    image = np.where(np.isnan(image),np.nanmedian(image),image)
    if region == 'full':
        home = image[300:700,300:700]
        left,right,ring_w,ring = ring_width_radial(image,all_data=True,plot=False) #ring_width(image,all_data=True)
        raise ValueError('Full region ring not fixed yet')
    elif region == 'edge':
        home = image[300:700,:400]# if obj==5 else image[:,:700]
        left, right, ring_w, ring_inds = ring_width(image, all_data=True)
    
    ring = np.array([row[r] for row,r in zip(image,ring_inds)]) #get the ring from the image
    #ring = remove_outliers(np.array([np.max(row) for row in image]),N=100,fill='median')
    all_feats.extend(['R-start','R-end','R-peak','R-heights'])
    coefs = np.polyfit(np.arange(len(ring)),ring,1)
    fit = np.sum([np.arange(len(ring))**(1-d)*c for d,c in enumerate(coefs)],axis=0)
    flucs = ring-fit
    rh = np.median(ring); rv = np.var(flucs)
    
    
    rw_med = np.median(ring_w)
    rw_var = np.var(ring_w)
    rw_cov = np.sqrt(rw_var)/rw_med
    med = np.median(home); var = np.var(home)
    
    powerfeats=[]
    if fftval is not None:
        x,spect=np.array(power_spectrum(home,plot=False))
        for val in fftval:
            index = np.argmin(np.abs(1/x*latres-val))
            powerfeats.append(np.log(spect[index]))

    loc, wloc = getwloc(flucs, latres, rx=0.3)
    l_sat, w_sat, h = w_data_extraction(loc,wloc)
    hurst = linregress(np.log10(loc)[:15],np.log10(wloc)[:15]).slope

    feats = [med,var,var/med,rh,rv/rh,w_sat,hurst,rh/med,rh*med,rw_med,rw_var,rw_cov]+powerfeats + [left,right,ring_inds,ring]#,spect[0],m]
    feats = {i:d for i,d in zip(all_feats,feats)}
    return feats

def fft_features(arr,latres,num_fftvals = 10,wavelengths=None,min_wavelength=0):
    #latres = {5.5:1.5614175326405541,50:1.7334291543917991e-01,5:1.6417910447761194,10: 8.605292006113655e-01} #latres in micron/px
    #latres = latres[obj]
    
    shape = arr.shape
    if len(shape)==1 or shape[-1] == 1:
        k = np.abs(np.fft.fftshift(np.fft.fftfreq(shape[0])))
    else:
        freqs=[]
        for n,i in enumerate(shape):
            k = np.fft.fftshift(np.fft.fftfreq(i))
            freqs.append(np.array([k for j in range(shape[len(shape)-1-n])]))
        k = np.sqrt(freqs[0]**2 + freqs[1].T**2)
    k = np.sort(k.ravel())
    if wavelengths is None:
        wavelengths = np.logspace(*np.log10(latres/k[[-1,1]]),num_fftvals,dtype=float)
        wavelengths = wavelengths[wavelengths>min_wavelength]
    
    x,     (spect_m,     spect_std     ,energy)     =np.array(power_spectrum(arr   ,plot=False,dk=3,avg=['mean','std','energy']),dtype=object)
    log_spect_m,log_spect_std = np.log([spect_m,spect_std])
    all_wavelengths = latres/x
    feats = {'power energy':energy}
    for name,stat in zip(['mean','std','cov'],[log_spect_m,log_spect_std,log_spect_std/log_spect_m]):
        if not len(arr.shape)==2 and name != 'mean': continue
        if wavelengths!='all':
            for val in wavelengths:
                index = np.argmin(np.abs(all_wavelengths-val))
                feats[f'power {name}-'+str(val)[:4]+'um'] = stat[index] #log accounted for by log_spect_...
        else:
            #text_size = 5
            #while len(feats)-1!=3*len(x) and text_size<10:
            for freq,val in zip(x,stat):
                wavelength = latres/freq if freq>0 else 'inf'
                feats.update({f'power {name}-{np.round(wavelength,2)}': val})
    return feats,(x, spect_m)
    






def get_all_fft_features(arr,latres,region='edge',wavelengths=None,plot=False,**kwargs):
    full     = arr.copy()
    homeland = arr[300:700,300:700] if region=='full' else arr[300:700,:400]
    #plot_contour(homeland)
    left,right,_,ring    = ring_width_radial(arr,all_data=True) if region=='full' else ring_width(arr,all_data=True)
    
    ring = np.array([row[r] for row,r in zip(arr,ring)])

    left_flucs = sym_reg(left,degree=2,normal=True)
    right_flucs = sym_reg(right,degree=2,normal=True)

    feats = {}
    for region,data in zip(['Full','Home','RingL','RingP','RingR'],[full,homeland,left_flucs,ring,right_flucs]):
        these_feats,(x,spect) = fft_features(data,latres,wavelengths=wavelengths)
        feats.update({region+i:d for i,d in these_feats.items()})
        if plot:
            plt.figure(region)
            plt.loglog(x,spect,**kwargs)
    return feats
    
    
    
def subtract_poly(line,deg=2,x_vals=None,just_fit=False):
    if not all([hasattr(l,'__iter__') for l in line]):
        lines = np.array([line])
    else:
        lines = np.array(line)
    fits=[]
    for line in lines:
        if x_vals is None:
            x = np.arange(-len(line)/2,len(line)/2,1)+0.5
        else:
            x = np.array(x_vals)
        coefs = np.polyfit(x,line,deg)
        fit = np.sum([x**(deg-d)*c for d,c in enumerate(coefs)],axis=0)
        fits.append(fit)
    fits = np.array(fits,dtype=object)
    if just_fit: return fits
    return lines - fits
    

    
    
    
    
    
def plot_df(df, split=None, sort=None, close=True, data_column='Data-array', background=None, unique_box=True, **kwargs):
    """
    Plots data from a DataFrame with interactive features.

    This function plots data from a DataFrame, allowing for interactive features such as clicking on axes to display detailed information.
    The data can be split into multiple subplots based on specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    split (list or str, optional): The column(s) to split the data into subplots. If None, no splitting is done. Default is None.
    sort  (list or str, optional): Additional column(s) to further split the data into subplots. If None, no additional splitting is done. Default is None.
    force (bool, optional): Whether to force the plotting even if some data is missing. Default is True.
    **kwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
    dict: A dictionary of axes objects for the created plots.
    pd.DataFrame: The DataFrame with the plotted data.
    """
    for col in ['Folder','FileBase','Replicate']:
        if col not in df.columns:
            df[col] = [str(i) for i in np.arange(len(df))]
    
    if split is None:
        split = []
    elif isinstance(split, str):
        split = [split]
    if sort is None:
        sort = []
    elif isinstance(sort, str):
        sort = [sort]
    ascending = [not col.startswith('-') for col in sort]
    sort = [col.lstrip('-') for col in sort]
    df['Unique Identifier'] = [i for i in range(df.shape[0])]
    giant = split + sort
    for n, spl in enumerate(giant):
        if '+' in spl:
            only = spl.split('+')
            keep = only.pop(0)
            if keep == 'Strain ID':
                only = ['0' * (4 - len(i)) + i for i in only]
            col_type = df[keep].map(type).iloc[0]
            if not df[keep].map(type).nunique() == 1:
                df.fillna({keep:'None'},inplace=True)
                col_type = df[keep].map(type).iloc[0]
                if not df[keep].map(type).nunique() == 1:
                    raise TypeError(f'Not all values in {keep} are the same type.')
            mask = [any([col_type(i) == col_type(j) for j in only]) for i in df[keep]]

        elif '-' in spl:
            rems = spl.split('-')
            keep = rems.pop(0)
            if keep == 'Strain ID':
                rems = ['0' * (4 - len(i)) + i for i in rems]
            mask = [all([str(i) != str(j) for j in rems]) for i in df[keep]]
        else:
            continue
        df = df.loc[mask]
        if n >= len(split):
            sort[n - len(split)] = keep
        else:
            split[n] = keep

    
    df_sorted = df.sort_values(by=sort, ascending=ascending)
    if 'Notes' in df_sorted.columns: df_sorted.fillna({'Notes':'None'}, inplace=True)
    true_split = [np.unique(df_sorted[s]) for s in split]
    split_names = [[s for i in true_split[n]] for n, s in enumerate(split)]
    dic = {data_column+'--'+str(vals): {'-'.join([fold, str(base), str(int(rep))]): d for base, rep, fold, d, *test in df_sorted[['FileBase', 'Replicate', 'Folder', data_column, *s]].values if all(test == np.array(vals))} for s, vals in zip(itertools.product(*split_names), itertools.product(*true_split))}
    axes = plot_all(dic, close=close, fullname=False, titles=False, **kwargs)
    global highlighted_rect, last_clicked_ax, on_ax
    highlighted_rect = None
    last_clicked_ax = None
    on_ax = None

    
    def on_click(event):
        global highlighted_rect, last_clicked_ax, on_ax
        for fig, axs in axes.items():
            check = event.inaxes == axs.ravel()
            if any(check):
                ii, jj = define_subplot_size(axs.size)
                for n, ax in enumerate(axs.ravel()):
                    if ax != event.inaxes or (ax == last_clicked_ax and on_ax == ax):
                        continue
                    this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in df[['Folder', 'FileBase', 'Replicate']].values]]
                    i, j = np.unravel_index(n, (ii, jj))
                    #print(f'\n\nAxis clicked! (Figure {fig.name}, Col {j + 1}, Row {ii - i})')
                    print('\n\n')
                    sorted_columns = sorted([col for col in this_df.columns if col != "Notes"]) + (["Notes"] if "Notes" in this_df.columns else [])
                    sorted_df = this_df[sorted_columns]
                    for name, val in zip(sorted_df.columns, sorted_df.iloc[0]):
                        if 'array' in name or name.startswith('f'):
                            continue
                        print(f'{name}: {val}')
                    # Highlight the clicked image with a rectangle
                    bbox = ax.get_position()
                    dr = 0.1
                    rect = Rectangle((bbox.x0 - dr * bbox.width, bbox.y0 - dr * bbox.height), 
                                     bbox.width * (1+2*dr), bbox.height * (1+2*dr),
                                     transform=fig.transFigure, color='gray', alpha=0.3, zorder=-1)
                    
                    if highlighted_rect is not None:
                        highlighted_rect['fig'].patches.remove(highlighted_rect['rect'])
                    fig.patches.append(rect)
                    highlighted_rect = {'fig': fig, 'rect': rect}
                    
                    if last_clicked_ax == ax and not on_ax == ax:
                        print('Axis double clicked!')
                        plot_selected_data(df, this_df['Unique Identifier'].values[0], data_column=data_column)
                        on_ax == ax
                    else:
                        print('Axis single clicked!')
                        last_clicked_ax = ax
            fig.canvas.draw_idle()  # Update the figure to reflect the change

    def on_button_click(event):
        global unid_axes
        for fig, axs in axes.items():
            ax = unid_axes[fig]
            check = event.inaxes == ax
            if not check: continue
            print('Check cleared!')
            for ax in axs.ravel():
                this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in df[['Folder', 'FileBase', 'Replicate']].values]]
                unique_id = this_df['Unique Identifier'].values[0]
                text = ax.texts
                if text:
                    for t in text:
                        t.remove()
                else:
                    ax.text(0, 1, str(unique_id), transform=ax.transAxes, color='black', fontsize=12, ha='left', va='top')
            fig.canvas.draw_idle()  # Update the figure to reflect the change

    # Connect the click event to the handler
    global unid_axes
    unid_axes = {}
    if background and background in df.columns:
        unique_vals = np.unique(df[background].values)
        background_cols = {val:plt.get_cmap('tab10')(n) for n,val in enumerate(unique_vals)}

    for fig,axs in axes.items():
        fig.subplots_adjust(bottom=0.1)
        if background is not None:
            for ax in axs:
                if ax.get_label() == '<colorbar>':
                    continue
                this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in df[['Folder', 'FileBase', 'Replicate']].values]]
                val = this_df[background].values[0]
                color = background_cols[val]
                bbox = ax.get_position()
                dx = 0.15 * bbox.width
                dy = 0.15 * bbox.height
                rect = Rectangle((bbox.x0-dx, bbox.y0-dy), bbox.width+2*dx, bbox.height+2*dy, transform=fig.transFigure, facecolor=color, alpha=0.3, clip_on=False, zorder=-2)
                fig.patches.append(rect)
            for n, (val, color) in enumerate(background_cols.items()):
                x_position = 0 + .125 * (n // 2)  # Adjust x_position for each value
                y_position = .04 - 0.04 * (n % 2)
                rect = Rectangle((x_position, y_position), 0.1, 0.03, transform=fig.transFigure, facecolor=color, alpha=0.2, clip_on=False, zorder=-1)
                fig.patches.append(rect)
                fig.text(x_position + 0.05, y_position + 0.015, str(val), ha='center', va='center', fontsize=12, color='black')
        
        
        
        if unique_box:
            ax = plt.axes([0.4, 0.025, 0.2, 0.05])
            unid_axes[fig] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('lightgray')
            ax.text(0.5,0.5,'Unique ID', fontsize=12, ha='center', va='center')
            fig.canvas.mpl_connect('button_press_event', on_button_click)
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        
    
    plt.pause(0.001)
    return axes, df
    
    
    
def plot_selected_data(df, identifiers, data_column='Data-array', color_lims=[None, None], threeD=False, **kwargs):
    """
    Plots specific data arrays from a DataFrame based on unique identifiers.

    This function plots specific data arrays from a DataFrame based on the provided unique identifiers.
    It can plot the data in 2D or 3D and allows for setting color limits for the plots.
    Additionally, it provides buttons to toggle between 2D and 3D plots, close the figure, and open raw data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data arrays to plot. Must include a "Unique Identifier" column.
    identifiers (list or str): The unique identifier(s) of the data arrays to plot. Can be a single identifier or a list of identifiers.
    color_lims (list, optional): The color limits for the plots. Default is [None, None].
    threeD (bool, optional): Whether to plot the data in 3D. Default is False.

    Returns:
    pd.DataFrame: The metadata of the plotted data arrays.
    """
    global dimensions
    dimensions = threeD 
    if not hasattr(identifiers, '__iter__') and not isinstance(identifiers, str):
        identifiers = [identifiers]
    if 'Unique Identifier' not in df.columns:
        raise KeyError('Unique Identifiers not found in dataframe. Use plot_df function to create them and identify which data you wish to plot.')
    
    response = False
    meta = pd.DataFrame()
    figs = []

    for i in identifiers:
        i = int(i)
        mask = df['Unique Identifier'] == i
        this_meta = df.loc[mask].copy()
        data = this_meta[data_column].values[0]
        resolution = this_meta['Lateral Resolution'].values[0] * 1e6 # in microns
        if dimensions:
            if len(identifiers) > 10 and not response:
                response = input('You are requesting to plot more than 10 images in 3D. Are you sure you wish to continue? (y,[n])').lower().strip() == 'y'
            else:
                response = True
            if response:
                fig = plot_3d(data, lims=color_lims, lat_res=resolution).figure
            else:
                break
        else:
            fig = plot_contour(data, vlims=color_lims,xy_scale=resolution,xy_units='um',**kwargs).ax.figure
        
        fig.identifier = i
        figs.append(fig)
    global cids
    cids = []
    def toggle_plot(event):
        global dimensions, cids
        dimensions = not dimensions
        main_ax = event.inaxes
        for fig, cid in cids:
            axs = np.array(fig.get_axes())
            mask = axs == main_ax
            if not any(mask):
                continue
            if not len(main_ax.texts)>0:
                break 

            button_text = event.inaxes.texts[0].get_text().lower()
            toggle = button_text == '2d/3d'
            close = button_text == 'close'
            raw = button_text == 'raw'
            replace_ax = None
            if close:
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)
                plt.pause(0.01)
                break
            elif raw:
                mask = df['Unique Identifier'] == fig.identifier
                this_meta = df.loc[mask].copy()
                raw_data = this_meta['Raw-array'].values[0]
                resolution=this_meta['Lateral Resolution'].values[0] * 1e6
                if raw_data is not None:
                    max_h = np.nanmax(raw_data.ravel())
                    plt.figure()
                    plot_3d(raw_data, zlims=[0,1.1*max_h],consistent_lims=False,lat_res=resolution*1e6)
                    plt.title('Raw Data')
                    plt.pause(0.01)
                else:
                    print('No raw data provided!')
                break
            elif not toggle:
                break
                
            for ax in axs[np.logical_not(mask)]:
                if len(ax.texts) > 0:
                    continue
                if isinstance(ax, plt.Axes) and not ax.get_label() == '<colorbar>':
                    replace_ax = ax
                else:
                    ax.remove()
            mask = df['Unique Identifier'] == fig.identifier
            this_meta = df.loc[mask].copy()
            data = this_meta[data_column].values[0]
            resolution = this_meta['Lateral Resolution'].values[0]*1e6
            if dimensions:
                plot_3d(data, ax=replace_ax, new_fig=False, lims=color_lims, lat_res=resolution)
            else:
                plot_contour(data, ax=replace_ax, new_fig=False, vlims=color_lims, xy_scale=resolution, xy_units='um')
            fig.canvas.draw_idle()
    for fig in figs:
        plt.figure(fig.number)
        plt.subplots_adjust(top=0.9)
        ax_button = plt.axes([0.3, 0.91, 0.2, 0.075])
        ax_button.text(0.5, 0.5, '2D/3D', transform=ax_button.transAxes, ha='center', va='center')
        ax_button.set_xticks([])
        ax_button.set_yticks([])
        for spine in ax_button.spines.values():
            spine.set_visible(False)
        ax_button.set_facecolor('lightgray')
        
        cl_button = plt.axes([0.55, 0.91, 0.2, 0.075])
        cl_button.text(0.5, 0.5, 'Close', transform=cl_button.transAxes, ha='center', va='center')
        cl_button.set_xticks([])
        cl_button.set_yticks([])
        for spine in cl_button.spines.values():
            spine.set_visible(False)
        cl_button.set_facecolor('lightcoral')
        
        raw_button = plt.axes([0.05, 0.91, 0.2, 0.075])
        raw_button.text(0.5, 0.5, 'Raw', transform=raw_button.transAxes, ha='center', va='center')
        raw_button.set_xticks([])
        raw_button.set_yticks([])
        for spine in raw_button.spines.values():
            spine.set_visible(False)
        raw_button.set_facecolor('lightblue')
        
        cid = fig.canvas.mpl_connect('button_press_event', toggle_plot)
        cids.append((fig, cid))
    plt.pause(0.1)
    this_meta.drop(columns=['Data-array'] + [i for i in this_meta.columns if i.startswith('f')], errors='ignore', inplace=True)
    meta = pd.concat([meta, this_meta])
    
    return meta, figs


def get_specific_power_wavelengths(data,latres,wavelengths=None,dr=3,avg=['mean']):
    window2d = np.ones_like(data)
    twoD = False
    if len(data.shape)==2:
        twoD = True
        window2d = np.sqrt(np.abs(np.outer(np.hanning(data.shape[0]),np.hanning(data.shape[1]))))
    data =np.where(np.isnan(data),np.nanmean(data),data)*window2d
    ft = fft.fftn(data)
    shift = fft.fftshift(ft)
    s = np.abs(shift)
    power = np.abs(s)

    # Get image parameters
    
    freqr = np.fft.fftshift(np.fft.fftfreq(power.shape[0],d=latres))
    dk = np.diff(freqr)[0] * dr
    if twoD:
        freqc = np.fft.fftshift(np.fft.fftfreq(power.shape[1],d=latres))
        # Find radial distances
        [kx, ky] = np.meshgrid(freqc,freqr)
        k = np.sqrt(np.square(kx) + np.square(ky))
    else:
        k = np.abs(freqr)
    
    wavs = 1/k 
    wavs = wavs[np.isfinite(wavs)]
    x=[]
    rad = {'mean':[],'std':[]} 
    
    if wavelengths is None:
        wavelengths = np.logspace(np.log10(wavs.min()),np.log10(wavs.max()),10)[1:]
    elif isinstance(wavelengths, int):
        wavelengths = np.logspace(np.log10(wavs.min()),np.log10(wavs.max()),wavelengths)[1:]
    elif wavelengths == 'all':
        wavelengths = np.logspace(np.log10(wavs.min()),np.log10(wavs.max()),num=1000)[1:]

    for w in wavelengths:
        mask = (k>1/w-dk/2)&(k<=1/w+dk/2)
        x.append(1/w)
        dat = power[mask]
        
        rad['mean'].append(np.nanmean(dat))
        if twoD:
            rad['std'].append(np.nanstd(dat))

    rad = {i:np.array(d) for i,d in rad.items()}
    x = np.array(x)
    if False:
        plt.close('all')
        if twoD:
            plot_contour(data)
            plt.figure()
            power_spectrum(data,resolution=latres,dk=3,log=True)
        else:
            plt.figure()
            plt.plot(data)
            plt.figure()
            plt.loglog(k[k>=0],power[k>=0])
        
        ax = plt.gca()
        ax.scatter(x*latres,rad['mean'])
        plt.show()
    return rad,wavelengths

def new_get_all_fft_features(arr,latres,region='edge',wavelengths=None,plot=False,**kwargs):
    full     = arr.copy()
    homeland = arr[300:700,300:700] if region=='full' else arr[300:700,:400]
    #plot_contour(homeland)
    left,right,_,ring    = ring_width_radial(arr,all_data=True) if region=='full' else ring_width(arr,all_data=True)
    if region == 'edge':
        ring = np.array([row[r] for row,r in zip(arr,ring)])
    left_flucs = sym_reg(left,degree=2,normal=True)
    right_flucs = sym_reg(right,degree=2,normal=True)

    feats = {}
    for region,data in zip(['Full','Home','RingL','RingP','RingR'],[full,homeland,left_flucs,ring,right_flucs]):
        vals,lamb = get_specific_power_wavelengths(data,latres,wavelengths=wavelengths)
        these_feats = {}
        for metric,values in vals.items():
            for w,val in zip(lamb,values):
                these_feats[f'power {metric}-{w:.1f}um'] = np.log(val)
                #print(w,val)
                


        feats.update({region+i:d for i,d in these_feats.items()})

    return feats
