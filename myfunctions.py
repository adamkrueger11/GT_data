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
from matplotlib.patches import Ellipse, Rectangle, Circle, Path, PathPatch
import matplotlib.transforms as transforms
import h5py
import pandas as pd
import re
import itertools
import warnings
import traceback
import json
from matplotlib.widgets import Button

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.color'] = 'black'
plt.rcParams['lines.color'] = 'black'


def plot_string_histogram(strings):
    """
    Plots a histogram of a list of strings.

    Parameters:
    strings (list): The list of strings to plot.
    """
    import matplotlib.pyplot as plt

    # Count the occurrences of each string
    labels, counts = np.unique(strings, return_counts=True)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel('Strings')
    plt.ylabel('Counts')
    plt.title('Histogram of Strings')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.pause(0.01)


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


def convert_data(file_name, resolution=False, remove=True):
    '''Convert datx file to numpy array
        resolution: if True, return tuple of data and lateral resolution
        get: list of attributes to receive. 'Heights' and 'Intensity' are the options. can be both. if both, will provide dictionary. '''
    values = {}
    if '.' not in file_name: file_name += '.datx'
    full_dict = datx2py(file_name)
    try:
        lat_res = full_dict['Measurement']['Attributes']['attrs']['Data Context.Lateral Resolution:Value']
        data = np.array(full_dict['Measurement']['Surface']['vals'])
    except:
        data = np.array(
            full_dict['Processed Data: ']['PM-Micro']['AP_DS:Regions']['SequenceMatrix1']['Surface']['vals'])
    if remove:
        data = np.where(data > 10 ** 100, np.nan, data)
        median = np.nanmedian(data)
        data /= median

        new = np.array([[i if abs(i) < 1000 else np.nan for i in j] for j in data]) * median

        cutoff = np.nanmean(new) - 5 * np.nanstd(new)
        new = np.where(new > cutoff, new, np.nan)
    else:
        new = np.where(np.log10(np.abs(data)) > 300, np.nan, data)
    values['Heights'] = new
    values['Pitch'] = full_dict['Measurement']['Attributes']['attrs']['Data Context.Data Attributes.Stage Pitch:Value']
    values['Roll'] = full_dict['Measurement']['Attributes']['attrs']['Data Context.Data Attributes.Stage Roll:Value']
    try:
        values['Intensity'] = np.array(full_dict['Measurement']['Intensity']['vals'])
    except:
        values['Intensity'] = None
    values['Resolution'] = lat_res
    plt.pause(0.01)
    return values


def timeit(func, data, reps=1):
    t1 = time.time()
    for i in range(reps):
        func(data)
    return (time.time() - t1) / reps


def get_xy(img):
    if img.shape[1] == 1:
        X = np.arange(len(img)).reshape(len(img), 1)
        Y = np.array([])
        return X, Y
    else:
        X = np.arange(len(img)).reshape(len(img), 1)
        Y = np.arange(len(img[0])).reshape(1, len(img[0]))
        return X, Y


def plot_3d(data, new_fig=True, axis=True, ax=None, lims=[None, None], zlims=[None, None], consistent_lims=True,
            lat_res=1, view=(13.601693557484248, 7.066860584210744), count=50, cmap='Spectral_r'):
    X, Y = [i * lat_res for i in get_xy(data)]

    if new_fig and ax is None:
        plt.figure()
        ax = plt.axes(projection='3d')
    elif ax is not None:
        plt.sca(ax)
        new_ax = plt.axes(projection='3d')
        ax.remove()
        del ax
        ax = new_ax
    else:
        ax = plt.gca()
    if consistent_lims:
        zlims = lims
    p = ax.plot_surface(X, Y, data, cmap=cmap, vmin=lims[0], vmax=lims[1], rcount=count, ccount=count)
    ax.view_init(view[0], view[1])
    zlims[0] = np.nanmin(data) if zlims[0] is None else zlims[0]
    zlims[1] = np.nanmax(data) if zlims[1] is None else zlims[1]
    ax.set_zlim(zlims)
    ax.set_xlabel('um')
    if not axis:
        ax.axis('off')
    plt.pause(0.01)
    return ax


def define_subplot_size(num):
    area = np.array([[i * j if j <= i and i * j >= num else 1e7 for i in range(1, num + 1)] for j in range(1, num + 1)])
    perim = np.array([[abs(i - j) if j <= i else 1e7 for i in range(1, num + 1)] for j in range(1, num + 1)])
    dim = [1e7, 0]
    shift = 0
    while np.ptp(dim) > np.sqrt(num):
        a_dim = np.array(np.where((area - num) <= np.min(area - num) + shift)).T
        p_dim = np.argmin([perim[tuple(a)] for a in a_dim])
        dim = a_dim[p_dim]
        shift += 1
    return np.array(dim) + np.ones_like(dim)


def plot_contour(data, location=111, xy_scale=1, xy_units=None, switch_xy=False, new_fig=True, clear=False,
                 cmap='coolwarm', vlims=[None, None], cbar=True, cbar_lims=[None, None], cbar_label='', axis=True,
                 alpha=1, ax=None, shape=None):
    if new_fig and ax is None:
        plt.figure()
    if cmap == None:
        cmap = cm.coolwarm
    if type(location) == str:
        num, this_one = np.int_(location.split('_'))
        if not isinstance(shape, (tuple, list, np.ndarray)):
            loc = define_subplot_size(int(num))
        else:
            loc = tuple(list(shape))
        if switch_xy:  # switch x and y
            this_one = ((this_one - 1) % loc[0]) * loc[1] + int((this_one - 1) / loc[0]) + 1
        location = (loc[0], loc[1], this_one)
    if clear:
        axis = False
        cbar = False
    if ax is None:
        try:
            ax = plt.subplot(*location)
        except:
            ax = plt.subplot(location)
    else:
        plt.sca(ax)
        new_ax = plt.axes()
        ax.figure.delaxes(ax)
        ax = new_ax
    if location == 111 and axis == False:
        fig = plt.gcf()
        sizes = np.shape(data)
        fig.set_size_inches(10. * sizes[1] / sizes[0], 10., forward=False)
    plt.imshow(data, cmap=cmap, vmin=vlims[0], vmax=vlims[1], alpha=alpha, interpolation=None,
               extent=np.array([0, data.shape[1], 0, data.shape[0]]) * xy_scale)

    if not axis:
        ax.axis('off')
    axis_label = 'Pixels' if xy_scale == 1 else xy_units if xy_units is not None else 'No units provided'
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    if cbar or len(cbar_label) > 0:
        if any([i is not None for i in cbar_lims]):
            cbar_lims = [0 if i is None else i for i in cbar_lims]
            colormap = plt.get_cmap(cmap)
            norm = Normalize(vmin=vlims[0], vmax=vlims[1])

            new_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                'truncated_' + cmap, colormap(norm(np.arange(cbar_lims[0], cbar_lims[1]))), N=100)
            norm2 = Normalize(vmin=cbar_lims[0], vmax=cbar_lims[1])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=new_cmap), label=cbar_label)
        else:
            cbar = plt.colorbar(ax=ax, label=cbar_label)
        plt.pause(0.01)
        return cbar
    plt.pause(0.01)


def plot_all(dic, close=False, vlims=[None, None], new_fig=True, individual=False, figsize=None, dpi=None,
             fullname=True, titles=True, **kwargs):
    """
    Plots all data arrays from a dictionary with interactive features.

    This function plots all data arrays from a dictionary, allowing for interactive features such as clicking on axes to display detailed information.
    The data can be split into multiple subplots based on specified columns.

    Parameters:
    dic (dict): The dictionary containing the data arrays to plot.
    close (bool, optional): Whether to close all existing plots before creating new ones. Default is False.
    vlims (list, optional): The color limits for the plots. Default is [None, None].
    new_fig (bool, optional): Whether to create a new figure for each plot. Default is True.
    individual (bool, optional): Whether to plot each data array individually. Default is False.
    figsize (tuple, optional): The size of the figure. Default is None.
    dpi (int, optional): The resolution of the figure. Default is None.
    fullname (bool, optional): Whether to use the full name for the plot titles. Default is True.
    titles (bool, optional): Whether to display titles for the plots. Default is True.

    Returns:
    dict: A dictionary of axes objects for the created plots.
    """
    if all(isinstance(value, dict) for value in dic.values()):
        full_dic = dic.copy()
        sup_title = True
    else:
        full_dic = {'All data': dic}
        sup_title = False
    if close:
        plt.close('all')
    axes = {}
    cbar_label = kwargs.pop('cbar_label', '')
    for name, dic in full_dic.items():
        if len(dic) == 0:
            continue
        if new_fig:
            fig = plt.figure(num=str(name), figsize=figsize, dpi=dpi)
            fig.name = str(name)
        if individual:
            for i, d in dic.items():
                if len(d.shape) > 1:
                    plot_contour(d, vlims=vlims)
                else:
                    plt.plot(d)
                    plt.ylims(vlims)
                if titles:
                    plt.title(i)
                plt.gca().set_gid(i)
        else:
            colorbar_bool = any([i is not None for i in vlims])
            cbar = kwargs.pop('cbar', True)
            for n, (i, d) in enumerate(dic.items(), 1):
                if d is None:
                    d = np.zeros((10, 10))
                    continue
                elif len(d.shape) > 1:
                    plot_contour(d, vlims=vlims, new_fig=False, location='{}_{}'.format(len(dic), n), axis=False,
                                 cbar=not colorbar_bool and cbar, cbar_label=cbar_label if not colorbar_bool else '',
                                 **kwargs)
                else:
                    plt.plot(d)
                if titles:
                    plt.title('_'.join(i.split('_')[::2]) if not fullname else i)
                plt.gca().set_gid(i)
            if colorbar_bool:
                fig = plt.gcf()
                mappable = matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(vmin=vlims[0], vmax=vlims[1], clip=False), cmap='coolwarm')
                cbar = fig.colorbar(mappable, ax=fig.get_axes(), fraction=0.046, pad=0.04, label=cbar_label)
                fig.tight_layout(rect=[0, 0, 0.8, 0.95])  # Adjust layout to make space for colorbar and suptitle
        if sup_title:
            plt.gcf().suptitle(name)
        axes[fig] = np.array(plt.gcf().axes)[::2 if not colorbar_bool else 1]
    return axes


def plot(image, ax_on_off='off'):
    fig, ax = plt.subplots()
    plt.imshow(image, cmap='gray')
    ax.axis(ax_on_off)
    plt.pause(0.01)


def write_excel(dic, path):
    if type(dic) != dict:
        dic = {'Sheet1': dic}
    writer = pd.ExcelWriter(path, date_format=None, mode='w')
    for df_name, df in dic.items():
        df.to_excel(writer, sheet_name=df_name)
    writer.close()


def get_corners(arr, corners=['tl', 'tr', 'bl', 'br'], minR_factor=1, reverse=False):
    X = np.array([np.arange(arr.shape[1]) for i in range(arr.shape[0])]) - arr.shape[1] / 2
    Y = np.array([np.arange(arr.shape[0]) for i in range(arr.shape[1])]).T - arr.shape[0] / 2
    bigR = np.sqrt(arr.shape[0] ** 2 + arr.shape[1] ** 2) / 2
    minR = min(arr.shape) / 2 * minR_factor
    R = minR + (bigR - minR) / 4

    rev = -1 if reverse else 1
    no = np.bool_(np.zeros(X.shape))
    border = np.where(((X ** 2 + Y ** 2) * rev > R ** 2 * rev) & (
            (((X > 0) & (Y > 0)) if 'br' in corners else no) |
            (((X > 0) & (Y < 0)) if 'tr' in corners else no) |
            (((X < 0) & (Y < 0)) if 'tl' in corners else no) |
            (((X < 0) & (Y > 0)) if 'bl' in corners else no)), arr, np.nan)
    return border


def get_elbow(data, x=None, xy=True, elbow_point=True):
    y = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    x = np.arange(0, 1 + 10 ** (-5), 1 / len(data[:-1]))
    flip = False
    if y[0] > y[-1]:
        x = np.flip(x)
        flip = True
    d = x - y
    if np.nanmax(d) != 0:  # and y[1]<x[1] * 0.9:
        elbow = np.argmax(d)
    # =============================================================================
    #         dd= np.where(abs(d-np.nanmax(d))/np.nanmax(d)<0.01,d,np.nan)
    #         elbow = d.tolist().index(np.nanmax(d))
    #         elbow = dd.tolist().index(np.nanmin(dd))
    # =============================================================================
    else:
        elbow = 0
    elbow = np.sum(y > y[elbow] * 0.99)  # (len(y)-1)/len(y))
    if not flip:
        elbow = len(y) - elbow

    if xy and not elbow_point:
        return x, y, d
    elif elbow_point and not xy:
        return d, elbow
    elif xy and elbow_point:
        return x, y, d, elbow
    else:
        return d


def remove_outliers(data, N=10, fill=None):
    global small_counts, x, h

    counts, bins = np.histogram(data[~np.isnan(data)], bins=N)
    for n, c in enumerate(counts):
        if c < 10 ** (-3) * data.size and c != 0:
            data = np.where(data < bins[n + 1], np.nan, data)
        elif c != 0:
            break
    for n2, c in enumerate(np.flip(counts[n + 1:])):
        if c < 10 ** (-3) * data.size and c != 0:
            data = np.where(data > bins[-(n2 + 2)], np.nan, data)
        elif c != 0:
            break
    if fill is None:
        pass
    elif fill == 'mean':
        data = np.where(np.isnan(data), np.nanmean(data), data)
    elif fill == 'median':
        data = np.where(np.isnan(data), np.nanmedian(data), data)
    return data


def watershed_alg(image, r):
    distance = ndimage.distance_transform_edt(image)
    dist = np.where(distance > np.sqrt(distance.size), np.sqrt(distance.size), distance)
    distance = np.where(distance < 0, 0, dist)
    distance = ndimage.gaussian_filter(distance, sigma=1)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((r, r)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = skimage.segmentation.watershed(-distance, markers, mask=image)
    return labels


def interpol(img):
    x, y = get_lib_x(img)
    z = img.flatten()
    X, Y, Z = np.array([x, y, z]).T[~np.isnan(z)].T
    x, y = np.array([x, y]).T[np.isnan(z)].T

    new = scipy.interpolate.griddata((X, Y), Z, (x, y), method='cubic')

    z[np.isnan(z)] = new

    final = z.reshape(img.shape)
    return final


def choose(n, r):  # binomial
    return int(math.factorial(n) / ((math.factorial(r) * math.factorial(n - r))))


def get_lib_x(img):
    x = []
    XX, YY = get_xy(img)
    if YY.size == 0:
        x.append(XX.flatten())
        return np.array(x, dtype=float)

    X = np.outer(XX, np.ones(len(YY.T))).flatten()
    Y = np.outer(YY.T, np.ones(len(XX))).T.flatten()

    x.append(X)
    x.append(Y)
    return np.array(x, dtype=float)


def create_library(x, degree):
    num_of_vars = len(x)
    num_of_terms = choose(num_of_vars + degree, degree)
    theta = [[] for i in range(num_of_terms)]  # this will be the library
    data_labels = ['{}'.format(i) for i in range(num_of_vars)]  # this is the name of all fo the variables (0,1,2,3...)

    theta[0] = np.ones(len(x[0]))  # constant term

    terms = ['' for i in range(num_of_terms)];  # Will show the combination for each term

    i = 1;
    d = 0;
    prev_combos = 0
    while i < num_of_terms:
        for j in range(prev_combos, choose(d + num_of_vars - 1,
                                           d) + prev_combos):  # goes through combinations of same length (xx,xy,zz..length 2)
            if j == 0:
                starting_var = 0
            else:
                starting_var = data_labels.index(
                    terms[j][-1])  # index makes sure we start with the 'next' variable in the combo

            for k in range(starting_var, num_of_vars):  # adds each new variable on to preexisting combination
                terms[i] = terms[j] + data_labels[k]
                theta[i] = x[k] * theta[j]
                i += 1
            # print()
        prev_combos += choose(d + num_of_vars - 1, d)
        d += 1
    theta = np.array(theta)
    return theta, terms


def residual(library, coef, data):
    dif = np.matmul(library.T, coef) - data
    res = np.linalg.norm(dif[~np.isnan(dif)]) / num_not_nans
    return res


def sym_reg(image, degree, X=[], full_image=None, include_z=False, res_terms=False, normal=True, all_fits=False,
            surf=False, just_fit=False, just_res=False, coefs=False, terms_in_fit=False, secondary_trials=False):
    orig_shape = image.shape
    global num_of_vars, num_of_terms, num_not_nans
    num_of_vars = len(image.shape)  # number of variables to have data for
    if num_of_vars == 1: image = image.reshape(len(image), 1)
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
    not_nans = ~np.isnan(data)
    # x = x.T[not_nans].T; data = data[not_nans]
    if include_z:
        x = np.concatenate((x, data.reshape(1, len(data))))
    if num_of_vars == 1:
        x = x.reshape(1, np.array(x).size)
    global normalizations, best
    normalizations = np.ones(num_of_vars)
    for var in range(num_of_vars):
        normalizations[var] *= abs((np.nanmean(np.abs(x[var]))))

        x[var] /= normalizations[var]
        # data.T[var]/=normalizations[var]
    theta_full, terms = create_library(x, degree)
    theta = theta_full.T[not_nans].T;
    data = data[not_nans]
    denorms = np.array([np.prod(normalizations[np.int_(list(i))]) for i in terms])
    # finds initial regression coefficients

    xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta, theta.T)), theta), data)
    xi_best = xi.copy();
    terms_best = terms.copy()

    xi_final = {x: [0] for x in
                terms};  # initialize dictionary that will show final coefficient matrix with term labels
    xi_real = np.zeros(num_of_terms)
    times = [];
    # for var in range(num_of_vars):
    global deleted
    skip, deleted = [], []
    magnitude = np.floor(np.log10(theta.shape[1]) - 4)
    if magnitude < 0: magnitude = 0

    step_size = int(10 ** magnitude)
    theta_var = theta[:, ::step_size]
    data = data[::step_size]
    terms_var = terms.copy()
    res_initial = residual(theta_var, xi, data)
    res = res_initial;
    res_best = res_initial
    global residues, all_terms, all_xi
    residues = [res_best];
    all_terms = [terms.copy()];
    all_xi = [xi]  # [res_initial]
    if not normal:
        for term in range(num_of_terms - 1):  # Delete all except one term in regression
            trial_residues = [];
            for i in range(
                    num_of_terms - term):  # for each term to delete, run through all possible ones to minimize residual
                trial = np.delete(theta_var, i, 0)  # trial library
                if np.linalg.det(np.matmul(trial, trial.T)) == 0:
                    trial_residues.append(res + 10 * res)
                    continue
                time1 = time.time()
                xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(trial, trial.T)), trial),
                               data)  # trial coefficient matrix
                time2 = time.time()
                times.append(time2 - time1)
                res = residual(trial, xi, data)
                if term == 0 and i in skip:
                    # print(skip)
                    res *= 1000
                trial_residues.append(res)
            # plt.semilogy(trial_residues)
            # plt.title('Fails: {}'.format(fails))
            # plt.close()
            term_to_del = trial_residues.index(
                min(trial_residues))  # find the best term to delete from the possibilities above
            deleted.append(terms_var[term_to_del])
            if term == 0 and secondary_trials == True: skip.append(term_to_del)
            # delete term that mimized residual
            terms_var.pop(term_to_del)

            theta_var = np.delete(theta_var, term_to_del, 0)
            # find coefficents and residual again
            xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta_var, theta_var.T)), theta_var), data)
            residues.append(min(trial_residues))
            # save the coefficient matrix with the best residual

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

        # else: break
        global d
        d, elbow = get_elbow(residues, xy=False, elbow_point=True)
        terms_best = all_terms[elbow]
        xi_best = all_xi[elbow]
    else:
        terms_best = all_terms[0]
        xi_best = all_xi[0]
        # undo normalization of best-fit term coefficients
    for sparse_term in terms_best:
        fix_norm = 1

        for term in sparse_term:
            fix_norm *= normalizations[int(term)]
        xi_final[sparse_term] = [xi_best[terms_best.index(sparse_term)] / fix_norm]
        xi_real[terms.index(sparse_term)] = xi_best[terms_best.index(sparse_term)] / fix_norm

    if full_image.shape == image.shape:
        theta = theta_full * denorms.reshape(-1, 1)
    else:
        x = get_lib_x(full_image);
        theta, _ = create_library(x, degree)
    # X,Y = get_xy(full_image)
    surface = theta.T @ xi_real;
    surface = surface.reshape(orig_shape)
    if all_fits:
        fits = []
        for one_xi, one_terms in zip(all_xi, all_terms):
            for sparse_term in one_terms:
                fix_norm = 1

                for term in sparse_term:
                    fix_norm *= normalizations[int(term)]
                xi_final[sparse_term] = [one_xi[one_terms.index(sparse_term)] / fix_norm]
                xi_real[terms.index(sparse_term)] = one_xi[one_terms.index(sparse_term)] / fix_norm

            x = get_lib_x(full_image);
            theta, _ = create_library(x, degree)
            # X,Y = get_xy(full_image)
            surface = theta.T @ xi_real;
            surface = surface.reshape(full_image.shape)
            fits.append(surface)
        return (fits, all_terms)
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
        return full_image - surface, terms_best, surface
    else:
        return full_image - surface


def sym_reg_sparse(X, Y, Z, degree, full_image, surf=False, just_fit=False, terms_in_fit=False, secondary_trials=False):
    global num_of_vars, num_of_terms, num_not_nans
    num_of_vars = 2  # number of variables to have data for
    if len(X.shape) != 2:
        X = X.reshape(-1, 1)
    if len(Y.shape) != 2:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) != 2:
        Z = Z.reshape(-1, 1)
    num_of_terms = choose(num_of_vars + degree, degree)
    num_not_nans = np.sqrt(np.sum(~np.isnan(full_image)))

    x = np.concatenate((X, Y, np.ones(X.shape)), axis=1).T
    data = Z

    global normalizations, best
    normalizations = np.ones(num_of_vars)
    for var in range(num_of_vars):
        normalizations[var] *= abs((np.mean(np.abs(x[var]))))

        x[var] /= normalizations[var]
        # data.T[var]/=normalizations[var]

    theta, terms = create_library(x, degree)
    # finds initial regression coefficients
    xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta, theta.T)), theta), data)
    xi_best = xi.copy();
    terms_best = terms.copy()

    xi_final = {x: [0] for x in
                terms};  # initialize dictionary that will show final coefficient matrix with term labels
    xi_real = np.zeros(num_of_terms)
    times = [];
    # for var in range(num_of_vars):
    skip = []
    magnitude = np.floor(np.log10(theta.shape[1]) - 4)
    if magnitude < 0: magnitude = 0

    step_size = int(10 ** magnitude)
    step_size = 1
    theta_var = theta[:, ::step_size]
    data = data[::step_size]
    terms_var = terms.copy()
    res_initial = residual(theta_var, xi, data)
    res = res_initial;
    res_best = res_initial
    global residues, all_terms, all_xi
    residues = [res_best];
    all_terms = [terms.copy()];
    all_xi = [xi]  # [res_initial]
    for term in range(num_of_terms - 1):  # Delete all except one term in regression
        trial_residues = [];
        for i in range(
                num_of_terms - term):  # for each term to delete, run through all possible ones to minimize residual
            trial = np.delete(theta_var, i, 0)  # trial library
            if np.linalg.det(np.matmul(trial, trial.T)) == 0:
                trial_residues.append(res + 10 * res)
                continue
            time1 = time.time()
            xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(trial, trial.T)), trial), data)  # trial coefficient matrix
            time2 = time.time()
            times.append(time2 - time1)
            res = residual(trial, xi, data)
            if term == 0 and i in skip:
                # print(skip)
                res *= 1000
            trial_residues.append(res)
        # plt.semilogy(trial_residues)
        # plt.title('Fails: {}'.format(fails))
        # plt.close()
        term_to_del = trial_residues.index(
            min(trial_residues))  # find the best term to delete from the possibilities above

        if term == 0 and secondary_trials == True: skip.append(term_to_del)
        # delete term that mimized residual
        terms_var.pop(term_to_del)

        theta_var = np.delete(theta_var, term_to_del, 0)
        # find coefficents and residual again
        xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta_var, theta_var.T)), theta_var), data)
        residues.append(min(trial_residues))
        # save the coefficient matrix with the best residual

        all_terms.append(terms_var.copy())

        all_xi.append(xi)
    global d
    d, elbow = get_elbow(residues, xy=False, elbow_point=True)
    terms_best = all_terms[elbow]
    xi_best = all_xi[elbow]
    # undo normalization of best-fit term coefficients
    for sparse_term in terms_best:
        fix_norm = 1

        for term in sparse_term:
            print(sparse_term)
            fix_norm *= normalizations[int(term)]
        xi_final[sparse_term] = [xi_best[terms_best.index(sparse_term)] / fix_norm]
        xi_real[terms.index(sparse_term)] = xi_best[terms_best.index(sparse_term)] / fix_norm

    x = get_lib_x(full_image);
    theta, _ = create_library(x)
    X, Y = get_xy(full_image)
    surface = theta.T @ xi_real;
    surface = surface.reshape(len(X), len(Y.T))
    if just_fit:
        return surface
    elif terms_in_fit:
        return full_image - surface, terms_best
    elif surf:
        return full_image - surface, terms_best, surface
    else:
        return full_image - surface


def get_r2(orig, fit):
    numer = np.nansum((orig - fit) ** 2)
    denom = np.nansum((orig - np.nanmean(orig)) ** 2)
    r2 = 1 - numer / denom
    return r2


def subparaboloid(img, full_image=np.array([1]), dx=1, just_fit=False, fit=False, background_fit=False):
    """Substracts a paraboloid from the image, dx is the px distance in
    which the image is sampled for the fitting"""
    if full_image.shape == (1,): full_image = img.copy()
    X, Y = np.meshgrid(np.arange(0, img.shape[1], dx), np.arange(0, img.shape[0], dx))
    XX = X.flatten()
    YY = Y.flatten()
    Z = img[YY, XX]
    idx = ~np.isnan(Z)  # Remove NaN values
    data = (np.vstack((XX[idx], YY[idx], Z[idx]))).transpose()
    X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    XX = X.flatten()
    YY = Y.flatten()
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]

    global C
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    X, Y = np.meshgrid(np.arange(full_image.shape[1]), np.arange(full_image.shape[0]))
    Z = C[4] * X ** 2. + C[5] * Y ** 2. + C[3] * X * Y + C[1] * X + C[2] * Y + C[0]
    if just_fit == True:
        return Z
    elif fit == True:
        return full_image - Z, Z
    elif background_fit:
        return full_image - Z, img - Z, Z
    else:
        return full_image - Z


def fit_lows(image, full_image=np.array([1]), N=5, distribution=False, percent=0.1, degree=2, get_images=False,
             just_fit=False):
    '''input:
        image - 2d array
        full_image - if different than image
        N - can be 1 or 2 dimensions. Number of windows in each axis
        percent - fraction of pixels in each frame to use
       Return:
        fitted'''
    X, Y = cartesian(image)
    if full_image.shape == (1,): full_image = image.copy()
    if not hasattr(N, '__iter__'): N = [N, N]

    images = [np.array_split(arr, N[1], axis=1) for arr in np.array_split(image, N[0], axis=0)]
    images_dic = {(j, i): d for i, dd in enumerate(images) for j, d in enumerate(dd)}
    if get_images:
        return images_dic
    XX = [np.array_split(arr, N[1], axis=1) for arr in np.array_split(X, N[0], axis=0)]
    YY = [np.array_split(arr, N[1], axis=1) for arr in np.array_split(Y, N[0], axis=0)]

    xs, ys, zs = [], [], []
    if distribution: plt.figure()
    for n1, (x, y, z) in enumerate(zip(XX, YY, images)):
        for n0, (x, y, z) in enumerate(zip(x, y, z)):
            if np.sum(~np.isnan(z)) > 0:
                flat = z.copy()  # sym_reg(z,1,normal=True)
                args = np.unravel_index(np.argsort(flat.flatten())[int(.01 * flat.size):int(percent * flat.size)],
                                        z.shape)
            else:
                args = []
            xs.extend(x[args])
            ys.extend(y[args])
            zs.extend(z[args])
            if distribution: plt.hist(z[args], bins=100, label=f'{n0}, {n1}')
    if distribution: plt.legend()
    X, Y, Z = np.array([xs, ys, zs], dtype=float)
    nan_mask = ~np.isnan(Z)
    X = X[nan_mask]
    Y = Y[nan_mask]
    Z = Z[nan_mask]
    theta, _ = create_library([X, Y], degree)
    xi = np.matmul(np.matmul(np.linalg.inv(np.matmul(theta, theta.T)), theta), Z)
    theta_full, _ = create_library(get_lib_x(full_image), degree)
    fit = np.matmul(theta_full.T, xi).reshape(full_image.shape)
    if just_fit:
        return fit
    return full_image - fit


def get_cid_from_array(array):
    string = array.tobytes()
    compressor = lzma.LZMACompressor()
    compressor.compress(string)
    compressed = compressor.flush()
    cid = len(compressed) / len(string)
    return cid


def draw_ellipse(ax, x, y, n_std=2, alpha=0.2, **kwargs):
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=lambda_[0] * n_std,
        height=lambda_[1] * n_std,
        angle=np.rad2deg(np.arccos(v[0, 0])),
        alpha=alpha,
        **kwargs
    )
    ax.add_patch(ell)


def cartesian(arr, central=False):
    [I, J] = [i.reshape(arr.shape) for i in get_lib_x(arr)]
    if central:
        I -= np.mean(I)
        J -= np.mean(J)
    return I, J


def polar(arr, ij=[None, None], ints=False):
    I, J = cartesian(arr)
    I -= ij[0] if not ij[0] == None else np.mean(I) if not ints else int(np.mean(I))
    J -= ij[1] if not ij[1] == None else np.mean(J) if not ints else int(np.mean(J))
    R = np.sqrt(I ** 2 + J ** 2)
    th = np.where(R > 0, np.arccos(-I / R), 0)
    th[J < 0] *= -1
    th[J < 0] += 2 * np.pi
    return R, th


def radial_avg(image, dk=5, get_x=False, mask=False, avg=['mean']):
    if len(image.shape) == 1: image = image.reshape(1, -1)
    image = np.abs(image)
    # Get image parameters
    freqr = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    freqc = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    # Find radial distances
    [kx, ky] = np.meshgrid(freqc, freqr)
    k = np.sqrt(np.square(kx) + np.square(ky));
    R, th = polar(image, ij=np.unravel_index(np.arange(k.size)[k.ravel() == 0][0], k.shape))
    stop = np.max(R)
    # dif = 1/np.sqrt(np.sum(k<=stop))*dk
    # x = np.arange(0,stop,dif)
    # if get_x: return x
    if type(mask) == int or type(mask) == float:
        I, J = cartesian(image, True)
        image = np.where((I < mask) | (J < mask), np.nan, image)
    elif type(mask) == np.ndarray:
        image = np.where(mask, image, np.nan)

    x = []
    rad = {'mean': [], 'std': [], 'energy': []}
    for r in np.arange(0, stop, dk):
        mask = (R > r - dk / 2) & (R <= r + dk / 2)
        x.append(np.nanmean(k[mask]))
        dat = image[mask]
        if 'mean' in [i.lower() for i in avg]:
            rad['mean'].append(np.nanmean(dat))
        if 'std' in [i.lower() for i in avg]:
            rad['std'].append(np.nanstd(dat))
    if 'energy' in [i.lower() for i in avg]:
        rad['energy'] = np.sum(np.abs(image) ** 2)
    rad = {i: np.array(d) if i != 'energy' else d for i, d in rad.items()}
    x = np.array(x)
    if get_x:
        return x
    return rad, x


def power_spectrum(data, c=None, fmt='-', label=None, new_fig=False, plot=True, log=False, mask=False, dk=5,
                   resolution=None, window=True, alpha=0.8, full=False, avg='mean'):
    window2d = np.ones_like(data)
    if len(data.shape) == 2 and window:
        window2d = np.sqrt(np.abs(np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))))
    elif window:
        window2d = np.hanning(data.shape[0])
    data = np.where(np.isnan(data), np.nanmean(data), data) * window2d
    ft = fft.fftn(data)
    shift = fft.fftshift(ft)
    s = np.abs(shift)

    if not isinstance(avg, (list, tuple, np.ndarray)): avg = [avg]
    spect = []
    spectra, x = radial_avg(s, mask=mask, dk=dk, avg=avg)
    for a in avg:
        spect.append(spectra[a])
    if len(spect) == 1: spect = spect[0]
    lamb = x
    if not plot:
        if full:
            return x, spect, shift * np.conj(shift)
        else:
            return lamb, spect
    elif len(avg) > 1:
        print('Not made to plot multiple version of spectra.')
    if new_fig: plt.figure()
    ax = plt.gca()
    if log:
        ax.loglog(x, spect, ls=fmt, c=c, alpha=alpha, label=label, )
    else:
        ax.semilogy(x, spect, ls=fmt, c=c, alpha=alpha, label=label)
    plt.xlabel('Wavenumber (1/px)')
    if resolution != None:
        tick_labels = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2e3, 5e3, 1e4, 2e4, 5e4]

        def ktoL(k, resolution=resolution * (1e6 if np.log10(resolution) < -4 else 1)):
            return resolution / k

        def Ltok(l, resolution=resolution * (1e6 if np.log10(resolution) < -4 else 1)):
            return resolution / l

        from matplotlib.ticker import FormatStrFormatter
        ax = plt.gca().secondary_xaxis('top', functions=(ktoL, Ltok))
        ax.invert_xaxis()
        ax.set_xticks(tick_labels)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('Wavelength (um)')
    plt.pause(0.05)
    # x = 1/np.flip(x)
    if label != None:
        plt.legend()
    if full:
        print('True')
        return x, spect, shift * np.conj(shift)
    return x, spect


def get_zoomed_images(image, title=None, N=4, histogram=False):
    data = image.copy()
    x_w = int(np.floor(data.shape[0] / N))
    y_w = int(np.floor(data.shape[1] / N))
    out = [];
    loc = []
    if histogram:
        l = 121
    else:
        l = 111
    for i in range(N):
        for j in range(N):
            loc.append([[i * x_w, (i + 1) * x_w], [j * y_w, (j + 1) * y_w]])
            dat = data[i * x_w:(i + 1) * x_w, j * y_w:(j + 1) * y_w]
            # dat,_ = fit_lows(dat)

            plot_contour(dat, location=l, vlims=[0, 1000])
            plt.title(title)
            if histogram:
                ax = plt.subplot(122)

                ax.hist(dat.flatten(), bins=np.arange(0, 1000 + 1, 20))
            plt.savefig('test.jpg')
            im = plt.imread('test.jpg')
            out.append(im)
    return out, loc


def show_zoom(data, locations, ith, subplot=111):
    y, x = data.shape
    outer = [[0, 0], [x, 0], [x, y], [0, y], [0, 0]]
    if type(ith) == int: ith = [ith]
    verts = []
    for i in ith:
        loc = locations[i]
        y = loc[0];
        x = loc[1]
        inner = [[x[0], y[0]], [x[0], y[1]], [x[1], y[1]], [x[1], y[0]], [x[0], y[0]]]
        verts.extend(inner)
    verts = outer + verts
    codes = np.ones(len(verts)) * Path.LINETO
    codes[::5] = np.ones(len(codes[::5])) * Path.MOVETO
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='black', alpha=0.5)

    plot_contour(data, location=subplot)
    ax = plt.gca()
    ax.add_patch(patch)


def approx_slopes(data, px_to_r=1000 / 868):
    grad = np.gradient(data)
    slope = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    reslope = slope / 1000 * px_to_r
    angles = np.where(~np.isnan(reslope), np.arctan(reslope), np.nan) * 180 / np.pi
    return angles


def make_confusion(df, get_r=False, model='Model_KMeans', truth='Class'):
    '''assumes it gets most correct'''
    truth = np.array(list(df[truth]))
    model = np.array(list(df[model]))
    cipher = model[truth == 'R']
    count = {'0': 0, '1': 0}
    for i in cipher:
        count[i[-1]] += 1
    R = np.argmax(list(count.values()))
    S = [1, 0][R]
    confusion = np.zeros((2, 2))
    for t, m in zip(truth, model):
        confusion[int(t == 'S'), 0 if int(m[-1]) == R else 1] += 1
    if get_r:
        return np.int_(confusion), {R: 'R', S: 'S'}
    else:
        return np.int_(confusion)


def make_confusion(df, get_cipher=False, model='Model_KMeans', truth='Class'):
    '''assumes it gets most correct'''
    truth = np.array(list(df[truth]))
    model = np.array(list(df[model]))
    labels, counts = np.unique(truth, return_counts=True)
    cipher = {str(i): None for i in range(len(labels))}
    index = {i: n for n, i in enumerate(labels)}
    for label in np.flip(labels[np.argsort(counts)]):
        u, c = np.unique([i[-1] for i in model if label in i], return_counts=True)
        ind = np.flip(np.argsort(c))
        num = 0
        try:
            while cipher[u[ind[num]]] is not None:
                num += 1
            cipher[u[ind[num]]] = label
        except:
            for i, d in cipher.items():  ####This is probably not general
                if d is None:
                    cipher[i] = label
                    break
    confusion = np.zeros((len(labels), len(labels)))
    for t, m in zip(truth, model):
        m = cipher[m[-1]]
        confusion[index[t], index[m]] += 1
    if get_cipher:
        return np.int_(confusion), cipher
    else:
        return np.int_(confusion)


def make_confusion_from_results(df, model='model_SVM-linear', truth='Class'):
    if 'model' not in model: model = 'model_' + model
    truth = np.array(df[truth].values)
    model = np.array(df[model].values)
    mat = np.zeros((len(np.unique(truth)), len(np.unique(model))))
    for t, m in zip(truth, model):
        i = ['R', 'S'].index(t)
        j = int(np.logical_not(bool(i)) ^ m)
        mat[i][j] += 1
    return mat


def plot_comparison(df, real, model, new_fig=False):
    if not new_fig:
        fig = plt.figure('Comparison Confusion Matrix')
        ax = plt.gca();
        ax.remove();
        ax = fig.add_subplot()
    else:
        fig, ax = plt.subplots()
    x_labels = np.unique(df[model])
    y_labels = np.unique(df[real])
    new_df = pd.DataFrame(np.zeros((len(y_labels), len(x_labels))),
                          columns=x_labels, index=y_labels)
    for g, m in zip(df[real], df[model]):
        new_df.at[g, m] += 1
    mat = np.int_(new_df.values)
    tot = np.sum(mat, axis=1);
    tot = np.where(tot == 0, 1, tot)
    percents = (mat.T / tot).T
    percents = np.concatenate((percents, [[1, 0]] * len(mat)), axis=1)
    ax.imshow(percents, cmap='coolwarm')
    ax.set_xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    ax.set_ylabel('Real Classification')
    ax.set_xlabel('Model Classification')
    ax.set_title('Confusion Matrix')
    fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height = bbox.height / mat.shape[0] * 7 * plt.rcParams['font.size']
    size = int(np.ceil(height / 4))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i][j])[:4], color='white', size=size, ha='center', va='center')
    ax.set_xlim([-0.5, mat.shape[1] - 0.5])
    ax.set_ylim([mat.shape[0] - 0.5, -0.5])


def plot_confusion(mat, labels=['R', 'H', 'S'], ax=None, stats=True, plot=True):
    if type(mat) == str:
        mat = np.loadtxt(mat, delimiter=',', dtype=int)
    else:
        mat = np.array(mat)
    if all([isinstance(i, list) for i in labels]):
        x_labels, y_labels = labels
        print(x_labels, y_labels)
    else:
        x_labels, y_labels = labels, labels
    tot = np.sum(mat, axis=1);
    tot = np.where(tot == 0, 1, tot)
    percents = (mat.T / tot).T
    percents = np.concatenate((percents, [[1, 0]] * len(mat)), axis=1)
    if plot:
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        ax.imshow(percents, cmap='coolwarm')
        ax.set_xticks(ticks=np.arange(len(x_labels)), labels=[x[0] for x in x_labels])
        ax.set_yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
        ax.set_ylabel('Real Classification')
        ax.set_xlabel('Model Classification')
        ax.set_title('Confusion Matrix')
        fig = plt.gcf()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        height = bbox.height / mat.shape[0] * 7 * plt.rcParams['font.size']
        size = int(np.ceil(height / 4))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(mat[i][j])[:4], color='white', size=size, ha='center', va='center')
        ax.set_xlim([-0.5, mat.shape[1] - 0.5])
        ax.set_ylim([mat.shape[0] - 0.5, -0.5])
    stat_dict, these_stats = {}, {}
    stat_dict['Accuracy'] = np.trace(mat) / np.sum(mat)
    stat_dict['Sensitivity'] = mat[0][0] / np.sum(mat[0])
    stat_dict['Specificity'] = np.sum(mat[1:].T[1:]) / np.sum(mat[1:]) if np.sum(mat[1:]) > 0 else np.nan
    all_stats = [stat.lower() for stat in stat_dict.keys()]
    if type(stats) == bool and stats:
        stats = 0
    elif type(stats) == int:
        negs = np.delete(np.arange(len(mat)), stats)
        acc = np.trace(mat) / np.sum(mat)
        sens = mat[stats][stats] / np.sum(mat[stats])
        spec = np.sum(mat[negs].T[negs]) / np.sum(mat[negs])
    elif isinstance(stats, (str, list, np.ndarray)):
        if isinstance(stats, str): stats = [stats]

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


def meta_from_excel(fold='', file=''):
    f = 0
    if len(fold) > 0:
        for root, folds, files in os.walk(fold):
            for file in files:
                if not file.endswith('.xlsx'): continue
                if not len(file.split('_')) == 3: continue
                excel = root + '/' + file
                xl = np.array(pd.read_excel(excel), dtype=str)
                for n, i in enumerate(xl):
                    if i[0].startswith('File'):
                        break
    elif len(file) > 0:
        xl = np.array(pd.read_excel(file), dtype=str)
        for n, i in enumerate(xl):
            if i[0].startswith('File'):
                break
            elif 'Interferometry_number' in i:
                f = list(i).index('Interferometry_number')
                break
    d = xl[n + 1:, 1:]
    data_categ = xl[n][1:]
    file_names = xl[n + 1:].T[f]
    dfd = {f: {c: d[fn, cn] for cn, c in enumerate(data_categ)} for fn, f in enumerate(file_names)}
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


def get_inds(arr):
    X = np.array([np.arange(arr.shape[1]) for i in range(arr.shape[0])])
    Y = np.array([np.ones(arr.shape[1]) * i for i in range(arr.shape[0])])
    return X, Y


def lmat(file, mat={}):
    scipy.io.loadmat(file, mat)
    mat = {i: mat[i] for i in mat if not i.startswith('__')}
    return mat


def vol_dilutions(data_dict, latres=False, log=False, norm=True, sns=['4', '7', '10', '11'],
                  ods=[0.1, 0.2, 0.3, 0.4, 0.5], close=True, best_fit=True):
    if close:
        plt.close('Volumes')
        plt.close('Avg Volumes')
    # plt.figure('Volumes')
    plt.figure('Avg Volumes')
    cmap = plt.get_cmap('Set1')
    vols = {sn: {od: [] for od in ods} for sn in sns}
    stds = vols.copy()
    this_vols = []
    if type(latres) == bool:
        mult = 1
        axis = 'mm*px^2'
    else:
        mult = (latres * 1000) ** 2
        axis = 'mm^3'
    for i, d in data_dict.items():
        # print(i)
        sn = i.split('_')[0]
        if sn not in vols: continue
        col = cmap(sns.index(sn))
        od = float(i.split('_')[2])
        vol = np.sum(d) / 1000 / 1000 * mult
        # stds[sn][int(float(od)*10)-1]= np.std(this_vols) if len(this_vols)>1 else np.sqrt(this_vols[0])
        vols[sn][od].append(vol)
    stds = {sn: {od: np.std(data) if len(data) > 1 else np.sqrt(data[0]) / 100 for od, data in sn_dict.items()} for
            sn, sn_dict in vols.items()}
    vols = {sn: {od: np.mean(data) for od, data in sn_dict.items()} for sn, sn_dict in vols.items()}
    plt.ylabel('Approx Volume [%s]' % axis)
    plt.xlabel('OD (5ul)')
    plt.figure('Avg Volumes')
    for i, d in vols.items():
        if norm:
            ref = list(d.values())[0]
        else:
            ref = 1
        x = np.array(list(d.keys()))
        y = np.array(list(d.values())) / ref
        st = np.array(list(stds[i].values())) / ref
        plt.errorbar(x, y, st, fmt='-o', color=cmap(sns.index(i)), alpha=0.9, label=i)
    if norm:
        plt.plot(x, x / x[0], '--k', alpha=0.7, label='Linear')
        plt.ylabel('Relative Approx. Volume')
    else:
        plt.ylabel('Approx Volume [%s]' % axis)
    plt.xlabel('OD (5ul)')
    plt.legend()
    fig = plt.figure('Avg Volumes')
    ax = plt.gca()
    if best_fit:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        height = bbox.height * fig.dpi
        m = 0;
        fits = {}
        for n, (s, v) in enumerate(vols.items()):
            if any([i == 0 for i in stds[s]]): continue
            std = np.array(list(stds[s].values()))
            x = ods
            y = np.array(list(v.values()))
            slope, intercept = np.polyfit(np.log(x), np.log(y), 1, w=1 / np.log(std))
            yfit = np.exp(slope * np.log(x) + intercept)
            fig = plt.figure('Avg Volumes')
            ax = plt.gca()
            ax.plot(x, yfit, color=cmap(sns.index(s)), alpha=0.5)
            ax.text(1, 0 + 15 / height * n, s + ': ' + str(slope)[:4], ha='right', transform=ax.transAxes)
            m += 1
            fits[s] = [slope, intercept, y[0]]
        ax.text(0.95, 15 / height * (m + 1), 'Slopes', transform=ax.transAxes, ha='center')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    if best_fit:
        return vols, stds, fits
    else:
        return vols, stds


def power_from_full(dic, rad=800, desired_od=0.1):
    '''not a general function'''
    plt.close('all')
    ods = {'0.1': 1, '0.2': 1}
    homes = {}
    for i, d in dic.items():
        od = i.split('_')[2]
        per = '_'.join(i.split('_')[-2:])
        if str(desired_od) != od: continue
        I, J = cartesian(d, True)
        dat = np.where((np.abs(I) < rad) & (np.abs(J) < rad), d, np.nan)
        bI = int(I[-1][-2] - rad)
        bJ = int(J[-1][-1] - rad)
        c1 = sym_reg(dat, 2, normal=True)[bI:bI + 2 * rad, bJ:bJ + 2 * rad]
        plt.figure(od)
        plot_contour(c1, location=(6, 6, ods[od]), new_fig=False, cbar=False, axis=False)
        plt.title(per)
        ods[od] += 1
        c1 = np.where(np.isnan(c1), np.nanmean(c1), c1)
        homes[i] = c1
        ft = 1 + np.log10(np.abs(np.fft.fftshift(np.fft.fft2(c1))))
        plot_contour(ft, cmap='gray', cbar=False, axis=False, new_fig=False, location=(6, 6, ods[od]))
        ods[od] += 1
        plt.figure('Power')
        col = {'R': 'red', 'H': 'blue', 'S': 'black'}[i[-1]]
        x, spect = power_spectrum(c1, c=col, label=per, log=True)


def update_database(new_data, time='8hr', drug='TZP', od='0.2', save=False):
    root = 'my data/for classification/'
    cur_files = os.listdir(root)
    files = []
    mat = new_data
    for file in cur_files:
        if not file.endswith('.mat'): continue
        spl = file[:-4].split('_')
        if not (time == spl[0] and drug == spl[1] and od == spl[2]): continue
        files.append([int(spl[3]), file])
    if len(files) > 0:
        file = files[np.argsort(np.array(files).T[0])[-1]][1]
        mat = lmat(root + file, new_data)
    new_file = '_'.join([time, drug, od]) + '_' + str(len(mat)) + '.mat'
    if save:
        scipy.io.savemat(new_file, mat)
    return mat


def make_PAP_curves(file='my data/clinical isolates/Selected_PAP_data.xlsx'):
    df = pd.read_excel(file, sheet_name='Percent Survival', header=0, index_col=1)
    plt.close('all')
    plt.figure()
    plt.plot(df.columns[1:], np.ones(len(df.columns) - 1) * 10 ** (-6), '--k')
    for sn in df.index:
        print(sn)
        bkpt = df.loc[sn][1]
        line = '-' if bkpt > 0.5 else '--' if bkpt > 10 ** (-6) else ':'
        plt.loglog(df.columns[1:], np.array(list(df.loc[sn][1:])), line + 'o', label=sn)
    plt.legend()
    plt.plot([1, 1], plt.ylim(), '--k', alpha=0.3)
    plt.xlabel('Breakpoint AB (TZP)')
    plt.ylabel('Relative CFU')
    plt.set_cmap('tab20')


def reclassify(des_dic, data):
    '''des_dic can be a filename or the pandas dataframe
       data should be the dictionary of data'''
    if type(des_dic) == str:
        des_dic = pd.read_excel(des_dic, header=0)
    new_data = {}
    for i, d in data.items():
        strain_num = int(i.split('_')[0])  # re.split('(\d+)',file[:-5])[1] + '_CRE'*('CRE' in file)
        try:
            strain_rep = int(i.split('_')[1])
        except:
            strain_rep = 1
        strain_fold = i.split('-')[1].split('_')[0] if '-' in i else ''
        des = list(des_dic.loc[[sn == strain_num and r == strain_rep and f == strain_fold for sn, r, f in
                                zip(des_dic['Strain Number'], des_dic['Replicate'], des_dic['Folder'])]][
                       'Classification'])[0]
        new_i = i[:-1] + des
        new_data[new_i] = d
    return new_data


def create_images(imgs, dt=20):
    newimgs = []
    for n, img in enumerate(imgs):
        if n != 4: continue
        plot_contour(img, clear=True)
        plt.text(*np.flip(img.shape[:2]), str(dt * n) + ' min\n' + str(dt * n / 60)[:3] + ' hr', ha='right', va='top',
                 color='white', font='Times New Roman', size=22)
        rect = plt.Rectangle([0, 0], img.shape[1] * n / (len(imgs) - 1), img.shape[0] / 20, color='white')
        plt.gca().add_patch(rect)
        for lab in [1, 2, 4, 8, 12, 24, 36, 48]:
            if lab > dt * (len(imgs) - 1) / 60: continue
            x = lab * 60 / (dt * (len(imgs) - 1))
            atmax = x > 0.95
            plt.text(min([x, 1]) * img.shape[1], img.shape[0] / 20, str(lab) + ' hrs',
                     ha='center' if not atmax else 'right', va='bottom', color='red', font='Times New Roman', size=18)
            plt.plot(x * img.shape[1] * np.ones(2), [0, img.shape[0] / 20], 'r')
        plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
        newimgs.append(plt.imread('test.png'))
    return newimgs


# =============================================================================
# import cv2,rawpy
# def make_video(imgs,filename,fps=10):
#     if type(imgs[0])==str:
#         newimgs = []
#         for imgf in imgs:
#             raw = rawpy.imread(imgf)
#             img = raw.postprocess()
#             newimgs.append(img)
#         imgs = newimgs.copy()
#         imgs = create_images(imgs,20)
#     video = cv2.VideoWriter('.'.join(filename.split('.')[:-1])+f'-{fps}fps.mp4',0x7634706d,fps,tuple(np.array(imgs[0].shape)[[1,0]]))
#     for img in imgs:
#         video.write(img)
#     video.release()
#     cv2.destroyAllWindows()
# =============================================================================


from sklearn.cluster import KMeans


def lab_scores(scores, n_clusters=3):
    '''rtlc = ['aawaz','maryam','ray','pablo','emma','miles','adam'];
hr = ['pablo','ray','adam','miles','maryam','aawaz','emma'];
mb = ['ray','miles','adam','aawaz','pablo','maryam','emma'];
names = ['Miles','Pablo','Aawaz','Adam','Emma','Ray','Maryam']
scores = np.array([[i.index(name.lower()) for i in [rtlc,hr,mb]] for name in names])'''
    fig = plt.figure();
    ax = fig.add_subplot(projection='3d')
    model = KMeans(n_clusters=n_clusters).fit(scores)
    ax.scatter(*(scores.T), color=np.array(['red', 'blue', 'black'])[model.labels_])
    ax.set_xlabel('RTLC')
    ax.set_ylabel('HR')
    ax.set_zlabel('MB')


# plt.close('all')
# lab_scores(scores,2)


def curvature(img, latres=1):
    '''Curvature found in https://mathworld.wolfram.com/Curvature.html eq 17. (Gray 1997)
        img: 2d image array
        latres: k given in units of [latres]^-1'''
    dx, dy = np.gradient(img, latres)
    dxx, dxy = np.gradient(dx, latres)
    dyx, dyy = np.gradient(dy, latres)
    return (dxx * dy ** 2 - 2 * dxy * dx * dy + dyy * dx ** 2) / (dx ** 2 + dy ** 2) ** (3 / 2)


def ring_width(arr, plot=False, vlims=[None, None], edges=False, all_data=False, just_ring=False, ax=None, **kwargs):
    arr = np.where(np.isnan(arr), 0, arr)
    max_inds = np.argmax(arr, axis=1)
    max_vals = np.diag(arr[:, max_inds])
    if just_ring:
        return max_vals
    start_counter = 1
    med = 0
    while True:
        start_counter += 1
        inds = max_inds.copy()
        test = np.array([row[i] for row, i in zip(arr, inds)]) > max_vals / 2
        while any(test):
            inds = inds - np.int_(test)
            if any((inds >= arr.shape[1] - 1) | (inds == 0)):
                break
            test = np.array([row[i] for row, i in zip(arr, inds)]) > max_vals / 2
        start = inds.copy()
        # break
        med_start = int(np.median(start))
        if np.sum(max_inds < med_start) > 0 and start_counter < 10:
            max_inds = np.argmax(arr.T[med_start:], axis=0) + med_start
        else:
            break
    end_counter = 1
    while True:
        end_counter += 1
        inds = max_inds.copy()
        test = np.array([row[i] for row, i in zip(arr, inds)]) > max_vals / 2
        while any(test):
            inds = inds + np.int_(test)
            if any((inds >= arr.shape[1] - 1) | (inds == 0)):
                break
            test = np.array([row[i] for row, i in zip(arr, inds)]) > max_vals / 2
        end = inds.copy()
        # break
        med_end = int(np.median(end))

        if np.sum(max_inds > med_end) > 0 and end_counter < 10:
            max_inds = np.argmax(arr.T[med_start:med_end], axis=0) + med_start
            max_vals = np.diag(arr[:, max_inds])
        else:
            break

    widths = end - start
    if ax is not None:
        plot = True
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        plot_contour(arr, ax=ax, vlims=vlims, cbar=False)
        ax.plot(np.flip(max_inds) - np.ones_like(max_inds), np.arange(len(max_inds)), 'k')
        ax.plot(np.flip(start), np.arange(len(start)), 'gray')
        ax.plot(np.flip(end), np.arange(len(end)), 'gray')
    if edges:
        return start, end, widths
    elif all_data:
        return start, end, widths, max_inds
    return widths


from roughness_PB import getwloc, w_data_extraction, linregress

from tqdm import tqdm

import json
import numpy as np


def save_json(filename, obj):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(f'{filename}' + ('.pkl' if 'json' not in filename else ''), 'w') as handle:
        json.dump(convert(obj), handle)


def read_json(filename):
    def convert(obj):
        if isinstance(obj, list):
            try:
                return np.array(obj)
            except:
                return [convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(f'{filename}' + ('.pkl' if 'json' not in filename else ''), 'r') as handle:
        obj = json.load(handle)
    return convert(obj)


import pickle


def save_dataframe(path, df):
    """
    Save a DataFrame using pickle.

    Parameters:
    - path: Name of the file (with .pkl extension).
    - df: DataFrame to save.
    """
    if not path.endswith('.pkl'):
        path += '.pkl'

    with open(path, 'wb') as f:
        pickle.dump(df, f)


def read_dataframe(path):
    """
    Load a DataFrame using pickle.

    Parameters:
    - path: Name of the file (with .pkl extension).

    Returns:
    - Loaded DataFrame.
    """
    if not path.endswith('.pkl'):
        path += '.pkl'

    with open(path, 'rb') as f:
        df = pickle.load(f)

    return df


def get_features(image, obj=10, zoom=1, fftval=None):
    latres = {5.5: 1.5614175326405541, 50: 1.7334291543917991e-01, 10: 8.605292006113655e-01}  # latres in micron/px
    latres = latres[obj]
    latres /= zoom
    all_feats = ['Home Height', 'Home Var', 'Home CoV', 'Ring Height', 'Ring CoV', 'Ring Wsat', 'Ring Hurst',
                 'ringH/homeH', 'ringH*homeH', 'RWidth', 'RWidth Var', 'RWidth CoV']
    if fftval is not None:
        if not hasattr(fftval, '__iter__'): fftval = [fftval]
        for val in fftval:
            all_feats.append('Power-{}um'.format(str(val)[:4]))
    if image is None: return all_feats
    image = np.where(np.isnan(image), np.nanmedian(image), image)
    home = image[300:700, :400] if max(image.shape) * .9 >= min(image.shape) else image[:, :700]
    left, right, ring_w, ring = ring_width(image, all_data=True)
    all_feats.extend(['R-inside-array', 'R-outside-array', 'R-peak-array'])
    coefs = np.polyfit(np.arange(len(ring)), ring, 1)
    fit = np.sum([np.arange(len(ring)) ** (1 - d) * c for d, c in enumerate(coefs)], axis=0)
    flucs = ring - fit
    rh = np.median(ring);
    rv = np.var(flucs)

    rw_med = np.median(ring_w)
    rw_var = np.var(ring_w)
    rw_cov = np.sqrt(rw_var) / rw_med
    med = np.median(home);
    var = np.var(home)

    powerfeats = []
    if fftval is not None:
        x, spect = np.array(power_spectrum(home, plot=False))
        for val in fftval:
            index = np.argmin(np.abs(1 / x * latres - val))
            powerfeats.append(np.log(spect[index]))

    loc, wloc = getwloc(flucs, latres, rx=0.3)
    l_sat, w_sat, h = w_data_extraction(loc, wloc)
    hurst = linregress(np.log10(loc)[:15], np.log10(wloc)[:15]).slope

    feats = [med, var, var / med, rh, rv / rh, w_sat, hurst, rh / med, rh * med, rw_med, rw_var,
             rw_cov] + powerfeats + [left, right, ring]  # ,spect[0],m]
    feats = {'f' + i: d for i, d in zip(all_feats, feats)}
    return feats


def fft_features(arr, obj=10, zoom=1, num_fftvals=10, wavelengths=None, min_wavelength=0):
    latres = {5.5: 1.5614175326405541, 50: 1.7334291543917991e-01, 10: 8.605292006113655e-01}  # latres in micron/px
    latres = latres[obj] / zoom

    shape = arr.shape
    if len(shape) == 1 or shape[-1] == 1:
        k = np.abs(np.fft.fftshift(np.fft.fftfreq(shape[0])))
        dim_check_fail = True
    else:
        dim_check_fail = False
        freqs = []
        for n, i in enumerate(shape):
            k = np.fft.fftshift(np.fft.fftfreq(i))
            freqs.append(np.array([k for j in range(shape[len(shape) - 1 - n])]))
        k = np.sqrt(freqs[0] ** 2 + freqs[1].T ** 2)
    k = np.sort(k.ravel())
    if wavelengths is None:
        wavelengths = np.logspace(*np.log10(latres / k[[-1, 1]]), num_fftvals, dtype=float)
        wavelengths = wavelengths[wavelengths > min_wavelength]

    x, (spect_m, spect_std, energy) = np.array(power_spectrum(arr, plot=False, dk=3, avg=['mean', 'std', 'energy']),
                                               dtype=object)
    log_spect_m, log_spect_std = np.log([spect_m, spect_std])
    all_wavelengths = latres / x
    feats = {'power energy': energy}
    for name, stat in zip(['mean', 'std', 'cov'], [log_spect_m, log_spect_std, log_spect_std / log_spect_m]):
        if dim_check_fail and name != 'mean': continue
        if wavelengths != 'all':
            for val in wavelengths:
                index = np.argmin(np.abs(all_wavelengths - val))
                feats[f'power {name}-' + str(val)[:4] + 'um'] = stat[index]  # log accounted for by log_spect_...
        else:
            for freq, val in zip(x, stat):
                wavelength = latres / freq if freq > 0 else 'inf'
                feats.update({f'power {name}-{np.round(wavelength, 2)}': val})
    return feats, (x, spect_m)


def get_all_fft_features(arr, obj, zoom, wavelengths=None, plot=False, **kwargs):
    full = arr.copy()
    homeland = arr[:999, :701]
    left, right, _, ring_inds = ring_width(arr, all_data=True)
    ring = np.array([row[r] for row, r in zip(arr, ring_inds)])
    left_flucs = sym_reg(left, degree=1, normal=True)

    feats = {}
    for region, data in zip(['Full', 'Home', 'RingInside', 'RingPeak'], [full, homeland, left_flucs, ring]):
        these_feats, (x, spect) = fft_features(data, obj=obj, zoom=zoom, wavelengths=wavelengths)
        feats.update({'f' + region + i: d for i, d in these_feats.items()})
        if plot:
            plt.figure(region)
            plt.loglog(x, spect, **kwargs)
    return feats


def subtract_poly(line, deg=2, x_vals=None, just_fit=False):
    if not all([hasattr(l, '__iter__') for l in line]):
        lines = np.array([line])
    else:
        lines = np.array(line)
    fits = []
    for line in lines:
        if x_vals is None:
            x = np.arange(-len(line) / 2, len(line) / 2, 1) + 0.5
        else:
            x = np.array(x_vals)
        coefs = np.polyfit(x, line, deg)
        fit = np.sum([x ** (deg - d) * c for d, c in enumerate(coefs)], axis=0)
        fits.append(fit)
    fits = np.array(fits, dtype=object)
    if just_fit: return fits
    return lines - fits


def check_merge(df1, df2, match_on=None, return_input=False):
    """
    Merges two DataFrames and handles duplicates.

    This function merges two DataFrames (`df1` and `df2`) based on the specified columns (`match_on`).
    It checks for duplicates and prompts the user to decide whether to overwrite the old data or keep it.
    The merged DataFrame is returned.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame to merge.
    df2 (pd.DataFrame): The second DataFrame to merge.
    match_on (list or str, optional): The column(s) to match on for merging. If None, all columns are used. Default is None.

    Returns:
    pd.DataFrame: The merged DataFrame with duplicates handled based on user input.
    """
    duplicates = df1.merge(df2, on=match_on, how='inner').set_index(match_on)
    if not duplicates.empty:
        mask1 = df1.set_index(match_on).index.isin(duplicates.index)
        mask2 = df2.set_index(match_on).index.isin(duplicates.index)
        exact_duplicates = df1.loc[mask1].drop(columns=[i for i in df1.columns if 'array' in i],
                                               errors='ignore').reset_index(drop=True).eq(
            df2.loc[mask2].drop(columns=[i for i in df2.columns if 'array' in i], errors='ignore')).all(axis=1)
        if exact_duplicates.all():
            n = len(exact_duplicates)
            print(f'{n} new entries already existed.')
            merge = True
        else:
            n = len(exact_duplicates) - exact_duplicates.sum()
            print(f'\n\nWARNING: {n} previous entries found that match new ones.')
            merge = input('Would you like to overwrite the old data? (y,[n]): ').strip().lower().startswith('y')
    else:
        merge = True
    merged = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=match_on,
                                                                      keep='last' if merge else 'first').reset_index(
        drop=True)
    if return_input:
        return merged, merge
    else:
        return merged


def extract_metadata(folder, master_list=None, save_to_meta=True, return_input=False):
    """
    Extracts metadata from a specified folder and merges it with a master list.

    This function extracts metadata from a specified folder, processes it, and optionally saves it to a metadata file.
    It also merges the extracted metadata with a master list if provided.

    Parameters:
    folder (str or int): The folder from which to extract metadata. If an integer is provided, it will be converted to a string.
    master_list (pd.DataFrame, optional): A master list DataFrame to merge with the extracted metadata. If None, the master list will be read from a default location. Default is None.
    save_to_meta (bool, optional): Whether to save the extracted metadata to a metadata file. Default is True.

    Returns:
    pd.DataFrame: The extracted and processed metadata.
    """
    if isinstance(folder, int):
        folder = str(folder)
    folder = folder + ('/' if not folder.endswith('/') else '')
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    date = folder.split('/')[-2]
    working_folder = root + 'Interferometer/' + folder

    df = pd.read_excel(working_folder + f'METADATA_{date}.xlsx')

    df['Treatment'] = df['Treatment'] + df['Treatment Notes'].where(df['Treatment Notes'].notna(), '').radd(' ')
    df.drop(columns=['Treatment Notes'], inplace=True)

    df['Drug Treatment'] = df['Drug'] + '-' + df['Drug Concentration'].apply(
        lambda x: f"{x:07.3f}" if pd.notna(x) else '').radd(' ')

    df.fillna({'Drug Treatment': 'None', 'Treatment': 'None'}, inplace=True)

    if master_list is None:
        dfs = pd.read_excel(root + 'Strain Master List.xlsx', sheet_name=None)
    else:
        dfs = master_list.copy()

    dic = {key: genus for key, genus in dfs['Strain Identification'].values if isinstance(genus, str)}
    drugs = {key: drug for key, drug, gram in dfs['Drugs'].values if isinstance(drug, str)}
    clsi_phens = {}
    strain_info = pd.DataFrame()
    for n, sn in enumerate(df['Strain ID']):
        if isinstance(sn, str) and len(sn) > 4:
            key = sn[:-4]
            genus = dic[key]
            genus_df = dfs[genus]
            genus_df = genus_df.drop(columns=[col for col in genus_df.columns if col.startswith('Unnamed')])
            breakpoints = genus_df.loc[(genus_df[genus] == 'R') | (genus_df[genus] == 'S')]
            breakpoints = breakpoints.set_index(breakpoints.columns[0])
            mask = genus_df[genus] == sn
            this_mic = genus_df.loc[mask]

            if sn not in clsi_phens:
                clsi_phens = {sn: {drug: 'U' if np.isnan(this_mic[drug].values[0]) else 'R' if all(
                    this_mic[drug] >= breakpoints[drug]['R']) else 'S' if all(
                    this_mic[drug] <= breakpoints[drug]['S']) else 'I' for drug in breakpoints.columns}}

            mask = dfs['Strain List']['Strain ID'] == sn
            strain_meta = dfs['Strain List'].loc[mask].copy()

            this_sn = {i: d for i, d in zip(df.columns, df.values[n])}
            if this_sn['Drug'] == 'nab':
                strain_meta['Phenotype Expected'] = 'N'
                strain_meta['Phenotype CLSI'] = 'N'
            elif this_sn['Drug'] in drugs and drugs[this_sn['Drug']] in clsi_phens[sn]:
                drug = drugs[this_sn['Drug']]
                conc = this_sn['Drug Concentration']
                if isinstance(conc, str):
                    conc = float(conc.split('/')[0])
                expected_phen = 'U' if np.isnan(this_mic[drug].values[0]) else 'R' if all(
                    this_mic[drug] > conc) else 'S'
                today = this_sn['Phenotype']
                expected = expected_phen
                clsi = clsi_phens[sn][drug]
                if expected == 'U':
                    print(f'Strain {sn} does not have a known MIC for {drug}')
                elif today == 'X':
                    print(f'Strain {sn} does not have sufficient CFU data today to know phenotype')
                elif today != expected:
                    print(f'Strain {sn} appeared {today} today to {drug}-{conc}, but its MIC predicts {expected}.')
                else:
                    print(today, expected)
                strain_meta['Phenotype Expected'] = expected
                strain_meta['Phenotype CLSI'] = clsi
            else:
                strain_meta['Phenotype Expected'] = 'X'
                strain_meta['Phenotype CLSI'] = 'X'
        else:
            strain_meta = pd.DataFrame({col: ['None'] for col in dfs['Strain List']})
            strain_meta['Phenotype Expected'] = 'X'
            strain_meta['Phenotype CLSI'] = 'X'
            df.at[n, 'Strain ID'] = 'None'

        strain_meta['Folder'] = folder
        test_meta = strain_meta.merge(df.iloc[[n]], on='Strain ID', how='left')
        strain_info = pd.concat([strain_info, test_meta], ignore_index=True)
    response = None
    if save_to_meta:
        path_to_metadata = root + 'METADATA_file.xlsx'
        if os.path.exists(path_to_metadata):
            full_meta = pd.read_excel(path_to_metadata, dtype=object)
            print('\nMerging the metadata to the large METADATA_file.')
            merged, response = check_merge(full_meta, strain_info, match_on=['Folder', 'FileBase'], return_input=True)
        else:
            merged = strain_info.copy()
        merged.to_excel(path_to_metadata, index=False)
    if return_input:
        return strain_info, response
    else:
        return strain_info


def clean_data(raw, response_init=None, manual=False, degree_of_fit=None, return_all=False, **kwargs):
    """
    Cleans data by interpolating and removing noise.

    This function cleans data by interpolating and removing noise.

    Parameters:
    raw (np.ndarray): The raw data to clean.
    latres (int, optional): The lateral resolution of the data. Default is 1.
    remove (bool, optional): Whether to remove noise. Default is True.

    Returns:
    np.ndarray: The cleaned data.
    """
    global response, button_pressed, fig1, fig2
    radius = 1.3
    init = ''
    cleaned_data = {}
    cleaned_data['lows'] = fit_lows(raw, degree=degree_of_fit if degree_of_fit else 1, N=4)
    corners = get_corners(raw, minR_factor=radius)
    cleaned_data['corners'] = sym_reg(corners, degree=degree_of_fit if degree_of_fit else 2, normal=True,
                                      full_image=raw)  # Analysis of full inoculation images is not yet prepared.
    cleaned_data['strip'] = None  # Analysis of strip is not yet prepared.
    cleaned_data['full'] = sym_reg(raw, degree=degree_of_fit if degree_of_fit else 1, normal=True)
    I, J = cartesian(raw)
    edge = np.where(J > J.shape[1] - 125, raw,
                    np.nan)  # this should be generalized to find the 125 and/or find it on the left in function (find_edge)
    cleaned_data['edge'] = sym_reg(edge, degree=degree_of_fit if degree_of_fit else 1, normal=True, full_image=raw)

    if return_all: return cleaned_data
    if response_init is None:
        response = ''
        manual = True
    elif str(response_init).lower() not in ['full', 'lows', 'edge', 'corners', 'strip', 'all']:
        print('Please select a valid cleaning method through the manual mode.')
        manual = True
    elif str(response_init).lower() in ['full', 'lows', 'edge', 'corners', 'strip', 'all'] and manual:
        print(f'The {response_init} cleaning method was suggested. Please confirm on the figure.')
        init = response_init.lower()
    else:
        response = response_init.lower()
    if manual:
        colors = plt.get_cmap('tab20')
        button_pressed, axes = False, None

        def action(event):
            global button_pressed, response, fig1, fig2
            response = event.inaxes.get_children()[0].get_text().lower()
            if response != 'show':
                button_pressed = True
                if fig2 is not None:
                    plt.close(fig2)
                plt.close(fig1)
            else:
                fig2 = list(plot_all({'Cleaning Methods:': cleaned_data}).keys())[0]
                colors = plt.get_cmap('tab20')
                for n, ax in enumerate(fig2.get_axes()):
                    if n % 2 == 1:
                        continue
                    rect = Rectangle((-0.1, -0.1), 1.2, 1.2, transform=ax.transAxes, facecolor=colors(n), alpha=0.5,
                                     clip_on=False, zorder=-1)
                    ax.add_patch(rect)
                plt.pause(0.01)

        plot_contour(sym_reg(raw, degree=1))
        fig1 = plt.gcf()
        fig2 = None
        ax = plt.gca()
        plt.subplots_adjust(top=0.9)
        buttons = {i: [] for i in ['Full', 'Lows', 'Edge', 'Corners', 'Strip', 'Show']}
        num_buttons = len(buttons) - 1
        x_start, x_end = 0.05, 0.95
        button_height = 0.075
        width = (x_end - x_start) / num_buttons
        spacing = width / num_buttons
        button_width = width - spacing
        rect_z = 0
        for n, label in enumerate(buttons.keys()):
            if label == 'Show':
                x_position = 1 - button_width
                y_position = 0
                ax_button = plt.axes([x_position, y_position, button_width, button_height])
                button = Button(ax_button, label, color='gray', hovercolor='lightgray')
                button.on_clicked(action)
                buttons[label] = button
                continue
            x_position = x_start + spacing / 2 + n * width
            y_position = 0.925
            ax_button = plt.axes([x_position, y_position, button_width, button_height])
            color = colors(2 * n)
            edge = None
            if label == 'Edge':
                edge = Rectangle((raw.shape[1] - 125, 0), 125, raw.shape[0], facecolor='none', edgecolor=color,
                                 linewidth=5)
            elif label in ['Full', 'Lows']:
                edge = Rectangle((0, 0), *np.flip(raw.shape), facecolor='none', edgecolor=color, linewidth=5,
                                 linestyle=['-', '--'][rect_z])
                rect_z += 1
            elif label == 'Corners':
                edge = Ellipse((raw.shape[1] / 2, raw.shape[0] / 2), raw.shape[1] * radius, raw.shape[0] * radius,
                               facecolor='none', edgecolor=color, linewidth=5)
            elif label == 'Strip':
                edge = None  ## must still define this

            if edge is not None:
                ax.add_patch(edge)
            button = Button(ax_button, label, color=color, hovercolor=colors(2 * n + 1))
            button.on_clicked(action)
            buttons[label] = button
            if label.lower() == init:
                suggested_rect = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', linewidth=4,
                                           linestyle='--')
                ax_button.add_patch(suggested_rect)
                ax_button.text(0.5, 0, 'Suggested', transform=ax_button.transAxes, ha='center', va='top', fontsize=10,
                               color='black')

        nn = 1
        while not button_pressed:
            dn = 0.5
            plt.pause(dn)
            nn += 1
            if nn * dn >= 100:
                print('100 seconds have passed. I am continuing by assuming the edge only method.')
                response = 'edge'
                plt.close()
                break
        print(f'Cleaning Method: {response}')
    if cleaned_data[response] is None:
        print('Analysis of full inoculation images is not yet prepared.')
        return None
    elif response == 'strip':
        print('Analysis of strip is not yet prepared.')
        return None
    else:
        return cleaned_data[response]


def clean_df(df, data_column='Data-array', response_init=None, manual=False, degree_of_fit=1):
    """
    Cleans the specified data column in the DataFrame using the clean_data function.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to clean.
    data_column (str): The column name of the data to clean. Default is 'Data-array'.
    response_init (str, optional): Initial cleaning method suggestion. Default is None.
    manual (bool, optional): Whether to manually decide the cleaning method. Default is False.
    degree_of_fit (int, optional): Degree of polynomial fit for cleaning. Default is 1.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned data in the specified column.
    """
    if data_column not in df.columns:
        raise KeyError(f"'{data_column}' column not found in the DataFrame.")

    cleaned_data = []
    for data in tqdm(df[data_column]):
        cleaned_data.append(clean_data(data, response_init=response_init, manual=manual, degree_of_fit=degree_of_fit))

    df[f'Cleaned-{data_column}'] = cleaned_data
    return df


strain_max_len = 4


def assemble_data(folders, return_path=False, save_to_metadata=True, cleaning_mode=None, degree_of_fit=1):
    """
    Assembles and cleans data from specified folders.

    This function assembles and cleans data from specified folders, processes the data, and optionally saves it to a metadata file.
    It also merges the extracted data with a master list if provided.

    Parameters:
    folders (str or list): The folder(s) from which to assemble data. If a string is provided, it will be converted to a list.
    return_path (bool, optional): Whether to return the path to the root directory. Default is False.
    save_to_metadata (bool, optional): Whether to save the assembled data to a metadata file. Default is True.
    manual_decision (bool, optional): Whether to manually decide how to process certain data. Default is False.

    Returns:
    pd.DataFrame: The assembled and cleaned data.
    str (optional): The path to the root directory if return_path is True.
    """
    if isinstance(folders, str):
        folders = [folders]
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    metadata_file = root + 'METADATA_file.xlsx'
    master_list = pd.read_excel(root + 'Strain Master List.xlsx', sheet_name=None)
    root = root + 'Interferometer/'

    df = pd.DataFrame()
    for folder in folders:
        folder = folder + ('/' if not folder.endswith('/') else '')
        working_folder = root + folder
        print(f'\nExtracting metadata from {folder}.')
        new_meta = pd.DataFrame()
        metadata, response = extract_metadata(folder, save_to_meta=save_to_metadata, master_list=master_list,
                                              return_input=True)
        metadata['Raw-array'] = [[] for i in range(metadata.shape[0])]
        metadata['Data-array'] = [[] for i in range(metadata.shape[0])]
        metadata['Replicate'] = [0 for i in range(metadata.shape[0])]
        metadata['Clean Method'] = [None for i in range(metadata.shape[0])]
        data = {}
        if os.path.exists(working_folder + 'features.pkl') and not response:
            data = read_dataframe(working_folder + 'features.pkl')
            df = pd.concat(
                [df.reset_index(drop=True), data.drop(columns=[i for i in data.columns if i.startswith('f')])],
                ignore_index=True)
            for sn, expected, drug, today, rep in df[
                ['Strain ID', 'Phenotype Expected', 'Drug', 'Phenotype', 'Replicate']].values:
                if rep > 1: continue
                if expected == 'U':
                    print(f'Strain {sn} does not have a known MIC for {drug}')
                elif today == 'X':
                    print(f'Strain {sn} does not have sufficient CFU data today to know phenotype')
                elif today != expected:
                    print(f'Strain {sn} appeared {today} today to {drug}, but its MIC predicts {expected}.')
            continue
        elif os.path.exists(working_folder + 'data.pkl') and not response:
            print('Collecting previously acquired and cleaned data.')
            data = read_dataframe(working_folder + 'data.pkl')
            df = pd.concat([df, meta], ignore_index=True)
            continue
        elif response:
            print('Cleaning data and connecting to the metadata.')
            for file in tqdm(os.listdir(working_folder)):
                if not file.endswith('datx'): continue
                if '_' in file:
                    base = '_'.join(file[:-5].split('_')[:-1])
                    rep = file[:-5].split('_')[-1]
                else:
                    base = file[:-5]
                    rep = ''
                rep = int(rep) if rep.isdigit() else (
                    ((df['FileBase'] == base) & (df['Folder'] == folder)).sum() + 1 if df.columns.isin(
                        ['FileBase', 'Folder']).sum() > 0 else 1)
                base_mask = [base == str(int(b)) if isinstance(b, float) and b.is_integer() else base == str(b) for b in
                             metadata['FileBase']]
                fold_mask = metadata['Folder'].values == folder
                meta = metadata.loc[fold_mask & base_mask].copy()  ###this is assuming the base is a number..
                if meta.empty:
                    print(f'FileBase {base} not found in the metadata of that folder.\nMoving on to the next file.')
                    print(f'Trying {base}_{rep}.')
                    base_mask = ['_'.join([str(base), str(rep)]) == b for b in metadata['FileBase']]
                    meta = metadata.loc[fold_mask & base_mask].copy()  ###this is assuming the base is a number..
                    if meta.empty:
                        print(f'Filebase {base}_{rep} still failed. Moving on to next file.')
                        continue
                latres = None
                fullraw = None
                if file[:-5] not in data:
                    values = convert_data(working_folder + file, resolution=True, remove=False)
                    raw = values['Heights']
                    latres = values['Resolution']
                    raw /= 1000  # convert height to um
                    if np.isnan(raw).sum() > 0:
                        print(f'{file} has not been interpolated. This could cause issues down the line.')
                    region = meta['Region'].values[0]
                    cfu = meta['Plated CFU'].values[0]
                    inctime = meta['Incubation'].values[0]
                    media = meta['Media'].values[0]

                    manual_decision = False
                    if isinstance(cfu, str) and not region == 'Agar':
                        cfu = 1e10
                        manual_decision = True
                    if cleaning_mode is not None:
                        if cleaning_mode.lower() == 'manual':
                            manual_decision = True
                            response = None
                        else:
                            manual_decision = False
                            response = cleaning_mode.lower()
                    elif region == 'Coffee Ring':
                        response = 'edge'
                    elif region == 'Homeland':
                        response = 'full'
                    elif region == 'Full':
                        response = 'corners'
                    elif region == 'Strip':
                        response = 'strip'
                    elif region == 'Agar':
                        response = 'full'
                    else:
                        response = None
                    if cleaning_mode is None and response != 'full' and ((cfu <= 1e4 and inctime <= 4) or (
                            inctime < 2 and media in ['Urine', 'PBS', 'None'])): response = 'lows'

                    c = clean_data(raw, response_init=response, manual=manual_decision, degree_of_fit=degree_of_fit)
                    cleaning_method = response
                    data[file[:-5]] = c
                else:
                    c = data[file[:-5]]
                    raw = None
                    cleaning_method = None  ##change this
                    print('No Raw.')
                if latres is None:
                    raw_values = convert_data(working_folder + file, resolution=True, remove=False)
                    raw = values['Heights']
                    latres = values['Resolution']
                raw_fold = ([os.path.join(working_folder, folder) + '/' for folder in os.listdir(working_folder) if
                             'raw' in folder.lower() and os.path.isdir(os.path.join(working_folder, folder))] + [None])[
                    0]
                if raw_fold and os.path.exists(raw_fold + file):
                    fullraw = convert_data(raw_fold + file, remove=False)['Heights']
                meta['Clean Method'] = cleaning_method
                meta['Lateral Resolution'] = latres
                meta['Data-array'] = [c]
                meta['Raw-array'] = [fullraw] if fullraw is not None else [raw]
                meta['Replicate'] = int(rep)
                meta['Roll'] = values['Roll']
                meta['Pitch'] = values['Pitch']
                new_meta = pd.concat([new_meta, meta])
            df = pd.concat([df, new_meta], ignore_index=True)
            save_dataframe(working_folder + 'data.pkl', new_meta)
        else:
            print('Something went terribly wrong!')
    if return_path:
        return df, root
    else:
        return df


def extract_features(df, gather_ffts=True, plot_ffts=False):
    """
    Extracts features from the data arrays in the DataFrame.

    This function extracts features from the data arrays in the DataFrame using the get_features function and optionally gathers FFT features using get_all_fft_features.
    It also optionally plots the FFT features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data arrays to extract features from. Must include a "Data-array" column.
    gather_ffts (bool, optional): Whether to gather FFT features. Default is True.
    plot_ffts (bool, optional): Whether to plot the FFT features. Default is False.

    Returns:
    pd.DataFrame: The DataFrame with the extracted features.
    """
    if 'Data-array' not in df.columns:
        raise ValueError('Data not in DataFrame.')

    feat_df = pd.DataFrame()
    for n, (data, obj, zoom) in enumerate(tqdm(df[['Data-array', 'Objective', 'Zoom']].values)):
        obj = float(obj[:-1])
        zoom = float(zoom[:-1])
        feats = {}

        height_feats = get_features(data, obj=obj, zoom=zoom, fftval=None)
        feats = feats | height_feats

        if gather_ffts:
            fft_feats = get_all_fft_features(data, obj=obj, zoom=zoom, wavelengths='all', plot=plot_ffts, c=None,
                                             alpha=0.2)
            feats = feats | fft_feats

        this_image = pd.concat([df.iloc[[n]].reset_index(drop=True), pd.DataFrame([feats])], axis=1)
        feat_df = pd.concat([feat_df, this_image], ignore_index=True, axis=0)

    return feat_df


def analyze_new_data(folders, ffts=True, save_to_metadata=True, save_to_allfeatures=False, features=False,
                     cleaning_mode=None, degree_of_fit=None):
    """
    Assembles, cleans, and analyzes new data from specified folders.

    This function assembles and cleans data from specified folders, extracts features from the data, and optionally saves the results to metadata and all features files.
    It also merges the extracted data with a master list if provided.

    Parameters:
    folders (str or list): The folder(s) from which to assemble and analyze data. If a string is provided, it will be converted to a list.
    ffts (bool, optional): Whether to gather FFT features during feature extraction. Default is True.
    save_to_metadata (bool, optional): Whether to save the assembled data to a metadata file. Default is True.
    save_to_allfeatures (bool, optional): Whether to save the extracted features to an all features file. Default is False.
    features (bool, optional): Whether to extract features from the data. Default is True.

    Returns:
    pd.DataFrame: The DataFrame with the assembled and analyzed data.
    """
    print('Assembling and Cleaning Data.')
    df, root = assemble_data(folders, return_path=True, save_to_metadata=save_to_metadata, cleaning_mode=cleaning_mode,
                             degree_of_fit=degree_of_fit)

    if features:
        print('\n\nMoving on to feature extraction.')
        feat_df = pd.DataFrame()
        for folder in np.unique(df['Folder']):
            path = root + folder + 'features.pkl'
            data_path = root + folder + 'data.pkl'

            if os.path.exists(path):
                response = input(
                    f'\nWARNING: {folder} already has features.\nDo you wish to overwrite? (y/[n]): ').strip().lower() == 'y'
            else:
                response = True

            if response:
                print(f'\nExtracting Features from {folder} data.')
                this_df = df.loc[df['Folder'] == folder].copy()
                this_feat_df = extract_features(this_df, gather_ffts=ffts)
                save_dataframe(path, this_feat_df)
                if os.path.exists(data_path):
                    os.remove(data_path)
            else:
                this_feat_df = read_dataframe(path)
                print('Features not overwritten.')

            feat_df = pd.concat([feat_df, this_feat_df], ignore_index=True)
    else:
        feat_df = df.copy()

    # if save_to_allfeatures and features:
    #    print('\n\nAdding features to All Features file')
    #    path_to_full = '/'.join(root.split('/')[:-1]) + '/allfeatures.pkl'
    #    if os.path.exists(path_to_full):
    #        full_df = read_dataframe(path_to_full)
    #        merged = check_merge(full_df, feat_df, match_on = ['Folder','FileBase','Replicate'])
    #    else:
    #        merged = feat_df.copy()
    #    save_dataframe(path_to_full, merged)
    return feat_df


def update_data(folder):
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/Interferometer/'
    path = root + folder + ('/' if not folder.endswith('/') else '')

    # Extract metadata
    metadata = extract_metadata(folder, save_to_meta=True)

    # Determine which file to update
    data_path = path + 'data.pkl'
    features_path = path + 'features.pkl'

    if os.path.exists(features_path):
        file_path = features_path
    elif os.path.exists(data_path):
        file_path = data_path
    else:
        raise FileNotFoundError("Neither 'data.pkl' nor 'features.pkl' exists in the specified folder.")

    # Load the existing data
    existing_data = read_dataframe(file_path)

    # Merge metadata with the existing data
    updated_data = check_merge(existing_data, metadata, match_on=['Folder', 'FileBase'])

    # Save the updated data back to the file
    save_dataframe(file_path, updated_data)

    return updated_data


def collect_data(dataframe=None, print_keys=False, **kwargs):
    """
    Collects and filters data based on specified criteria.

    This function collects data from the metadata and features files, filters the data based on the specified criteria, and returns the matching data.

    Parameters:
    print_keys (bool): Whether to return all possible keys and their values.
    **kwargs: Arbitrary keyword arguments specifying the criteria for filtering the data replacing spaces with _. The keys should correspond to column names in the metadata file, and the values should be the desired values for filtering.

    Returns:
    pd.DataFrame: The DataFrame with the matching data.
    """
    kwargs = {i.replace('_', ' '): vals for i, vals in kwargs.items()}
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    path_to_metadata = root + 'METADATA_file.xlsx'
    print('Pulling all metadata.')
    if dataframe is None:
        meta = pd.read_excel(path_to_metadata)
    else:
        meta = dataframe.copy()

    if print_keys:
        possible_keys = {col: meta[col].unique() for col in meta.columns}
        return possible_keys

    mask = pd.Series(True, index=meta.index)  # Start with a mask of all True values

    print('Extracting only requested data.\n')
    cols = {i.lower(): i for i in meta.columns}
    for col, values in kwargs.items():
        if values is None: continue
        if col.endswith(' '):
            mode = 'contains'
            col = col[:-1]
        elif col.startswith(' '):
            mode = 'excludes'
            col = col[1:]
        else:
            mode = 'matches'
        if col.lower() in cols.keys():  # Ensure column exists in the DataFrame
            col = cols[col.lower()]
            meta.fillna({col: 'None'}, inplace=True)
            if isinstance(values, (str, int, float)):  # Prevent strings from being treated as iterables
                values = [values]
            col_types = meta[col].map(type)
            col_type = str if any(col_types == str) else col_types.iloc[0]
            values = [col_type(val if type(val) != int else float(val)) for val in values]
            if mode == 'matches':
                hold = meta[col].astype(col_type).isin(values)
            elif mode == 'contains':
                hold = np.array([any([str(value) in str(data) for value in values]) for data in meta[col]])
            elif mode == 'excludes':
                hold = np.array([all([str(value) not in str(data) for value in values]) for data in meta[col]])
            print(f'{col} - {mode}')
            mask &= hold  # Apply filtering condition
            print(hold.sum(), mask.sum())
        else:
            print(f'{col} is not in the metadata.')
    collect = meta.loc[mask]

    if dataframe is not None:
        return collect
    if collect.empty:
        print('No data found matching the specified criteria.')
        return None

    print('\nPulling features')
    features_list = []
    for folder in collect['Folder'].unique():
        feature_path = root + f'Interferometer/{folder}/features.pkl'
        data_path = root + f'Interferometer/{folder}/data.pkl'
        if os.path.exists(feature_path):
            features = read_dataframe(feature_path)
        elif os.path.exists(data_path):
            print(f'Features not found for {folder}. Using data instead.')
            features = read_dataframe(data_path)
        else:
            print(f'No data found in {folder}')
            continue
        filebases = collect.loc[collect['Folder'] == folder, 'FileBase']
        matching_features = features[features['FileBase'].isin(filebases)]
        features_list.append(matching_features)

    print('Matching features with desired qualities')
    matching = pd.concat(features_list, ignore_index=True)

    return matching


def functions(name):
    lim = 0.5

    def log(data, latres=1):
        return np.log10(data)

    def volume(array, latres=1):
        return np.nansum(array) * (latres * 1e6) ** 2

    def kurtosis(array, latres=1):
        mask = (array >= -lim) & (array <= lim)
        filtered_data = array[mask]
        return scipy.stats.kurtosis(filtered_data, nan_policy='omit')

    def skewness(array, latres=1):
        mask = (array >= -lim) & (array <= lim)
        filtered_data = array[mask]
        return scipy.stats.skew(filtered_data, nan_policy='omit')

    def ra(array, latres=1):
        return np.nanmean(np.abs(array))

    def rz(array, latres=1):
        return np.nanmax(array) - np.nanmin(array)

    def rq(array, latres=1):
        return np.sqrt(np.nanmean(array ** 2))

    def max_slope(array, latres=1e-3):
        dx, dy = np.gradient(array, latres * 1e3)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        return np.nanmax(np.arctan(slope) * 180 / np.pi)

    def mean_slope(array, latres=1e-3):
        dx, dy = np.gradient(array, latres * 1e3)
        slope = np.sqrt(dx ** 2 + dy ** 2)
        return np.nanmean(np.arctan(slope) * 180 / np.pi)

    def power(array, latres=1e-6):
        return power_spectrum(array, dk=2, resolution=latres * 1e-3, plot=False, window=True)

    def calculate_slope(array, x=None, latres=1e-3):
        '''Gives slope in [input]/mm'''
        if np.sum(np.isnan(array)) > 0:
            return None  # Return a default value if the array contains NaN values
        if len(array.shape) == 1:  # Handle 1D array
            if x is None:
                x = np.arange(len(array))
            slope, _ = np.polyfit(x, array, 1)
            return slope / (latres * 1e3)  # per mm
        elif len(array.shape) == 2:  # Handle 2D array
            if x is None:
                x = np.arange(array.shape[1])
                y = np.arange(array.shape[0])
                X, Y = np.meshgrid(x, y)
            else:
                X, Y = x
            d = array
            A = np.c_[X.ravel(), Y.ravel(), np.ones(d.size)]
            C, _, _, _ = scipy.linalg.lstsq(A, d.ravel())
            slope_x, slope_y = C[0], C[1]
            combined_slope = np.sqrt(slope_x ** 2 + slope_y ** 2)
            return combined_slope / (latres * 1e3)  # per mm
        else:
            raise ValueError("Input array must be 1D or 2D.")

    def power_slope(frequencies, amplitude, lims=None, latres=None):
        if lims is None:
            lims = [min(frequencies), max(frequencies)]
        if latres is not None:
            frequencies = latres / frequencies
        mask = (frequencies >= lims[0]) & (frequencies <= lims[1])
        if mask.sum() < 2:
            raise ValueError("Not enough data points within the specified range to calculate the slope.")
        log_frequencies = np.log10(frequencies[mask])
        log_amplitude = np.log10(amplitude[mask])
        slope, _ = np.polyfit(log_frequencies, log_amplitude, 1)
        return slope

    if name.lower() == 'volume':
        return volume
    elif name.lower().startswith('kurt'):
        return kurtosis
    elif name.lower().startswith('skew'):
        return skewness
    elif name.lower() == 'ra':
        return ra
    elif name.lower() == 'rz':
        return rz
    elif name.lower() == 'rq':
        return rq
    elif name.lower() == 'max_slope':
        return max_slope
    elif name.lower() == 'avg_slope':
        return mean_slope
    elif name.lower() == 'power':
        return power
    elif name.lower() == 'fit_slope':
        return calculate_slope
    elif name.lower() == 'power_slope':
        return power_slope
    elif name.lower() == 'log':
        return log
    else:
        return None


def plot_features(df, y_data='volume', x_data='Incubation', normalize=False, scale=False, color=None, markers=None,
                  plot='both', plot_type='auto', reorder=None):
    """
    Plots features from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    y_data (str or list): The column name(s) for the y-axis data. Can be a single column or a list of columns.
    x_data (str): The column name for the x-axis data.
    normalize (bool): Whether to normalize the y-axis data. Default is False.
    color (str): The column name for coloring the plots. Default is None.
    plot (str): The type of plot to create. Can be 'both', 'replicates', or 'averages'. Default is 'both'.
    """
    if normalize and scale:
        resp = input('Normalize and Scale cannot be used together. Pick one.')
        if resp.lower().strip().startswith('n'):
            scale = False
        else:
            normalize = False

    if isinstance(y_data, str):
        y_data = [y_data]

    columns = []
    for column in [x_data] + y_data:
        while True:
            if column not in df.columns:
                if column.lower().strip() == 'exit':
                    return None
                calculation_func = functions(column)
                if calculation_func is None:
                    column = input('That function does not exist, please try again: ')
                    continue
                df[column] = df.apply(lambda row: calculation_func(row['Data-array'], row['Lateral Resolution']),
                                      axis=1)
            break
        columns.append(column)

    do_treatment = x_data == 'Treatment'
    grouped = df.groupby(
        ['Folder', 'Strain ID'] + (['Treatment'] if not do_treatment else []) + ([color] if color else []))

    cmap = plt.get_cmap('tab20')
    if color and color in df.columns:
        types = df[color].map(type)
        this_type = str if any(types == str) else df[color].dtype.type
        u_vals = np.unique(df[color].astype(this_type)).tolist()
        colors = {df.loc[df[color].map(this_type) == val, color].map(type).values[0](val): u_vals.index(val) for val in
                  u_vals}
        # u_vals = np.unique(df[color]).tolist()
    # colors = {val: u_vals.index(val) for val in u_vals}
    else:
        colors = None
    nx, ny = define_subplot_size(len(y_data))
    fig, axes = plt.subplots(nx, ny, figsize=(5 * ny, 4 * nx), sharex=True)
    if len(y_data) == 1:
        axes = [axes]
    axes = np.array(axes)
    colored, reps, avgs = {}, {}, {}
    for ax, y_col in zip(axes.ravel(), y_data):
        types = df[columns[0]].map(type)
        this_type = str if any(types == str) else df[columns[0]].dtype.type
        x_values = np.unique(df[columns[0]].astype(this_type)).tolist()

        if normalize or scale:
            val = np.array([normalize, scale])[np.bool_([normalize, scale])].tolist()
            if val and val[0] in x_values:
                idx = x_values.index(val[0])
                x_values = val + x_values[:idx] + (x_values[idx + 1:] if idx < len(x_values) - 1 else [])
        ax.set_title(f'{y_col} vs {x_data}')
        ax.set_xlabel(x_data)
        ax.set_ylabel(y_col)
        colored[y_col], reps[y_col], avgs[y_col] = [], {}, {}

        for n, (group_info, group) in enumerate(grouped):
            folder = group_info[0]
            strain_id = group_info[1]
            if not do_treatment:
                treatment = group_info[2]
            else:
                treatment = ''
            if colors:
                c = colors[group[color].iloc[0]] * 2
                label = group[color].iloc[0]
            else:
                c = n * 2
                label = f'{strain_id}+{treatment.split(" ")[0]}'
            if c in colored[y_col]:
                legend = False
            else:
                colored[y_col].append(c)
                legend = True
            if c not in reps[y_col]:
                reps[y_col][c] = []
                avgs[y_col][c] = []

            max_n = group['Replicate'].nunique() - 1
            for n_rep, (replicate, rep_group) in enumerate(group.groupby('Replicate')):
                x_init = rep_group[columns[0]].values
                x = np.array([x_values.index(this_type(val)) for val in x_init])
                y = rep_group[y_col].values
                args = np.argsort(x)
                if reorder:
                    if reorder in x_init:
                        ind = x_values.index(this_type(reorder))
                        args = args.tolist()
                        args.pop(ind)
                        args = [ind] + args
                    else:
                        args = args[reorder]

                x = x[args]
                y = y[args]
                if normalize:
                    if type(normalize) is not bool and normalize in x_init:
                        y -= y[x_init[args] == normalize][0]
                    else:
                        y -= y[0]
                elif scale:
                    if type(scale) is not bool and scale in x_init:
                        y /= y[x_init[args] == scale][0]
                    else:
                        y /= y[0]

                plot_colors = cmap(c)

                if markers and markers in df.columns:
                    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
                    unique_markers = df[markers].unique()
                    marker_map = {val: marker_styles[i % len(marker_styles)] for i, val in enumerate(unique_markers)}
                    marker_style = marker_map[rep_group[markers].iloc[0]]
                else:
                    marker_style = 'o'

                if x_data in ['Notes'] or plot_type == 'scatter':
                    reps[y_col][c].append(
                        ax.scatter(x, y, label=label if legend and (plot == 'replicates' or max_n == 0) else None,
                                   marker=marker_style, color=plot_colors))
                else:
                    reps[y_col][c].append(
                        ax.plot(x, y, label=label if legend and (plot == 'replicates' or max_n == 0) else None,
                                marker=marker_style, color=plot_colors))
                if plot == 'replicates':
                    legend = False
                if plot not in ['both', 'replicates']:
                    reps[y_col][c][-1][0].set_visible(False)

            if group[columns[1]].dtype in (int, float) and all(group.groupby(columns[0])[columns[1]].count() > 1):
                avg_group = group.groupby(columns[0]).agg({y_col: 'mean'}).reset_index()
                avg_color = cmap(c + int(plot != 'averages'))
                x_init = avg_group[columns[0]].values
                x = np.array([x_values.index(this_type(val)) for val in x_init])
                y = avg_group[y_col].values
                args = np.argsort(x)
                if reorder:
                    args = args[reorder]

                x = x[args]
                y = y[args]
                if normalize:
                    if type(normalize) is not bool and normalize in x_init:
                        y -= y[x_init[args] == normalize][0]
                    else:
                        y -= y[0]
                elif scale:
                    if type(scale) is not bool and scale in x_init:
                        y /= y[x_init[args] == scale][0]
                    else:
                        y /= y[0]

                # x = avg_group[columns[0]].values
                # y = avg_group[y_col].values
                # args = np.argsort([x_values.index(this_type(val)) for val in x])
                # x = x[args]
                # y = y[args]
                # if normalize:
                #    y -= y[0]
                # elif scale:
                #    y /= y[0]
                if plot == 'averages':
                    std_group = group.groupby(columns[0]).agg({y_col: 'std'}).reset_index()
                    yerr = std_group[y_col].values[args]
                    avgs[y_col][c].append(
                        ax.errorbar(x, y, yerr, label=f'{strain_id}+{treatment.split(" ")[0]}', marker='x',
                                    linestyle='--', color=avg_color))
                else:
                    label = None
                    avgs[y_col][c].append(
                        ax.plot(x, y, label=label if legend else None, marker='x', linestyle='--', color=avg_color))
            if plot not in ['both', 'averages']:
                for avg in avgs[y_col][c]:
                    avg[-1][0].set_visible(False)

        ax.legend()

        # Rotate x-axis tick labels if x values are strings and set scales to log

        if df[columns[0]].dtype == object:
            if reorder and False:
                if reorder in x_values:
                    ind = x_values.index(reorder)
                    x_values.pop(ind)
                    x_values = [reorder] + x_values
                else:
                    x_values = [x_values[i] for i in reorder]
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_values, rotation=45, ha='right', fontsize=10)
        elif np.ptp(np.log10(df[x_data].values)) > 2:
            ax.set_xscale('log')
        if np.ptp(df[y_col].map(np.log10).values) > 2:
            ax.set_yscale('log')

    # plt.tight_layout()
    plt.pause(0.01)
    return axes


def make_sub_images(arr, num_images=1):
    if isinstance(num_images, (int, float)):
        num_images = [num_images, num_images]
    elif not isinstance(num_images, list):
        num_images = list(num_images)
        if len(num_images) < len(arr.shape):
            num_images = [num_images[0] for i in arr.shape]
    return [subsub for sub in np.array_split(arr, num_images[0], axis=0) for subsub in
            np.array_split(sub, num_images[1], axis=1)]


def plot_hist(df, data='volume', data_column='Data-array', num_images=None, color=None, lims=[None, None],
              num_bins=None, bin_width=1, plot_together=False, split_treatment=True):
    """
    Plots histograms from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    data (str): The column name for the data to plot.
    color (str): The column name for coloring the plots. Default is None.
    """

    # plt.close('all')
    column = data
    df.reset_index(drop=True, inplace=True)
    while True:
        calculation_func = functions(column)
        if calculation_func:  # column not in df.columns:
            if calculation_func is None:
                column = input('That function does not exist, please try again (or type "exit" to quit): ')
                if column.lower().strip() == 'exit':
                    return None
                continue
            if data_column == 'Data-array':
                if num_images:
                    df[column] = df.apply(
                        lambda row: [calculation_func(sub_image, row['Lateral Resolution']) for sub_image in
                                     make_sub_images(row['Data-array'], num_images)], axis=1)
                else:
                    df[column] = df.apply(lambda row: row[data_column].ravel(), axis=1)
            else:
                df[column] = df.apply(lambda row: calculation_func(row[data_column], row['Lateral Resolution']), axis=1)
        break
    df.fillna('None', inplace=True)
    group_cols = ['Folder', 'Strain ID', 'Dilution', 'Treatment', 'Replicate', 'Incubation']
    for col in group_cols:
        if col not in df.columns:
            df[col] = ['' for i in range(df.shape[0])]
    grouped = df.groupby(['Folder', 'Strain ID', 'Dilution'] + (['Treatment'] if split_treatment else []))
    cmap = plt.get_cmap('tab20')
    if color and color in df.columns:
        u_vals = np.unique(df[color]).tolist()
        colors = {val: u_vals.index(val) for val in u_vals}
    else:
        colors = None
    # plt.figure()
    min_val = lims[0] if lims[0] else np.nanmin([np.nanmin(row) for row in df[column].values])
    max_val = lims[1] if lims[1] else np.nanmax([np.nanmax(row) for row in df[column].values])
    rang = max_val - min_val
    if not num_bins:
        num_bins = int(rang * 1000 / bin_width)  # int(np.sqrt(df.loc[0,column].size))
    bins = np.linspace(min_val - 0.02 * rang, max_val + 0.02 * rang, num_bins)
    if plot_together:
        plt.figure()
        colored = []
    for n, (group_info, group) in enumerate(grouped):
        folder = group_info[0]
        strain_id = group_info[1]
        dilution = group_info[2]
        if split_treatment:
            treatment = group_info[3]
        else:
            treatment = ''
        for n_r, (replicate, rep_group) in enumerate(group.groupby('Replicate')):
            if not plot_together:
                fig_name = str((strain_id + f'-{dilution}logs', replicate, treatment))
                if plt.fignum_exists(fig_name):
                    plt.close(fig_name)
                plt.figure(fig_name)
            if not plot_together: colored = []
            for ni, (incubation, inc_group) in enumerate(rep_group.groupby('Incubation')):
                if colors:
                    c = lambda n: colors[inc_group[color].iloc[n]]
                    label = lambda n: inc_group[color].iloc[n]
                else:
                    c = lambda n: ni
                    label = lambda n: f'{incubation}hrs'
                y = inc_group[column].values
                for n, yi in enumerate(y):
                    if c(n) in colored:
                        legend = False
                    else:
                        colored.append(c(n))
                        legend = True
                    plot_color = cmap(c(n) * 2)
                    plt.hist(yi, bins=bins, density=False, histtype='step', label=label(n) if legend else None,
                             color=plot_color)
                    legend = False

            plt.title(f'Histogram of {data}')
            plt.xlabel(data)
            plt.ylabel('Count')
            plt.legend(title=color if color else '')
            plt.pause(0.01)


def plot_power(df, color=None, split=None, plot_fit=True, sort=False, wave_lims=[20, 100]):
    """
    Plots arrays from the DataFrame on the same graph with color separating images.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    color (str or list): The column name(s) for coloring the plots. Default is None.
    split (str or list): The column(s) to split the data into subplots. Default is None.
    plot_fit (bool): Whether to plot the fitted lines. Default is True.
    sort (bool): Whether to sort the groups. Default is False.
    """

    df.drop(columns=['power'], inplace=True, errors='ignore')
    df['power'] = [functions('power')(row['Data-array'], row['Lateral Resolution']) for n, row in
                   tqdm(df.iterrows(), total=len(df))]

    df.reset_index(drop=True, inplace=True)
    df.fillna('None', inplace=True)

    cmap = plt.get_cmap('tab20')
    if color:
        if isinstance(color, str):
            color = [color]
        if all(col in df.columns for col in color):
            color_values = df[color].astype(str).agg('-'.join, axis=1)
            unique_colors = sorted(color_values.unique())
            colors = {val: idx for idx, val in enumerate(unique_colors)}
        else:
            raise ValueError("One or more columns specified in 'color' are not in the DataFrame.")
    else:
        colors = None

    group_columns = ['Strain ID', 'Folder', 'Dilution', 'Media', 'Region', 'Treatment', 'Inoculated Volume', 'Drug',
                     'Drug Concentration', 'Lateral Resolution', 'Imager', 'Notes']
    group_columns = [col for col in group_columns if col not in [color, []]]

    split_groups = df.groupby(split) if split else [(None, df)]

    for split_key, split_group in split_groups:
        fignum = f'Power Spectrums - {split_key}' if split else 'Power Spectrums'
        if plt.fignum_exists(fignum):
            plt.close(fignum)

        if sort:
            grouped = split_group.groupby(group_columns)
            group_keys = np.array(list(grouped.groups.keys())).T
            non_unique_cols = [n for n, cols in enumerate(group_keys) if len(np.unique(cols)) > 1]
        else:
            grouped = [(None, split_group)]
            non_unique_cols = []
        fig = plt.figure(fignum, figsize=(10, 8))  # figsize is (width, height) in inches
        subplot_size = define_subplot_size(len(grouped))
        axes = fig.subplots(*subplot_size, sharex=True, sharey=True)
        fig.name = fignum

        if len(grouped) == 1:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = axes.reshape(subplot_size)

        for ax in axes.ravel():
            ax.invert_xaxis()

        fig.supxlabel('Wavelength (um)')
        fig.supylabel('Power Spectrum Amplitude')

        for n_ax, (ax, (group_key, group_df)) in enumerate(zip(axes.ravel(), grouped)):
            ax.set_title(
                ', '.join([str(group_key[col]) for col in non_unique_cols]) if len(non_unique_cols) > 0 else None,
                fontsize=12)

            # if axes.size == 0 or n_ax ==0:#% axes.shape[0] == axes.shape[0] - 1:
            # print(n_ax)
            #    ax.invert_xaxis()  # Flip the x-axis for visual consistency

            colored = []
            fits = {}
            for n, row in group_df.iterrows():
                if colors:
                    color_key = color_values.iloc[n]
                    c = colors[color_key] * 2 + int(plot_fit)
                else:
                    c = n * 2 + int(plot_fit)

                plot_color = cmap(c)
                x = row['power'][0]
                y = row['power'][1]
                wavelength = row['Lateral Resolution'] * 1e6 / x  # Adjust x-axis to be latres/x

                # Collect data for fitting
                if color_key not in fits:
                    fits[color_key] = {'wavelength': [], 'y': []}
                mask = (wavelength >= wave_lims[0]) & (wavelength <= wave_lims[1])
                fits[color_key]['wavelength'].extend(wavelength[mask])
                fits[color_key]['y'].extend(y[mask])

                label = f"{group_key} - {color_key}" if colors and c not in colored and not plot_fit else None
                if label:
                    colored.append(c)
                ax.loglog(wavelength, y, label=label, color=plot_color)

            # Fit and plot lines for each color
            for n, (key, data) in enumerate(fits.items()):
                if len(data['wavelength']) > 1 and plot_fit:
                    if colors:
                        c = colors[key] * 2
                    else:
                        c = n * 2

                    plot_color = cmap(c)
                    log_wavelength = np.log10(data['wavelength'])
                    log_y = np.log10(data['y'])
                    slope, intercept = np.polyfit(log_wavelength, log_y, 1)
                    fit_line = 10 ** (slope * log_wavelength + intercept)

                    label = f'{key}->{np.round(slope, 2)}' if colors and c not in colored and not plot_fit else None
                    if label:
                        colored.append(c)
                    ax.loglog(data['wavelength'], fit_line, label=f'{key}->{np.round(slope, 2)}', color=plot_color)

            handles, labels = ax.get_legend_handles_labels()
            sorted_labels_handles = sorted(zip(labels, handles), key=lambda x: x[0])
            sorted_labels, sorted_handles = zip(*sorted_labels_handles)
            ax.legend(sorted_handles, sorted_labels, title='-'.join(color) if color else '')
            plt.pause(0.05)


def plot_subimages(df, num_images=2, data_column='Data-array', value_column='volume', x_column='Incubation'):
    """
    Plots a 2D grid of sub-images from the data arrays in the DataFrame.

    This function plots a 2D grid of sub-images from the data arrays in the DataFrame.
    Each axis in the plots is clickable, and clicking on an axis displays the value for a specified column in a scatter plot in a different figure.
    Clicking on a new axis updates the corresponding point on the scatter plot.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data arrays to plot.
    num_images (int): The number of sub-images to create along each axis. Default is 4.
    data_column (str): The column name for the data arrays. Default is 'Data-array'.
    value_column (str): The column name for the values to display on the scatter plot. Default is 'volume'.
    x_column (str): The column name for the x-axis values in the scatter plot. Default is 'Incubation'.
    """
    global scatter_fig, scatter_ax, scatter_points, highlighted_point, highlighted_line, final_df
    scatter_fig, scatter_ax, scatter_points, highlighted_point, highlighted_line, final_df = None, None, None, {}, None, None
    df = df.fillna('None')
    plt.close('all')
    # Separate df by all metadata except for time
    metadata_columns = ['Folder', 'Strain ID', 'Treatment', 'Replicate', 'Dilution', 'Inoculated Volume', 'Objective']
    # metadata_columns = [col for col in df.columns if not col.startswith('f') and '-array' not in col and col != x_column and col!='FileBase']
    unique_combinations = df[metadata_columns].drop_duplicates().reset_index(drop=True)

    # Provide the different folders, treatments, and antibiotic combinations that exist in the df
    print("Available combinations:")
    for idx, row in unique_combinations.iterrows():
        print(f"{idx}: {row.to_dict()}")

    # User selects the combination to plot
    selected_idx = int(input("Select the combination index to plot: "))
    selected_combination = unique_combinations.iloc[selected_idx].to_dict()
    # Filter the df based on the selected combination
    # print('\n\n\n')
    ddf = df.shape[0]
    for col, val in selected_combination.items():
        df = df[df[col] == val]
        if df.shape[0] < ddf:
            # print(col, df.shape, val)
            ddf = df.shape[0]

    def on_click(event):
        global scatter_fig, scatter_ax, scatter_points, highlighted_point, highlighted_line
        global final_df
        for fig, axs in axes.items():
            check = event.inaxes == axs.ravel()
            if any(check):
                for ax in axs.ravel():
                    if ax != event.inaxes:
                        continue
                    gid = ax.get_gid()
                    n = int(gid.split(' ')[-1])
                    title = fig._suptitle.get_text()
                    fignum = int(title.split(' ')[-1])
                    this_df = final_df.loc[fignum].copy()
                    value = this_df['Values'][n]
                    x_value = this_df[x_column]

                    scatter_points.set_offsets(np.append(scatter_points.get_offsets(), [[x_value, value]], axis=0))
                    scatter_ax.figure.canvas.draw_idle()

                    highlighted_point[x_value] = value
                    # Update the line connecting highlighted points
                    highlighted_x, highlighted_y = np.array([[x, y] for x, y in highlighted_point.items()]).T
                    if highlighted_line:
                        highlighted_line.remove()
                    highlighted_line, = scatter_ax.plot(highlighted_x, highlighted_y, 'ro-')
                    scatter_ax.figure.canvas.draw_idle()
                    break

    df['Values'] = pd.Series([[] for _ in range(df.shape[0])], dtype=object)
    df['Values'] = df['Values'].astype(object)  # Ensures it can hold lists
    df.reset_index(drop=True, inplace=True)
    axes = {}
    for n, data in enumerate(df[data_column].values):
        dic = {}
        dmin = data.ravel().min()
        dmax = data.ravel().max()
        sub_images = np.array_split(data, num_images, axis=0)
        sub_images = [np.array_split(sub_image, num_images, axis=1) for sub_image in sub_images]
        sub_images = [item for sublist in sub_images for item in sublist]

        calculation_func = functions(value_column)
        if calculation_func is None:
            raise ValueError(f'Function for {value_column} does not exist.')

        values = [calculation_func(sub_image, df['Lateral Resolution'].iloc[n]) for sub_image in sub_images]

        df.at[n, 'Values'] = values

        dic = {
            f'Image {n}': {
                f'Sub-image {i}': sub_image for i, sub_image in enumerate(sub_images)
            }
        }

        axes |= plot_all(dic, close=False, fullname=False, titles=False, vlims=[dmin, dmax])
    if scatter_fig is None:
        print('Making Scatterplot fig')
        scatter_fig, scatter_ax = plt.subplots()
        scatter_ax.set_title(f'{value_column} vs {x_column}')
        scatter_ax.set_xlabel(x_column)
        scatter_ax.set_ylabel(value_column)
        scatter_points = scatter_ax.scatter([], [])
    final_df = df.copy()
    for fig in axes.keys():
        fig.canvas.mpl_connect('button_press_event', on_click)
    plt.pause(0.1)


def remove_from_combinedfiles(folders):
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    path_to_metadata = root + 'METADATA_file.xlsx'
    path_to_allfeatures = root + 'Interferometer/allfeatures.pkl'

    if isinstance(folders, (str, int)): folders = [str(folders)]
    for folder in folders:
        folder = folder + '/' if not folder.endswith('/') else ''
        # metadata
        if os.path.exists(path_to_metadata):
            metadata = pd.read_excel(path_to_metadata)
            metadata = metadata[metadata['Folder'] != folder]
            metadata.to_excel(path_to_metadata, index=False)
        # features
        if os.path.exists(path_to_allfeatures):
            allfeatures = read_dataframe(path_to_allfeatures)
            allfeatures = allfeatures[allfeatures['Folder'] != folder]
            save_dataframe(path_to_allfeatures, allfeatures)


def clear_example():
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    example_folder = root + 'Interferometer/example/'
    if os.path.exists(example_folder + 'features.pkl'):
        os.remove(example_folder + 'features.pkl')
    if os.path.exists(example_folder + 'data.pkl'):
        os.remove(example_folder + 'data.pkl')

        # metadata
    path_to_metadata = root + 'METADATA_file.xlsx'
    if os.path.exists(path_to_metadata):
        metadata = pd.read_excel(path_to_metadata)
        metadata = metadata[metadata['Date'] != 'example']
        metadata.to_excel(path_to_metadata, index=False)

    # features
    path_to_allfeatures = root + 'Interferometer/allfeatures.pkl'
    if os.path.exists(path_to_allfeatures):
        allfeatures = read_dataframe(path_to_allfeatures)
        allfeatures = allfeatures[allfeatures['Date'] != 'example']
        save_dataframe(path_to_allfeatures, allfeatures)


def renumber_replicates(folder, n_replicates=2, n_samples=10, min_num=None, max_num=None, mode=None, starting_num=None):
    if mode is None:
        while mode not in ['samples', 'replicates']:
            mode = input('Please specify a mode of either "samples" or "replicates": ')
            print('Using mode:', mode)
    n_samples = int(input(f'How many samples do you have? Default is {n_samples}.') or n_samples)
    n_replicates = int(input(f'How many replicates do you have? Default is {n_replicates}.') or n_replicates)
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    folder = root + f'Interferometer/{folder}' + ('/' if not folder.endswith('/') else '')
    file_map = {}
    for file in os.listdir(folder):
        if '.datx' in file[:-5]:
            os.rename(folder + file, folder + file[:-5])
    if starting_num is None:
        starting_num = max([-1] + [int(file.split('_')[0]) for file in os.listdir(folder) if
                                   file.endswith('.datx') and '_' in file]) + 1
    nums = np.sort([int(file[:-5]) for file in os.listdir(folder) if file.endswith('.datx') and '_' not in file])
    if min_num is None:
        min_num = nums.min()
    if max_num is None:
        max_num = nums.max()
    if mode == 'samples':
        names = {num: str(((num - min_num) // n_samples) // n_replicates * n_samples + (
                    num - min_num) % n_samples + starting_num) + '_' + str(
            (((num - min_num) // n_samples) % n_replicates) + 1) for num in nums if num >= min_num and num <= max_num}
    elif mode == 'replicates':
        names = {
            num: str((num - min_num) // n_replicates + starting_num) + '_' + str((num - min_num) % n_replicates + 1) for
            num in nums if num >= min_num and num <= max_num}
    for file in os.listdir(folder):
        file = file[:-5]
        if file.isdigit() and int(file) >= min_num and int(file) <= max_num:
            file_map[file] = names[int(file)]
        else:
            continue
    for old in sorted(file_map.keys(), key=int):
        new = file_map[old]
        print(old + '.datx', '-->', new + '.datx')
    resp = input('Check the changes above. Would you like to proceed? (y,[n]): ').strip().lower().startswith('y')
    if not resp: return
    for old, new in file_map.items():
        os.rename(folder + old + '.datx', folder + new + '.datx')
        if os.path.exists(folder + 'raw/'):
            os.rename(folder + 'raw/' + old + '.datx', folder + 'raw/' + new + '.datx')
    if os.path.exists(folder + 'file_map.json'):
        file_map |= read_json(folder + 'file_map.json')
    save_json(folder + 'file_map', file_map)


def restore_replicates(folder):
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    folder = root + f'Interferometer/{folder}' + ('/' if not folder.endswith('/') else '')
    file_map = read_json(folder + 'file_map')
    for new, old in file_map.items():
        os.rename(folder + new + '.datx', folder + old + '.datx')
    os.remove(folder + 'file_map.pkl')


def plot_df(df, split=None, sort=None, close=True, data_column='Data-array', background=None, unique_box=True,
            **kwargs):
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
    if split is None:
        split = []
    elif isinstance(split, str):
        split = [split]
    if sort is None:
        sort = []
    elif isinstance(sort, str):
        sort = [sort]
    if 'Replicate' not in df.columns:
        df['Replicate'] = [1 for i in range(df.shape[0])]
    if 'Folder' not in df.columns:
        df['Folder'] = ['Unknown' for i in range(df.shape[0])]
    ascending = [not col.startswith('-') for col in sort]
    sort = [col.lstrip('-') for col in sort]
    df['Unique Identifier'] = [i for i in range(df.shape[0])]
    giant = split + sort
    for n, spl in enumerate(giant):
        if '+' in spl:
            only = spl.split('+')
            keep = only.pop(0)
            if keep == 'Strain ID':
                only = ['0' * (strain_max_len - len(i)) + i for i in only]
            col_type = df[keep].map(type).iloc[0]
            if not df[keep].map(type).nunique() == 1:
                df.fillna({keep: 'None'}, inplace=True)
                col_type = df[keep].map(type).iloc[0]
                if not df[keep].map(type).nunique() == 1:
                    raise TypeError(f'Not all values in {keep} are the same type.')
            mask = [any([col_type(i) == col_type(j) for j in only]) for i in df[keep]]

        elif '-' in spl:
            rems = spl.split('-')
            keep = rems.pop(0)
            if keep == 'Strain ID':
                rems = ['0' * (strain_max_len - len(i)) + i for i in rems]
            mask = [all([str(i) != str(j) for j in rems]) for i in df[keep]]
        else:
            continue
        df = df.loc[mask]
        if n >= len(split):
            sort[n - len(split)] = keep
        else:
            split[n] = keep
    print(sort)
    df_sorted = df.sort_values(by=sort, ascending=ascending)
    if 'Notes' in df_sorted.columns: df_sorted.fillna({'Notes': 'None'}, inplace=True)
    true_split = [np.unique(df_sorted[s]) for s in split]
    split_names = [[s for i in true_split[n]] for n, s in enumerate(split)]
    dic = {
        data_column + '--' + str(vals): {'-'.join([fold, str(base), str(int(rep))]): d for base, rep, fold, d, *test in
                                         df_sorted[['FileBase', 'Replicate', 'Folder', data_column, *s]].values if
                                         all(test == np.array(vals))} for s, vals in
        zip(itertools.product(*split_names), itertools.product(*true_split))}
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
                    this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in
                                      df[['Folder', 'FileBase', 'Replicate']].values]]
                    i, j = np.unravel_index(n, (ii, jj))
                    print(f'\n\nAxis clicked! (Figure {fig.name}, Col {j + 1}, Row {ii - i})')
                    sorted_columns = sorted([col for col in this_df.columns if col != "Notes"]) + (
                        ["Notes"] if "Notes" in this_df.columns else [])
                    sorted_df = this_df[sorted_columns]
                    for name, val in zip(sorted_df.columns, sorted_df.iloc[0]):
                        if 'array' in name or name.startswith('f'):
                            continue
                        print(f'{name}: {val}')
                    # Highlight the clicked image with a rectangle
                    bbox = ax.get_position()
                    dr = 0.1
                    rect = Rectangle((bbox.x0 - dr * bbox.width, bbox.y0 - dr * bbox.height),
                                     bbox.width * (1 + 2 * dr), bbox.height * (1 + 2 * dr),
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
                this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in
                                  df[['Folder', 'FileBase', 'Replicate']].values]]
                unique_id = this_df['Unique Identifier'].values[0]
                text = ax.texts
                if text:
                    for t in text:
                        t.remove()
                else:
                    ax.text(0, 1, str(unique_id), transform=ax.transAxes, color='black', fontsize=12, ha='left',
                            va='top')
            fig.canvas.draw_idle()  # Update the figure to reflect the change

    # Connect the click event to the handler
    global unid_axes
    unid_axes = {}
    if background and background in df.columns:
        unique_vals = np.unique(df[background].values)
        background_cols = {val: plt.get_cmap('tab10')(n) for n, val in enumerate(unique_vals)}

    for fig, axs in axes.items():
        fig.subplots_adjust(bottom=0.1)
        if background is not None:
            for ax in axs:
                if ax.get_label() == '<colorbar>':
                    continue
                this_df = df.loc[['-'.join([d, str(f), str(int(r))]) == ax.get_gid() for d, f, r in
                                  df[['Folder', 'FileBase', 'Replicate']].values]]
                val = this_df[background].values[0]
                color = background_cols[val]
                bbox = ax.get_position()
                dx = 0.15 * bbox.width
                dy = 0.15 * bbox.height
                rect = Rectangle((bbox.x0 - dx, bbox.y0 - dy), bbox.width + 2 * dx, bbox.height + 2 * dy,
                                 transform=fig.transFigure, facecolor=color, alpha=0.3, clip_on=False, zorder=-2)
                fig.patches.append(rect)
            for n, (val, color) in enumerate(background_cols.items()):
                x_position = 0 + .125 * (n // 2)  # Adjust x_position for each value
                y_position = .04 - 0.04 * (n % 2)
                rect = Rectangle((x_position, y_position), 0.1, 0.03, transform=fig.transFigure, facecolor=color,
                                 alpha=0.2, clip_on=False, zorder=-1)
                fig.patches.append(rect)
                fig.text(x_position + 0.05, y_position + 0.015, str(val), ha='center', va='center', fontsize=12,
                         color='black')

        if unique_box:
            ax = plt.axes([0.4, 0.025, 0.2, 0.05])
            unid_axes[fig] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('lightgray')
            ax.text(0.5, 0.5, 'Unique ID', fontsize=12, ha='center', va='center')
            fig.canvas.mpl_connect('button_press_event', on_button_click)

        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.pause(0.001)
    return axes, df


import gc


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
        raise KeyError(
            'Unique Identifiers not found in dataframe. Use plot_df function to create them and identify which data you wish to plot.')

    response = False
    meta = pd.DataFrame()
    figs = []

    for i in identifiers:
        i = int(i)
        mask = df['Unique Identifier'] == i
        this_meta = df.loc[mask].copy()
        data = this_meta[data_column].values[0]
        resolution = this_meta['Lateral Resolution'].values[0] * 1e6  # in microns
        if dimensions:
            if len(identifiers) > 10 and not response:
                response = input(
                    'You are requesting to plot more than 10 images in 3D. Are you sure you wish to continue? (y,[n])').lower().strip() == 'y'
            else:
                response = True
            if response:
                fig = plot_3d(data, lims=color_lims, lat_res=resolution).figure
            else:
                break
        else:
            fig = plot_contour(data, vlims=color_lims, xy_scale=resolution, xy_units='um', **kwargs).ax.figure

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
            if not len(main_ax.texts) > 0:
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
                gc.collect()
                break
            elif raw:
                mask = df['Unique Identifier'] == fig.identifier
                this_meta = df.loc[mask].copy()
                raw_data = this_meta['Raw-array'].values[0]
                resolution = this_meta['Lateral Resolution'].values[0] * 1e6
                if raw_data is not None:
                    max_h = np.nanmax(raw_data.ravel())
                    plt.figure()
                    plot_3d(raw_data, zlims=[0, 1.1 * max_h], consistent_lims=False, lat_res=resolution * 1e6)
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
            resolution = this_meta['Lateral Resolution'].values[0] * 1e6
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
    this_meta.drop(columns=['Data-array'] + [i for i in this_meta.columns if i.startswith('f')], errors='ignore',
                   inplace=True)
    meta = pd.concat([meta, this_meta])

    return meta, figs


def get_root():
    split_obj = '\\' if '\\' in os.getcwd() else '/'
    root = '/'.join(os.getcwd().split(split_obj)[:-1]) + '/Data/'
    return root


from scipy.stats import binom, nbinom


def plot_binomial_distribution(n_list, p, negative=False, together=False):
    """
    Plots binomial distributions for a list of different n values on the same graph,
    dynamically limiting the x-axis to 10% above the highest number of trials.

    Parameters:
    n_list (list of int): List of numbers of trials. (Total bacteria in total volume)
    p (float): Probability of success in each trial. (fraction of total volume you sample)
    """

    plt.figure(figsize=(10, 6))
    max_n = 0  # To track the highest number of trials
    min_n = 0
    all_probs = {}
    if not isinstance(n_list, list):
        n_list = list(n_list) if hasattr(n_list, '__iter__') else [n_list]
    for n in n_list:
        x = np.arange(int(n + 1)) if not negative else np.arange(1, int(n * (1 - p) / p) * 10)
        probabilities = binom.pmf(x, n, p) if not negative else nbinom.pmf(x, n, p)
        mask = probabilities > .001 * np.max(probabilities)
        max_n = max(max_n, x[mask][-1])  # Update max_n
        min_n = min(min_n, x[mask][0])
        plt.plot(x, probabilities, label=f'n={n}', alpha=0.7)
        all_probs[n] = probabilities

    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distributions (prob={p:.5f})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Dynamically limit the x-axis to 10% above the highest number of trials
    plt.xlim(min_n * min_n, max_n * 1.1)

    plt.tight_layout()
    if together:
        lens = [len(l) for l in all_probs.values()]
        arr = [l.tolist() + [0 for n in range(max(lens) - len(l))] for l in all_probs.values()]
        tot = np.sum(arr, axis=0) / len(arr)
        plt.plot(np.arange(len(tot)), tot, label='Together')
    plt.legend()
    plt.pause(0.01)
    return all_probs


import importlib


def reload():
    import myfunctions
    importlib.reload(myfunctions)


def full_characterization(df, x_data='Notes'):
    """
    Characterizes agar plates by calculating surface roughness, slope, and Rz.
        NOTE THAT THE RAW DATA MUST BE COLLECTED WITH ZERO MACHINE TILT to apply slope limits
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'Data-array' and 'Raw-array' columns.
    x_data (str): The column to plot against (e.g., 'Notes' or 'Treatment').
    plot (bool): Whether to plot the results. Default is True.

    Returns:
    pd.DataFrame: The DataFrame with added columns for surface roughness, slope, and Rz.
    """
    if 'Data-array' not in df.columns or 'Raw-array' not in df.columns:
        raise KeyError("The DataFrame must contain 'Data-array' and 'Raw-array' columns.")

    # Calculate surface roughness, slope, and Rz
    df['Sa (um)'] = df['Data-array'].apply(functions('ra'))
    df['Sq (um)'] = df['Data-array'].apply(functions('rq'))
    df['slope (um/mm)'] = df['Raw-array'].apply(functions('fit_slope')) / 1e3 / (df['Lateral Resolution'].values * 1e3)
    df['m-max (deg)'] = df.apply(lambda row: functions('max_slope')(row['Data-array'], row['Lateral Resolution'] * 1e3),
                                 axis=1)
    df['m-avg (deg)'] = df.apply(lambda row: functions('avg_slope')(row['Data-array'], row['Lateral Resolution'] * 1e3),
                                 axis=1)
    df['Sz (um)'] = df['Data-array'].apply(functions('rz'))
    num_images = lambda obj, zoom: int(np.ceil(10 / (float(obj[:-1]) * float(zoom[:-1]))))
    df['Sz-mean (um)'] = df.apply(lambda row: np.mean([functions('rz')(sub_image) for sub_image in
                                                       make_sub_images(row['Data-array'],
                                                                       num_images=num_images(row['Objective'],
                                                                                             row['Zoom']))]), axis=1)
    df['Sz-std (um)'] = df.apply(lambda row: np.std([functions('rz')(sub_image) for sub_image in
                                                     make_sub_images(row['Data-array'],
                                                                     num_images=num_images(row['Objective'],
                                                                                           row['Zoom']))]), axis=1)
    df['Sa-mean (um)'] = df.apply(lambda row: np.mean([functions('ra')(sub_image) for sub_image in
                                                       make_sub_images(row['Data-array'],
                                                                       num_images=num_images(row['Objective'],
                                                                                             row['Zoom']))]), axis=1)
    df['Sa-std (um)'] = df.apply(lambda row: np.std([functions('ra')(sub_image) for sub_image in
                                                     make_sub_images(row['Data-array'],
                                                                     num_images=num_images(row['Objective'],
                                                                                           row['Zoom']))]), axis=1)
    # Plot the results if requested

    limits = {'Sa (um)': 0.08,
              # 'Sa-mean (um)': 0.02,
              # 'Sa-std (um)':  0.01,
              'slope (um/mm)': 0.1,
              'm-avg (deg)': 0.001,
              'm-max (deg)': 0.01,
              'Sq (um)': 0.1,
              'Sz (um)': 0.5,
              'Sz-mean (um)': 0.1,
              'Sz-std (um)': 0.05
              }
    y_data = list(limits.keys())
    thresholds = list(limits.values())

    grouped = df.groupby(['Objective', 'Zoom'])
    for (objective, zoom), group in grouped:
        fig_title = f'Objective: {objective}, Zoom: {zoom}'
        axes = plot_features(group, x_data=x_data, y_data=y_data, color=x_data)

        for ax, y_col, thresh in zip(axes.ravel(), y_data, thresholds):
            ax.axhline(thresh, color='red', linestyle='--', label=f'{y_col} < {thresh}')
            ax.legend().set_visible(False)
            ax.set_yscale('log')

        plt.suptitle(fig_title)
        plt.pause(0.01)

    return axes


def characterize_agar_plates(data, x_data='Notes'):
    """
    Characterizes agar plates by calculating surface roughness, slope, and Rz.
        NOTE THAT THE RAW DATA MUST BE COLLECTED WITH ZERO MACHINE TILT to apply slope limits
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'Data-array' and 'Raw-array' columns.
    x_data (str): The column to plot against (e.g., 'Notes' or 'Treatment').
    plot (bool): Whether to plot the results. Default is True.

    Returns:
    pd.DataFrame: The DataFrame with added columns for surface roughness, slope, and Rz.
    """
    if isinstance(data, (str, int, float)):
        date = data
        df = None
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        date = None
    else:
        raise TypeError('Data is not a readable type. Please provide a date or dataframe')

    high_obj = '10x'
    low_obj = '2.75x'

    df = collect_data(dataframe=df, date=date, region='Agar', objective=[low_obj, high_obj])

    if 'Data-array' not in df.columns or 'Raw-array' not in df.columns:
        raise KeyError("The DataFrame must contain 'Data-array' and 'Raw-array' columns.")

    # Calculate surface roughness, slope, and Rz
    df['Sq (um)'] = df['Data-array'].apply(functions('rq'))
    df['Sz (um)'] = df['Data-array'].apply(functions('rz'))
    df['m-max (deg)'] = df.apply(lambda row: functions('max_slope')(row['Data-array'], row['Lateral Resolution'] * 1e3),
                                 axis=1)
    df['m-avg (deg)'] = df.apply(lambda row: functions('avg_slope')(row['Data-array'], row['Lateral Resolution'] * 1e3),
                                 axis=1)
    df['power'] = [np.array(functions('power')(row['Data-array'], row['Lateral Resolution'])) for n, row in
                   tqdm(df.iterrows(), total=len(df))]
    df['Power slope-medium'] = df.apply(
        lambda row: functions('power_slope')(*row['power'], lims=[20, 100], latres=row['Lateral Resolution'] * 1e6),
        axis=1)

    df['Power slope-large'] = df.apply(
        lambda row: functions('power_slope')(*row['power'], lims=[200, 1000], latres=row['Lateral Resolution'] * 1e6),
        axis=1)
    df['Tilt (um/mm)'] = df['Raw-array'].apply(functions('fit_slope')) / 1e3 / (df['Lateral Resolution'].values * 1e3)

    num_images = lambda obj, zoom: int(np.ceil(10 / (float(obj[:-1]) * float(zoom[:-1]))))
    df['Tilt-mean (um/mm)'] = df.apply(lambda row: np.mean(
        [functions('fit_slope')(sub_image, latres=row['Lateral Resolution']) for sub_image in
         make_sub_images(row['Data-array'], num_images=num_images(row['Objective'], row['Zoom']))]), axis=1)

    limits = {high_obj: {
        'Sq (um)': 0.005,
        'Sz (um)': 0.01,
        'm-max (deg)': 1,
        'm-avg (deg)': 0.1,
        'Power slope-medium': 1
    },
        low_obj: {
            'Power slope-large': -2,
            'Tilt (um/mm)': .1,
            'Tilt-mean (um/mm)': .1
        }}
    fignum = f'Plate Evaluation {date}'
    if plt.fignum_exists(fignum):
        plt.close(fignum)
    fig = plt.figure(fignum, figsize=(14, 8))
    axes = fig.subplots(*define_subplot_size(len(limits[low_obj]) + len(limits[high_obj])))
    grouped = df.groupby('Objective')
    n_ax = 0
    color_func = lambda arr, thresh: ['blue' if val < thresh else 'red' for val in arr]
    colors = plt.get_cmap('tab10')
    for n_obj, (objective, group) in enumerate(grouped):
        these_lims = limits[objective]
        plates = np.unique(group['FileBase'].values).tolist()
        x_data = [plates.index(i) + 1 for i in group['FileBase'].values]
        for metric, threshold in these_lims.items():
            ax = axes.ravel()[n_ax]
            n_ax += 1
            y_data = group[metric].values
            ax.scatter(x_data, y_data, c=color_func(y_data * np.sign(threshold), threshold), alpha=0.5)
            ax.axhline(abs(threshold), color='red', linestyle='--', label=f'{metric} < {threshold}')
            # Add a small black arrow pointing in the opposite direction of the sign of threshold
            unique_x = np.unique(x_data)
            # Add a triangle marker on the horizontal line at abs(threshold)
            triangle_marker = 'v' if threshold > 0 else '^'  # Down triangle for positive threshold, up triangle for negative
            ax.scatter([np.mean(unique_x)], [abs(threshold)], marker=triangle_marker, color='black', s=100, zorder=5)
            # Average over y_data for each x_data
            avg_y = np.array([np.mean(y_data[np.array(x_data) == ux]) for ux in unique_x])
            ax.scatter(unique_x, avg_y, c=color_func(avg_y * np.sign(threshold), threshold), marker='x', alpha=1,
                       label='Average')

            ax.legend().set_visible(False)
            ax.set_yscale('log')
            ax.set_title(metric, fontsize=10)
            ax.set_xticks(unique_x)
            # ax.set_xticklabels(unique_x, fontsize=14)
            # Add background color to axes
            rect = Rectangle((-0.1, -0.1), 1.2, 1.2, transform=ax.transAxes, facecolor=colors(n_obj), alpha=0.2,
                             clip_on=False, zorder=-1)
            ax.add_patch(rect)
        x_position = 0.2 + .125 * (n_obj // 2)  # Adjust x_position for each objective
        y_position = 0.94 - 0.04 * (n_obj % 2)
        rect = Rectangle((x_position, y_position), 0.1, 0.03, transform=fig.transFigure, facecolor=colors(n_obj),
                         alpha=0.2, clip_on=False, zorder=-1)
        fig.patches.append(rect)
        fig.text(x_position + 0.05, y_position + 0.01, objective, ha='center', va='center', fontsize=12, color='black')

    fig.supxlabel('Plate Number')
    fig.suptitle(f'Plate Evaluation')
    fig.tight_layout()
    plt.pause(0.01)

    return axes


def plot_grid(df, x_data='Treatment', y_data='Strain ID', close=True):
    if close:
        plt.close('all')
    plt.figure()
    x_values = np.unique(df[x_data].values).tolist()
    y_values = np.unique(df[y_data].values).tolist()
    for i, (y_val, y_df) in enumerate(df.groupby(y_data)):
        for x_val, x_df in y_df.groupby(x_data):
            data = x_df['Data-array'].values[0]
            plot_contour(data,
                         location=f'{len(y_values) * len(x_values)}_{i * len(x_values) + x_values.index(x_val) + 1}',
                         vlims=[-.1, 1], new_fig=False, axis=False, cbar=False)

    plt.gcf().subplots_adjust(wspace=0.1, hspace=0.1)
    left = plt.gcf().subplotpars.left
    right = plt.gcf().subplotpars.right
    bottom = plt.gcf().subplotpars.bottom
    top = plt.gcf().subplotpars.top
    height = top - bottom
    width = right - left
    wspace = plt.gcf().subplotpars.wspace
    hspace = plt.gcf().subplotpars.hspace

    for n in np.arange(len(x_values) * len(y_values)):
        if n % len(x_values) == 0:
            plt.gcf().text(left, top - height * ((n // len(x_values) + 0.5) / len(y_values)),
                           y_values[n // len(x_values)], rotation=0, va='center',
                           ha='right', fontsize=10)
        if n // len(x_values) == len(y_values) - 1:
            plt.gcf().text(left + width * ((n % len(x_values) + 0.5) / len(x_values)), bottom,
                           x_values[n % len(x_values)], va='top', ha='right',
                           fontsize=10, rotation=30)
    plt.pause(0.01)


def cross_correlation_2d(array_a, array_b):
    """
    Computes the 2D cross-correlation function of two arrays.

    Parameters:
    array_a (numpy.ndarray): First input 2D array.
    array_b (numpy.ndarray): Second input 2D array.

    Returns:
    numpy.ndarray: 2D cross-correlation function.
    """
    # Ensure both arrays are the same size
    if array_a.shape != array_b.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Subtract the mean to normalize the arrays
    array_a = array_a - np.mean(array_a)
    array_b = array_b - np.mean(array_b)

    # Compute the 2D FFT of both arrays
    fft_a = np.fft.fft2(array_a)
    fft_b = np.fft.fft2(array_b)

    # Compute the cross-power spectrum
    cross_power_spectrum = fft_a * np.conj(fft_b)

    # Compute the inverse FFT of the cross-power spectrum to get the cross-correlation
    cross_corr = np.fft.ifft2(cross_power_spectrum)

    # Shift the zero-frequency component to the center
    cross_corr = np.fft.fftshift(cross_corr)

    # Take the real part (imaginary part should be negligible)
    cross_corr = np.real(cross_corr)

    return cross_corr


def find_translation(array_a, array_b, plot_metrics=False):
    """
    Aligns two 2D arrays based on the maximum of their cross-correlation and plots the difference.

    Parameters:
    array_a (numpy.ndarray): First input 2D array.
    array_b (numpy.ndarray): Second input 2D array.

    Returns:
    numpy.ndarray: The difference between the aligned arrays.
    """
    # Compute the cross-correlation
    min_i = min(array_a.shape[0], array_b.shape[0])
    min_j = min(array_a.shape[1], array_b.shape[1])
    array_a = array_a[:min_i, :min_j]
    array_b = array_b[:min_i, :min_j]
    cross_corr = cross_correlation_2d(array_a, array_b)

    # Find the indices of the maximum cross-correlation
    max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
    shift = np.array(max_idx) - np.array(cross_corr.shape) // 2
    if plot_metrics:
        if plt.fignum_exists('Align Metrics'):
            fig = plt.figure('Align Metrics')
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure('Align Metrics')
            ax = fig.add_subplot(121)
        ax.scatter(*shift)
        ax.set_xlim(-cross_corr.shape[0] // 2, cross_corr.shape[0] // 2)
        ax.set_ylim(-cross_corr.shape[1] // 2, cross_corr.shape[1] // 2)
        ax.set_ylabel('Shift in axis 1')
        ax.set_xlabel('Shift in axis 0')
    # Shift the arrays
    shifted_a = array_a[max(0, shift[0]):array_a.shape[0] + min(0, shift[0]),
                max(0, shift[1]):array_a.shape[1] + min(0, shift[1])]
    shifted_b = array_b[max(0, -shift[0]):array_b.shape[0] + min(0, -shift[0]),
                max(0, -shift[1]):array_b.shape[1] + min(0, -shift[1])]

    return shifted_a, shifted_b, shift


def find_rotation(array_a, array_b, metric='correlation', small_angle=True, plot_metrics=False):
    """
    Aligns two 2D arrays by rotating one array incrementally and finding the best match based on a specified metric.

    Parameters:
    array_a (numpy.ndarray): First input 2D array.
    array_b (numpy.ndarray): Second input 2D array.
    metric (str): Metric to determine the best match. Options are 'correlation' or 'difference'. Default is 'correlation'.
    angle_step (float): Step size for rotation in degrees. Default is 1.
    max_angle (float): Maximum rotation angle to consider in degrees. Default is 360.

    Returns:
    tuple: The best rotation angle and the aligned difference array.
    """

    def calculate_metric(array1, array2, metric):
        if metric == 'correlation':
            return np.corrcoef(array1.ravel(), array2.ravel())[0, 1]
        elif metric == 'difference':
            return -np.sum((array1 - array2) ** 2)
        else:
            raise ValueError("Unsupported metric. Use 'correlation' or 'difference'.")

    best_metric = -np.inf if metric == 'correlation' else np.inf
    courseness = [5, 1, .1][int(small_angle):]
    best_angle = 0
    for n, angle_step in enumerate(courseness):
        init_angle = best_angle - 10 * angle_step
        angles = np.arange(init_angle, init_angle + angle_step * 21, angle_step)
        metrics = []
        for angle in angles:
            rotated_b = scipy.ndimage.rotate(array_b, angle, reshape=False, mode='nearest')

            min_i, min_j = min(array_a.shape[0], rotated_b.shape[0]), min(array_a.shape[1], rotated_b.shape[1])
            # Align the centers of the two arrays
            center_a = (array_a.shape[0] // 2, array_a.shape[1] // 2)
            center_b = (rotated_b.shape[0] // 2, rotated_b.shape[1] // 2)

            # Calculate the starting indices for slicing
            start_a_i, start_a_j = max(0, center_a[0] - min_i // 2), max(0, center_a[1] - min_j // 2)
            start_b_i, start_b_j = max(0, center_b[0] - min_i // 2), max(0, center_b[1] - min_j // 2)

            # Slice the arrays to align their centers
            rotated_a = array_a[start_a_i:start_a_i + min_i, start_a_j:start_a_j + min_j]
            rotated_b = rotated_b[start_b_i:start_b_i + min_i, start_b_j:start_b_j + min_j]

            current_metric = calculate_metric(rotated_a, rotated_b, metric)
            metrics.append(current_metric)
            if (metric == 'correlation' and current_metric > best_metric) or \
                    (metric == 'difference' and current_metric < best_metric):
                best_metric = current_metric
                best_angle = angle
                aligned_b = rotated_b
                aligned_a = rotated_a
        if plot_metrics:
            if plt.fignum_exists('Align Metrics'):
                fig = plt.figure('Align Metrics')
                ax = fig.get_axes()[1]
            else:
                fig = plt.figure('Align Metrics')
                ax = fig.add_subplot(121)
            ax.plot(angles, metrics, alpha=0.4)
            ax.scatter(best_angle, best_metric, color=['red', 'gold', 'green'][len(courseness) - 1 - n],
                       label='Best Angle')
            ax.set_xlabel('Rotation Angle (degrees)')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'Rotation Metric: {metric}')
    return aligned_a, aligned_b, best_angle


def align_and_plot_differences(df, plot_metrics=False, split=['Drug Treatment', 'Strain ID'], data_column='Data-array',
                               **kwargs):
    df = collect_data(dataframe=df, **kwargs).copy()
    df.fillna({i: 0 for i in split}, inplace=True)
    grouped = df.groupby(split)
    vlims = kwargs.get('vlims', [-0.2, 0.2])
    if plot_metrics:
        fig = plt.figure('Align Metrics', figsize=(12, 7))
        plt.clf()
        fig.subplots(1, 2)

    for group_info, group in grouped:
        group.sort_values(by=['Incubation'], inplace=True)
        group = group.reset_index(drop=True)
        if len(group) < 2:
            print(f"Skipping group {group_info} with only one entry.")
            continue
        num_combos = choose(len(group), 2)
        for n, combo in enumerate(tqdm(itertools.combinations(group['Incubation'], 2), total=num_combos)):
            latres = group['Lateral Resolution'].values[0] * 1e6
            # Extract the two data arrays
            array_a = group.loc[group['Incubation'] == combo[0], data_column].values[0]
            array_b = group.loc[group['Incubation'] == combo[1], data_column].values[0]

            # Align and plot the difference
            shifted_a, shifted_b, translation = find_translation(array_a, array_b, plot_metrics=plot_metrics)
            rotated_a, rotated_b, angle = find_rotation(shifted_a, shifted_b, metric='correlation', small_angle=True,
                                                        plot_metrics=plot_metrics)
            figname = f'Differences of {group[split].values[0]}'
            if plt.fignum_exists(figname) and n == 0:
                plt.close(figname)
            plt.figure(figname)
            plot_contour(rotated_b - rotated_a, vlims=vlims, location=f'{num_combos}_{n + 1}', new_fig=False,
                         cbar=False, axis=False)
            plt.title(combo)
            # plot_contour(array_b  -array_a  , axis=False, vlims=[-0.2, 0.2], location=f'{choose(len(group),2)*2}_{2*n+2}', new_fig=False, cbar=False)
            # plt.title(combo)
        plt.gcf().suptitle(group[split].values[0])
        plt.pause(.01)


def plot_urine_cfus():
    data = collect_data(media='Urine', incubation=0, treatment='None', _plated_cfu='#DIV/0!')
    dat = data.loc[data['Plated CFU'] != '#DIV/0!'].copy()
    d = dat.groupby('Strain ID').agg({'Plated CFU': 'mean'})
    doubles = d.apply(lambda x: np.log2(1e5 / min([x['Plated CFU'], 1e5])), axis=1)

    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    axs = fig.subplots(1, 2, sharey=True)
    ax = axs[0]
    y, bins = np.histogram(d.map(np.log10).values, bins=np.arange(-1, 7, .1))
    cumulative = np.cumsum(y)
    y = cumulative / cumulative[-1]
    ax.plot(bins[1:], y)
    for c, cut in zip(['r', 'k'], [4, 5]):
        p = y[np.argmin(np.abs(bins[1:] - cut))]
        ax.plot([cut, cut], [0, p], '--', color=c)
        ax.plot([-1, cut], [p, p], '--', color=c)
        ax.text(cut, p, f'{p:.2f}', color=c, ha='right', va='bottom')

    ax.set_xlabel('Log10(CFU/1.5ul) of Urine Samples')
    ax.set_ylabel('Cumulative Probability')

    ax = axs[1]
    y, bins = np.histogram(doubles.values, bins=np.arange(0, 10, .5))
    y = y / np.sum(y)  # = cumulative / cumulative[-1]
    ax.plot(bins[1:], y)
    ax.set_xlabel('Doublings to reach 1e5 CFU/1.5ul')
    ax.axvline(doubles.mean(), color='r', linestyle='--')
    ax.set_ylabel('Probability')
    # Add a second x-axis
    ax2 = ax.twiny()
    ax2.set_xlabel('Time to reach 1e5 CFU/1.5ul (hours)')
    ax2.set_xlim(ax.get_xlim()[0] * 0.5, ax.get_xlim()[1] * 0.5)
    plt.tight_layout()
    plt.pause(1)
    return fig