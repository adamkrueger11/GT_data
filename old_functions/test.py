import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname('__file__'))
from functions import convert_data, ring_width_radial, get_features_ID, get_all_fft_features
from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning)

raw,latres = convert_data('/Users/adamkrueger/Downloads/HR/230815_1-per-agar/filled/32_1xTZP.datx',resolution=True, remove=False)
#start, end, width, maxes = ring_width_radial(raw, plot=True, all_data=True)

height_feats = get_features_ID(raw,latres*1e6,obj=1.375,fftval=None)
fft_feats = get_all_fft_features(raw,latres*1e6, region='full', wavelengths='all',plot=False,alpha=0.2)
#print(fft_feats.keys())
#print(height_feats.keys())
plt.show()