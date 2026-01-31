import numpy as np
import pandas as pd
from scipy.stats import linregress

def linear_regions(x, y, dxi=1, tol=0.25):
    maxit = len(x) // dxi
    prev_tang = linregress(x[:max(dxi, 5)], y[:max(dxi, 5)]).slope
    lrs = [0]  # start of the first linear region is always the first index
    last_k = 0
    tangents,rs=[],[]
    # Start loop over all partitions of `x` into `dxi` intervals:
    for k in range(5, maxit):
        r = slice(last_k, (k + 1) * dxi)
        if len(np.unique(x[r]))==1: continue
        tang = linregress(x[r], y[r]).slope
        rs.append(np.arange(last_k,k+1))
        if np.isclose(tang, prev_tang, rtol=tol, atol=0):
            # Tangent is similar with initial previous one (based on tolerance)
            continue
        else:
            # Tangent is not similar enough
            # Set the START of a new linear region
            # which is also the END of the previous linear region
            tangents.append(tang)
            lrs.append(k * dxi)
            last_k = k
            # Set new previous tangent (only if it was not the same as the current)
            prev_tang = tang

    # The final linear region always ends here:
    lrs.append(len(x))
    # Reformat into ranges
    lranges = [np.arange(lrs[i], lrs[i+1]) for i in range(len(lrs)-1)]
    # create new tangents that do have linear regression weighted
    tangents = [linregress(x[r], y[r]).slope for r in lranges]
    return lranges, tangents
# Example usage:
# x = np.array([...])  # Your x values
# y = np.array([...])  # Your y values
# lrs, tangents = linear_regions(x, y)
# print("Linear Regions:", lrs)
# print("Tangents:", tangents)

def mirror(z, spaces):
    mirrored_vector = np.roll(z, spaces)
    if spaces >= 0:
        mirrored_vector[:spaces] = np.nan
    else:
        mirrored_vector[spaces:] = np.nan
    return mirrored_vector

def update_moments(z, r, counts, moment_1, moment_2):
    for current_shift in (r, -r):
        mirrored_vector = mirror(z, current_shift)
        idx = np.isnan(mirrored_vector)
        counts += np.logical_not(idx)
        mirrored_vector[idx] = 0
        moment_1 += mirrored_vector
        moment_2 += mirrored_vector ** 2
    return counts, moment_1, moment_2

def getwloc(z, rpx, rx=0.4):
    S = np.floor(2 ** np.arange(0, np.log2(len(z)), rx)).astype(int)
    counts = np.zeros(len(z))
    moment_1 = np.zeros(len(z))
    moment_2 = np.zeros(len(z))
    wloc = np.zeros(len(S) - 1)
    
    for i in range(len(S) - 1):
        counts, moment_1, moment_2 = update_moments(z, S[i + 1], counts, moment_1, moment_2)
        norm_moment_1 = moment_1 / counts
        norm_moment_2 = moment_2 / counts
        variances = norm_moment_2 - (norm_moment_1 ** 2)
        wloc[i] = np.nanmean(np.sqrt(np.abs(variances)))
    
    return S[1:], wloc * rpx

def w_data_extraction(distances, local_widths):
    spec_x, spec_y = np.log10(distances), np.log10(local_widths)
    lr,h_exponents = linear_regions(spec_x, spec_y)  # Find regions where the slope is positive
    idx = lr[0]
    l_sat = distances[idx].sum()
    w_sat = local_widths[-1]
    hurst = h_exponents[0]
    return l_sat, w_sat, hurst

def df_hurst(df):
    df['rpx'] = [0.173 if row['zoom'] == 50 else 0.865 for _, row in df.iterrows()]
    locs, wlocs = [], []

    for _, row in df.iterrows():
        l, w = getwloc(row['homeland'], row['rpx'], rx=0.3)
        locs.append(l)
        wlocs.append(w)

    df['loc'] = locs
    df['wloc'] = wlocs
    l_sat, w_sat, hurst = [], [], []

    for _, row in df.iterrows():
        l, w, h = w_data_extraction(row['loc'], row['wloc'])
        l_sat.append(l)
        w_sat.append(w)
        h = linregress(np.log10(row['loc'])[:15],np.log10(row['wloc'])[:15]).slope
        hurst.append(h)

    df['l_sat'] = l_sat
    df['w_sat'] = w_sat
    df['hurst'] = hurst

    return df

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame
# df = df_hurst(df)
##################################################################
##################################################################
##################################################################
##################################################################
from scipy.signal import find_peaks, argrelextrema

def _changeinbool(b):
    new_vector = np.add(b[:-1].astype(int), b[1:].astype(int))
    change_points = np.where(new_vector == 1)[0]
    in_change = change_points[~b[change_points]]
    out_change = change_points[b[change_points]]
    return in_change, out_change

def _changeinbool(b):
    new_vector = np.diff(b.astype(int))
    change_points = np.where(new_vector != 0)[0]
    in_change = change_points[new_vector[change_points] > 0]
    out_change = change_points[new_vector[change_points] < 0]
    return in_change, out_change
def find_profile_elements(z):
    z_min = 150#0.1 * np.abs(z).max()
    peaks, _ = find_peaks(z, height=z_min)
    valleys = argrelextrema(z, np.less)[0]
    peaks = z>z_min
    valleys = z<-z_min
    in_valley, out_valley = _changeinbool(valleys)
    in_peak, out_peak = _changeinbool(peaks)
    amps = np.int_(peaks)-np.int_(valleys)
    transition_points = []

    n_switches = 0
    last = 0
    for n,a in enumerate(amps):
        if a!=0 and a!=last:
            n_switches +=1
            transition_points.append(n)
            last = a

    t= np.array(transition_points)
    dif = t[1:]-t[:-1];cut=[]
    for n,d in enumerate(dif):
        if d< 0.01*len(z):
            cut.extend([n,n+1])
    for c in np.flip(np.unique(cut)):
        transition_points.pop(c)
    #transition_points.append(len(z))
    element_bounds = [[transition_points[i], transition_points[i+1]] for i in range(len(transition_points)-1)]
    return element_bounds

def element_counter(z):
    my_elements = find_profile_elements(z)
    return int(len(my_elements)/2)

def df_elements(df):
    fluctuations = df['homeland']
    n_elems,elem_w = [],[]
    for fluct in fluctuations:
        n_elements = element_counter(fluct)
        n_elems.append(n_elements)
        
        elem_w.append(RSm(fluct))
    df['Elements'] = n_elems
    df['Element W'] = elem_w
    return df





##################################################################
##################################################################
##################################################################
def _moment(z, i):
    return np.nanmean((z - np.nanmean(z))**i)

def Ra(z):
    return np.nanmean(np.abs(z - np.nanmean(z)))

def Rz(z):
    return np.nanmax(z) - np.nanmin(z)

def Rp(z):
    return np.nanmax(z) - np.nanmean(z)

def Rv(z):
    return np.abs(np.nanmin(z) - np.nanmean(z))

def Rq(z):
    return np.sqrt(_moment(z, 2))

def Rsk(z):
    return _moment(z, 3) / (_moment(z, 2) ** 1.5)

def Rku(z):
    return _moment(z, 4) / (_moment(z, 2) ** 2)

def Rc(z):
    profile_elements = find_profile_elements(z)
    return np.nanmean([Rz(z[bounds[0]:bounds[1]]) for bounds in profile_elements])

def RSm(z):
    profile_elements = find_profile_elements(z)
    return np.nanmean([bounds[1] - bounds[0] for bounds in profile_elements])
