# %%
import os 
from tqdm import tqdm
import pandas as pd 
import shutil
import seaborn as sns

import myfunctions as mf
import functions as f
import matplotlib
import matplotlib.pyplot as plt

%matplotlib tk

# %%
root = '/Users/adamkrueger/Library/CloudStorage/GoogleDrive-akrueger@topodx.com/Shared drives/R&D/Data/Interferometer/previous/HR'
meta = pd.read_excel(os.path.join(root,'METADATA.xlsx'))

combined_dir = os.path.join(root,'combined')
raw_dir = os.path.join(combined_dir,'raw')
full_dir = os.path.join(combined_dir,'filled')
new_dirs = {'raw': raw_dir, 'filled': full_dir}
os.makedirs(full_dir,exist_ok=True)
os.makedirs(raw_dir,exist_ok=True)
meta['Date'].unique()
# %%
dirs = {'raw': '', 'filled': ''}
file_num = 1
for idx, row in tqdm(meta.iterrows(),total=meta.shape[0]):
    date = str(int(row['Date']))
    dir = os.path.join(root,date+'_1-per-agar')
    dirs['raw'] = os.path.join(dir,'raw')
    dirs['filled'] = os.path.join(dir,'filled')

    for dtype, dir in dirs.items():
        new_dir = new_dirs[dtype]
        new_name = os.path.join(new_dir, str(file_num).zfill(3)+'.datx')
        if os.path.exists(new_name):
            print(f"File {new_name} already exists. Skipping.")
            break
        fname = [f for f in os.listdir(dir) if f.startswith(str(row['Strain ID'].astype(int)))]
        if len(fname) > 1:
            print(f"Multiple files found for Strain ID {row['Strain ID']}:")
            for i, f in enumerate(fname):
                print(f"{i}: {f}")
            print("Type the number corresponding to the file you want to select, or type 'none' to skip this row.")
            choice = input("Your choice: ").strip()
            if choice.lower() == 'none':
                break
            try:
                choice = int(choice)
                fname = fname[choice]
            except (ValueError, IndexError):
                print("Invalid choice. Skipping this row.")
                break
        elif len(fname) == 0:
            print(f"No file found for Strain ID {row['Strain ID']}. Skipping this row.")
            break
        else:
            fname = fname[0]
        old_name = os.path.join(dir, fname)
        
        shutil.copy(old_name, new_name)

    meta.loc[idx, 'FileBase'] = str(file_num).zfill(3)
    file_num += 1
meta.to_excel(os.path.join(combined_dir,'METADATA_combined.xlsx'),index=False)
# %%
all_data = pd.read_excel(os.path.join(combined_dir,'METADATA_combined.xlsx'))
dir = new_dirs['filled']
data_arrs = []
for idx, row in tqdm(all_data.iterrows(),total=meta.shape[0]):
    file_base = str(row['FileBase']).zfill(3)
    fname = os.path.join(dir, file_base + '.datx')
    data_arrs.append(mf.convert_data(fname)['Heights']/1000)

all_data['Data-array'] = data_arrs
# %%
mf.plot_df(all_data, data_column='Data-array', split='Phenotype',sort='Log HR frac')

# %%
use_no = [29,26,23,38,27,78,75,30,69]
all_data['use'] = True 
all_data.loc[all_data['FileBase'].astype(int).isin(use_no),'use'] = False
all_data['use'].value_counts()
# %%
clean_data = mf.clean_df(all_data[all_data['use']], data_column='Data-array', response_init='corners',degree_of_fit=2)
# %%
mf.plot_df(clean_data, data_column='Cleaned-Data-array', split='Phenotype', sort='Log HR frac',vlims = [-1,15])
# %%
feats = clean_data['Cleaned-Data-array'].apply(lambda row: f.get_features_ID(row, obj=5.5, zoom = 0.5))
feats_df = pd.DataFrame(list(feats))
feats_df = pd.concat([clean_data.reset_index(drop=True), feats_df], axis=1)

# %%
