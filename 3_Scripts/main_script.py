# %%
import pandas as pd
import os
import warnings

# %% 
"""
####### Initialization of variables ######
"""
# site name and associated comID:
site            = "Marsyangdi"
## set file path:
rt_dir          = r"../1_Data/Fcst_data"
obs_dir         = r"../1_Data/obs_data"
site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
# date list of interest:
date_range      = ['20140101', '20141231']
ens_members     = [*range(1,53)]

# %% Loop through all the files and create a dataframe:
try:
    fcst_data = pd.read_pickle("./pickle_dfs/"+site+".pkl")   
except:
    warnings.warn("Pickle file missing. Creating forecast database using \
            individual ensemble files")
    fcst_data = df_creator(rt_dir, date_range, site_comID, ens_members)

t2 = fcst_data
# 
fcst_data = fcst_data.sort_index().loc(axis=0)[
    (slice(None), slice(None), slice(date_range[0], date_range[1]))
]




# forecast day of interest:
day             = 2
win_len         = 7
