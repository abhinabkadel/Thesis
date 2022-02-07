# %%
from datetime import date
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *
import warnings
warnings.filterwarnings("ignore")

# %% 
"""
####### Initialization parameters and get frcst data ######
"""
def get_fcst_data (date_range):
    # site name and associated comID:
    site            = "Balephi"
    ## set file path:
    rt_dir          = r"../1_Data/Fcst_data"
    obs_dir         = r"../1_Data/obs_data"
    site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
    # date list of interest:
    ens_members     = [*range(1,53)]

    try:
        fcst_data = pd.read_pickle("./pickle_dfs/"+site+".pkl")   
    except:
        warnings.warn("Pickle file missing. Creating forecast database using \
                individual ensemble files")
        fcst_data = df_creator(rt_dir, date_range, site_comID, ens_members)

    fcst_data = fcst_data.sort_index().loc(axis=0)[
        (slice(None), slice(None), slice(date_range[0], date_range[1]))
    ]

    return site, obs_dir, fcst_data

# %% DIFFERENT DMB APPROACHES:

# bias correction approaches:
fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]

# for multiple horizons add the required day no. as an array element:
days  = range(1,11)

# window length:
win_len = 3

det_verif, prob_verif = dmb_vars_test(fcst_types, days, win_len, site, fcst_data, obs_dir)
# %% 
fig, fig_crps = kge_crps_plttr (det_verif, prob_verif, site, fcst_types)


# %% Plot KGE horizon:

fig.show(renderer = 'iframe')

# %% Plot CRPS 
fig_crps.show(renderer = 'iframe')