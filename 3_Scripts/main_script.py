# %%
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *


# %% 
"""
####### Initialization parameters and get frcst data ######
"""
def get_fcst_data ():
    # site name and associated comID:
    site            = "Marsyangdi"
    ## set file path:
    rt_dir          = r"../1_Data/Fcst_data"
    obs_dir         = r"../1_Data/obs_data"
    site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
    # date list of interest:
    date_range      = ['20150101', '20151231']
    ens_members     = [*range(1,53)]

    try:
        fcst_data = pd.read_pickle("./pickle_dfs/"+site+".pkl")   
    except:
        warnings.warn("Pickle file missing. Creating forecast database using \
                individual ensemble files")
        fcst_data = df_creator(rt_dir, date_range, site_comID, ens_members)

    # reset index 
    ## FIX NEEDED: include reset index in the df_creator file, and recreate the forecast files. 
    fcst_data = fcst_data.sort_index().loc(axis=0)[
        (slice(None), slice(None), slice(date_range[0], date_range[1]))
    ]

    return site, obs_dir, fcst_data

# %% ######################################### %% #
############# Load Forecast Data ################

site, obs_dir, fcst_data = get_fcst_data ()
# fcst_data is the original df used for all later calculations:

# -----------------******--------------------- #
 
# %% ######################################### %% #
##### Calibration for multiple horizons ###########
# calibrator runs across multiple window lengths for DMB correction:

days = [5, 7, 9, 10]
for day in days:
    # add observations:
    [fcst_data_day, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

    # optimum calibration:
    lo_verif, hi_verif = fcst_calibrator (fcst_data_day, q70_flo, lo_flo_clim, hi_flo_clim)

    flo_conditions = ["low", "high"]
    for flo_con in flo_conditions:
        fig = calibrtn_plttr (hi_verif, lo_verif, site, day, flo_con = flo_con)
        fig.show()

# %% ######################################### %% #
######### Verification Set Forecast Analysis ############

# for multiple horizons add the required day no. as an array element:
days = [1]
win_len = 3
for day in days:
    # add observations:
    [fcst_data_day, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df =  post_process(fcst_data_day, win_len, 
                            q70_flo, lo_flo_clim, hi_flo_clim)

# ----------------- ******************* --------------------- #
# %%  

day = 1
win_len = 3

# add observations:
[df_data, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

# bias correct and return 2 deterministic forecast outputs and 
# also the overall bias corrected dataframe:
lo_df, hi_df, bc_df, prob_verif =  post_process(df_data, win_len, 
                        q70_flo, lo_flo_clim, hi_flo_clim)


# %% 


    
# %% ######################################### %% #
############## All Plot Functions #################
# 3 in 1 subplot:
fig = time_series_plotter(bc_df.reset_index(), site, day, win_len)
fig.show(renderer = "iframe")

# %% Individual bias correction only:
type = "Qout"
fig1 = time_series_individual(bc_df.reset_index(), site, day, win_len, type)
fig1.show(renderer = "iframe")

