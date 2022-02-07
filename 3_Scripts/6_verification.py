import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *

warnings.filterwarnings("ignore")

# %% 
"""
####### Initialization parameters and get frcst data ######
"""
def get_fcst_data (date_range, site):

    ## set file path:
    rt_dir          = r"../1_Data/Fcst_data"
    obs_dir         = r"../1_Data/obs_data"
    site_comID      = pd.read_pickle (
            r"./Sites_info/sites_tbl.pkl").loc[site].values[0]

    print(site_comID)
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

    return obs_dir, fcst_data

# %% ######################################### %% #
############# Load Forecast Data ################
site = "Marsyangdi";  date_range = ['20150101', '20151231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:

# %% ######################################### %% #
"""
####### Perform Verfication ######
"""

# for multiple horizons add the required day no. as an array element:
# days = [1,3,7]
days  = range(1,11)

win_len = 2
lo_verif    = []
hi_verif    = []
prob_verif  = []

for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df, prob_df =  post_process(
        fcst_data_day, win_len, clim_vals)

    lo_df["day"]   = day
    hi_df["day"]   = day
    prob_df["day"] = day
    
    # one large df with day information
    lo_verif.append(lo_df) 
    hi_verif.append(hi_df)
    prob_verif.append(prob_df)
    
    # # plot the time_series for different window length:
    # fig = time_series_plotter(bc_df.reset_index(), site, day)
    # save_pth = f'./iframe_figures/{site}-Q_time_series-day_{day}.html' 
    # # fig.show()
    # fig.write_html(save_pth)

    # # Plot the DMB:
    # fig = dmb_vars_plttr (bc_df, dmb_vars = ['dmb'])
    # save_pth = f'./iframe_figures/{site}-day_{day}-dmb_time_series.html' 
    # # fig.show()
    # fig.write_html(save_pth)

# create a single large dataframe for low and high flow seasons:
lo_verif        = pd.concat(lo_verif)
hi_verif        = pd.concat(hi_verif)
prob_verif      = pd.concat(prob_verif)

lo_verif["flow_clim"]   = "low"
hi_verif["flow_clim"]   = "high"

det_verif   = pd.concat([lo_verif, hi_verif])

det_verif    = det_verif.set_index(["day", "flow_clim", "fcst_type"], append= True
                ).reorder_levels(["day", "flow_clim", "fcst_type", "det_frcst"]).sort_index()
prob_verif  = prob_verif.set_index(["day"], append= True
                ).reorder_levels(["day", "flow_clim", "fcst_type"]).sort_index()


"""
####### Plot the verification results ######
"""
# plot the crps vs horizon:
fig  = crps_horizon_plttr(prob_verif, site)
fig.show(renderer = 'iframe')

# %%
# plot the deterministic metrics vs horizon:
fig  = det_skill_horizon_plttr(det_verif, site)
fig.show(renderer = 'iframe')
