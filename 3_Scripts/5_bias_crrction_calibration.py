from datetime import date
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *

warnings.filterwarnings("ignore")

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
site = "Tumlingtar";  date_range = ['20140101', '20141231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)

# %% ######################################### %% #
##### Calibration for multiple horizons ###########
# calibrator runs across multiple window lengths for dmb correction:

# fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]
days = [4, 7, 9]
# days = range(1,11)

lo_verif    = []
hi_verif    = []
prob_verif  = []
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
    # optimum calibration:
    lo_df, hi_df, prob_df = fcst_calibrator (fcst_data_day, clim_vals)

    lo_df["day"]   = day
    hi_df["day"]   = day
    prob_df["day"] = day

    # one large df with day information
    lo_verif.append(lo_df) 
    hi_verif.append(hi_df)
    prob_verif.append(prob_df)

    flo_conditions = ["low", "high"]

    for flo_con in flo_conditions:
        fig = calibrtn_plttr (hi_df, lo_df, prob_df, site, day, 
                flo_con)     
    
        # save the figure
        save_pth = f'./iframe_figures/{site}-calibration-day_{day}-flow_{flo_con}.html' 
        # fig.show()
        fig.write_html(save_pth)

# create a single large dataframe for low and high flow seasons:
lo_verif    = pd.concat(lo_verif)
hi_verif    = pd.concat(hi_verif)
prob_verif  = pd.concat(prob_verif)

lo_verif["flow_clim"]   = "low"
hi_verif["flow_clim"]   = "high"

det_verif   = pd.concat([lo_verif, hi_verif])

det_verif    = det_verif.set_index(["day", "flow_clim"], append= True
                ).reorder_levels(["win_length", "day", "flow_clim", 
                    "fcst_type", "det_frcst"]).sort_index()
prob_verif  = prob_verif.set_index(["day"], append= True
                ).reorder_levels(["win_length", "day", "flow_clim", 
                    "fcst_type"]).sort_index()

# %%
# use only the ensemble median, drop the raw forecasts as it does 
# determine which calibration window is best. 
# level = [1,2] = [day, flow_clim]

# produces output in (window length, fcst_type) for each horizon
# and flow climatology
det_sub  = det_verif.xs("median", level = "det_frcst").drop(
            'Q_raw', level = 'fcst_type').reset_index(level = [1, 2]
                )[['NSE','KGE', 'day', 'flow_clim']].groupby(['day', 
                    'flow_clim']).apply(lambda x: x.idxmax())
        
prob_sub = prob_verif.drop('Q_raw', level = 'fcst_type').reset_index(level = [1, 2]
                )[['crps', 'day', 'flow_clim']].groupby(['day', 
                    'flow_clim']).apply(lambda x: x.idxmin())

# final metric:
optim_cal_window = det_sub.join(prob_sub)
optim_cal_window
