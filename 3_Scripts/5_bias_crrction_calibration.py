# %%
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
site = "Tumlingtar";  date_range = ['20140101', '20141231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)

# %% ######################################### %% #
##### Calibration for multiple horizons ###########
# calibrator runs across multiple window lengths for dmb correction:

fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]
days = [11]
# days = range(1,16)
# days = [1,3,8,13]

lo_verif    = []
hi_verif    = []
prob_verif  = []
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
    # optimum calibration:
    # lo_df, hi_df, prob_df = fcst_calibrator (fcst_data_day, clim_vals, fcst_types)

    # windows     = [2,5]

    windows     = [2, 3, 5, 7, 10, 15, 20, 30]
    lo_df    = []
    hi_df    = []
    prob_df  = []
    for win_len in windows:
        # l_df, h_df, bc_df, p_df, det_df =  post_process(fcst_data_day, win_len, 
        #                     clim_vals, fcst_types)

        fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]
        if fcst_types == ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]:
            bc_df = bc_fcsts_variations(fcst_data_day, win_len)
        else :
            # Bias correct the forecasts using DMB and LDMB
            bc_df = bc_fcsts(df = fcst_data_day, win_len = win_len )

        fcst_types = bc_df.columns[bc_df.columns.str.startswith("Q")].values.tolist()
        # Separate dataframes for deterministic forecasts:
        # df = t1.reset_index()
        df_det = det_frcsts(bc_df.reset_index(), fcst_types)

        # calculate the metrics:
        l_df, h_df = metric_calc(df_det, bc_df, clim_vals, fcst_types)

        print(fcst_types)

        # calculate probabilitic verification (CRPS):
        p_df = prob_metrics(bc_df, clim_vals, fcst_types)        

        l_df["win_length"]     = win_len
        h_df["win_length"]     = win_len
        p_df["win_length"]   = win_len
        lo_df.append(l_df) 
        hi_df.append(h_df)
        prob_df.append(p_df)

    # create a single large dataframe for low and high flow seasons:
    lo_df    = pd.concat(lo_df)
    hi_df    = pd.concat(hi_df)
    prob_df  = pd.concat(prob_df)

    lo_df = lo_df.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    hi_df = hi_df.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    prob_df = prob_df.set_index(["win_length"], append= True
                    ).reorder_levels(["win_length", "flow_clim", "fcst_type"]).sort_index()

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
                flo_con, fcst_types=["Q_dmb_2", "Q_ldmb"])     
    
        # save the figure
        save_pth = f'../4_results/08-calibration/{site}-calibration-day_{day}-flow_{flo_con}.html' 
        fig.show()
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
# use only the ensemble median, drop the raw forecasts as it does not
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

#%%
save_dir         = f"../4_results/08-calibration/{site}_opt_cal.csv"
optim_cal_window.to_csv(save_dir)

            # %%
