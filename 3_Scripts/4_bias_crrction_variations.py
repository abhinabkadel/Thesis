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

def dmb_vars_test(fcst_types, days, win_len, site, fcst_data, obs_dir): 
# tests the 4 different variations of degree of mass balance:

    lo_verif    = []
    hi_verif    = []
    prob_verif  = []
    # loop through the forecast horizons
    for day in days:

        # add observations:
        [fcst_data_day, clim_vals] = add_obs(
        place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

        bc_df   = bc_fcsts_variations(fcst_data_day, win_len)   
        df_det  = det_frcsts(bc_df.reset_index(),
                fcst_types)

        # loop through the two dataframes to create:
        for flo_con in ["low", "high"]:

            # defines dataframes for low_flow and high_flow values
            if flo_con == "low":
                df      = df_det[df_det["Obs"] <= clim_vals["q60_flo"]]
                df_p    = bc_df[bc_df["Obs"] <= clim_vals["q60_flo"]]
                flo_mean = clim_vals["lo_flo_clim"]

            else:
                df      = df_det[df_det["Obs"] > clim_vals["q60_flo"]]
                df_p    = bc_df[bc_df["Obs"] > clim_vals["q60_flo"]]
                flo_mean = clim_vals["hi_flo_clim"]

            # empty list that holds the information on the verification:
            data = []
            
            # loop through the raw and bias corrected forecasts:
            for i in fcst_types:
                # NSE:
                nse = df.groupby(by = "det_frcst").apply(
                        lambda x:nse_form(x, flo_mean, i)
                    )
                # NSE for individual ensemble members as det frcsts:
                nse_all_mem = df_p.groupby(by = "ens_mem").apply(
                        lambda x:nse_form(x, flo_mean, i)
                    )            

                # KGE:
                kge = df.groupby(by = "det_frcst").apply(
                        lambda x:kge_form(x, i)
                    )
                # KGE for individual ensemble members as det frcsts:
                kge_all_mem = df_p.groupby(by = "ens_mem").apply(
                        lambda x:kge_form(x, i)
                    )

                # concatenate and create a dataframe
                verifs = pd.concat(
                            [ pd.concat( [ nse, nse_all_mem ] ), 
                                pd.concat( [ kge.droplevel(1), kge_all_mem.droplevel(1)] )]
                            , axis = 1).set_axis(
                                ["NSE", "r", "flo_var", "bias", "KGE"], axis = 1
                        )
                # new index with the fcst_type information:
                verifs["fcst_type"] = i

                data.append(verifs)

            # end for along fcst_types

            if flo_con == "low":
                lo_df = pd.concat(data)
                lo_df.index.rename("det_frcst", inplace = True)

            else : 
                hi_df = pd.concat(data)
                hi_df.index.rename("det_frcst", inplace = True)

        lo_df, hi_df

        # calculate probabilitic verification (CRPS):
        prob_df = prob_metrics(bc_df, clim_vals, fcst_types)

        lo_df["day"]   = day
        hi_df["day"]   = day
        prob_df["day"] = day

        # one large df with day information
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)
        prob_verif.append(prob_df)

    # create a single large dataframe for low and high flow seasons:
    lo_verif    = pd.concat(lo_verif)
    hi_verif    = pd.concat(hi_verif)
    prob_verif  = pd.concat(prob_verif)

    lo_verif["flow_clim"]   = "low"
    hi_verif["flow_clim"]   = "high"

    det_verif   = pd.concat([lo_verif, hi_verif])

    det_verif    = det_verif.set_index(["day", "flow_clim", "fcst_type"], append= True
                    ).reorder_levels(["day", "flow_clim", "fcst_type", "det_frcst"]).sort_index()
    prob_verif  = prob_verif.set_index(["day"], append= True
                    ).reorder_levels(["day", "flow_clim", "fcst_type"]).sort_index()

    return det_verif, prob_verif


# %% DIFFERENT DMB APPROACHES:
site = "Tumlingtar";  date_range = ['20140101', '20141231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)


# bias correction approaches:
fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]

# for multiple horizons add the required day no. as an array element:
days  = range(1,11)

# window length:
win_len = 3

#%%
det_verif, prob_verif = dmb_vars_test(fcst_types, days, win_len, site, fcst_data, obs_dir)
# %% 
fig, fig_crps = kge_crps_plttr (det_verif, prob_verif, site, fcst_types)


# %% Plot KGE horizon:
fig.show()

# %% Plot CRPS 
fig_crps.show()
# %%
