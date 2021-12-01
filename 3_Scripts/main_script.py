# %%
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
# make plots:
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as pc

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
    date_range      = ['20140101', '20141231']
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

# %% load forecast data
site, obs_dir, fcst_data = get_fcst_data ()

# %% Add observations:
day             = 2
[fcst_data, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

# %% ######################################### %% #
############# Forecast Calibration ################
def fcst_calibrator (fcst_data, q70_flo, lo_flo_clim, hi_flo_clim):

    windows = [2, 3, 5, 7, 10, 15, 20, 30]
    lo_verif = []
    hi_verif = []

    for win_len in windows:
        lo_df, hi_df =  post_process(fcst_data, win_len, 
                            q70_flo, lo_flo_clim, hi_flo_clim)
        lo_df["win_length"] = win_len
        hi_df["win_length"] = win_len 
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)

    # create a single large dataframe for low and high flow seasons:
    lo_verif = pd.concat(lo_verif)
    hi_verif = pd.concat(hi_verif)
    lo_verif = lo_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    hi_verif = hi_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()

    return lo_verif, hi_verif

# function to plot the calibration graphs:
def calibrtn_plttr (hi_verif, lo_verif, site, day, flo_con = "high"):
    if flo_con == "high" : 
        df_big = hi_verif
    elif flo_con == "low" :
        df_big = lo_verif
    else : print ("wrong input")

    df_big = df_big.xs('median', level = 2)

    # make subplot interface
    fig = make_subplots(rows = 2, cols = 1,
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing = 0.09,
                        subplot_titles=("DMB", "LDMB"),
                        x_title = "window length (days)",
                        y_title = "Score",
                        )
    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Verification scores for different window lengths"+
            f"<br> site = {site}, day = {day}, flow season = {flo_con} ",
        title_x = 0.5,
        legend_title = "Legend", 
        )
    # update both y axes:
    fig.update_yaxes(
        rangemode = "tozero",
        range = [0.5, 1.1]
        )

    # loop through the forecast types:
    fcst_types = ["Q_dmb", "Q_ldmb"]
    for type in fcst_types:
        df = df_big.xs(type, level = 1)
        
        # show only one legend entry per verification metric:
        legend_decide = True if type == "Q_dmb" else False
        
        # define color to be used:
        color   = pc.qualitative.D3
        # metrics to plot:
        metrics = ["NSE", "r", "flo_var", "bias", "KGE"]
        
        for metric in metrics:     
            # plot different metrics:
            fig.append_trace( 
                go.Scatter(x = df.index, 
                        y = df[metric], 
                        name = metric,
                        legendgroup = metric,
                        marker_color = color[metrics.index(metric)],
                        showlegend = legend_decide
                        ),
                row = fcst_types.index(type) + 1, col = 1
            )

    return fig

# %% calibrate forecast
lo_verif, hi_verif = fcst_calibrator (fcst_data, q70_flo, lo_flo_clim, hi_flo_clim)

# %% plot the calibration curves:
flo_conditions = ["low", "high"]
for flo_con in flo_conditions:
    fig = calibrtn_plttr (hi_verif, lo_verif, site, day, flo_con = flo_con)
    fig.show()

# %% ######################################### %% #
############## Verif Metrics Plot #################





fig.show()

# %%
# forecast day of interest:
day             = 2
win_len         = 7

