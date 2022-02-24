# %%
from matplotlib.pyplot import title
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
site = "Naugadh";  date_range = ['20140101', '20141231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:

# %% Prepare scatter plot:
def scatter_plt (det_df, bc_df, fcst_type = "Q_raw") :
    # Creates forecast vs observation scatter plots. 
    # Individual scatter plot created for a forecast type, 
    # forecast horizon, site and flow conditions
    
    # Setup figure
    fig = go.Figure(
        layout = {
            "xaxis_title"       : "observations (<i>m<sup>3</sup>/s</i>)",
            "yaxis_title"       : "forecasted discharge (<i>m<sup>3</sup>/s</i>)",    
            "yaxis_range"       : [0, 170],
            # "xaxis_range"       : [0, 370],
            "xaxis_rangemode"   : "tozero",
            "font_size"         : 18,
            "title"             : f"forecast horizon = {bc_df.day.unique()[0]}",
            "title_x"           : 0.5,
            "showlegend"        : False,
            "title_yanchor"     : "bottom",
            "title_y"           : 0.92,
            "margin_t"          : 60,
            "margin_r"          : 10, 
            "legend"            : {
                                    "yanchor"   : "top",
                                    "y"         : 0.98,
                                    "xanchor"   : "left",
                                    "x"         : 0.01,
                                    "font_size" : 18
                                }
        }
    )

    ####
    # Plot all forecasts vs observations
    fig.add_trace(
        go.Scattergl(
            x = bc_df['Obs'], y = bc_df[fcst_type], 
            mode = "markers", showlegend = True, 
            name = "all fcst/obs pairs", marker = {"color":"grey"})
    )

    ####
    # add y = x line
    fig.add_trace(
        go.Scattergl(
            x = np.arange(
                    bc_df.Obs.min()*0.95, 
                    min( bc_df.Q_raw.max(),bc_df.Obs.max() ) * 1.05
                ), 
            y = np.arange(
                    min( bc_df.Q_raw.min(),bc_df.Obs.min() ), 
                    min( bc_df.Q_raw.max(),bc_df.Obs.max() )
                ),
                name = "y = x", line = {"color":"black"})
    )


    ####
    # Deterministic forecast plots
    for det_type, det_dat in det_df.groupby(by = "det_frcst"):        

        # colour scheme for the deterministic forecasts
        if det_type == "median":
            colr_val = "cyan"
        elif det_type == "mean" : 
            continue
            # colr_val = "green" 
        else :
            colr_val = "blue"     

        # plot the trace
        fig.add_trace(
            go.Scattergl(x = det_dat["Obs"], y = det_dat[fcst_type],
                name = det_type, mode = 'markers', legendgroup = det_type,
                marker = {"color": colr_val}, showlegend = True
            )                      
        )    

    return fig

#%%
"""
####### Compile overall bias corrected dataset ######
"""
days            = range(1,16)
# days            = [1]
win_len         = 2
complete_data   = []
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
    
    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df, prob_df, det_df =  post_process(
        fcst_data_day, win_len, clim_vals)

    # add day as index 
    bc_df["day"]   = day
    # append complete_data, the list of bias corrected dataframes.
    complete_data.append(bc_df)

    # create the scatter plot
    fig = scatter_plt (det_df, bc_df, fcst_type = "Q_raw")

    # save as html
    save_pth = f'../4_Results/01-Scatter_raw/{site}-day_{day}-raw-scatter.html' 
    fig.write_html(save_pth)

    # save as jpg:
    save_pth = f'../4_Results/01-Scatter_raw/{site}-day_{day}-raw-scatter.jpg' 
    fig.write_image(save_pth)

    # show on screen
    fig.show( )

# combine all 15 days of data
complete_data   = pd.concat(complete_data)
complete_data   = complete_data.set_index('init_date', append=True
    ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
        [['Q_raw', 'Q_dmb', 'Obs', 'day']]


# %%






























# %%
date_range  = ['20140101', '20141231']
site        = 'Tumlingtar'
try:
    runoff_data = pd.read_pickle("./pickle_dfs/" + site + "_runoff.pkl")
except:
    runoff_data = runoff_data_creator(site, date_range)

# %% plot the time series
day = 3
plt_data = runoff_data.xs(day, level = 'day_no').reset_index()
# plt_data = runoff_data.xs(3, level = 'day_no').loc(axis=0
#     )[(slice(None), slice('20150103', '20150105'))].reset_index()
fig = time_series_individual(plt_data, site, day, type = 'runoff')
fig.show('iframe')

# %%
