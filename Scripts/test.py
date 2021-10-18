# %%
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import os 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# changes made:
#   df_creator: file name changed
#   riv_id: changed

# to do:
# replace nan with 0

#%% Defined functions:
# create single database from 52 ens. mems. across given time period:
def df_creator(rt_dir, init_date_list, riv_id, ens_members):
    fcst_data = []
    for init_date in init_date_list:
        for i in ens_members:
            # for ICIMOD archives:
            fname       = f"Qout_npl_geoglowsn_{i:d}.nc"
            # for Jorge's data:
            # fname       = f"Qout_npl_{i:d}.nc"
            
            file_pth    = os.path.join(rt_dir, init_date, fname)

            # Create a mini-dataframe:
            # mini_df contains data of individual ens member
            data  = xr.open_dataset(file_pth)
            Qfcst = data.sel(rivid = riv_id)

            # resample the data to daily 
            df    = Qfcst.Qout.to_dataframe()
            df    = df.resample('D').mean()
            df.index.names = ['date']

            # set the ensemble value based on the range index
            df['ens_mem'] = i

            # add in information on initial date:
            df["init_date"] = init_date

            # specify the day of the forecast
            df["day_no"] = 1 + (df.index.get_level_values('date') -  
                            pd.to_datetime(init_date, 
                                format = '%Y%m%d')).days 

            # append the fcst_data with mini_df
            fcst_data.append(df)
        # end for ensemble list
    # end for montlhly file list

    # concatanate individual mini-dfs to create a dataframe for the time period
    fcst_data = pd.concat(fcst_data)
    fcst_data.set_index(['ens_mem', 'day_no'] , append = True, inplace = True )
    fcst_data = fcst_data.reorder_levels(["ens_mem", "day_no", "date"])
    
    return fcst_data

def add_obs(place, obs_dir, day, fcst_df):
    # Load the observations csv file and load the dataframe
    # make the data compatible with the fcst dataframe format
    obs = pd.read_csv( os.path.join(obs_dir, place+".csv"), 
            names = ["date", "Obs"], header=0, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])


    # merge the forecasts and the observations datasets together. 
    # perform a left join with fcsts being the left parameter:
    df = pd.merge( fcst_df.xs(key = day, level = "day_no")
                    [["Qout","init_date"]],
                    obs, left_index=True, 
                    right_index=True).sort_index()
    return df

# function to calculate the DMB ratio:
def dmb_calc(df, window, weight = False):
    if weight == True:
        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Qout'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.vstack(Q_wins) * wts * np.reciprocal(np.vstack(Obs_wins))
        # add padding and sum the array
        df["LDMB"] = np.pad(np.sum(wt_DMB, axis = 1), pad_width = (window-1,0), mode = "constant", constant_values = np.nan)
        return df

    else:
        return df.Qout.rolling(window).sum().values / df.Obs.rolling(window).sum().values

# create bias-corrected forecasts:
def bc_fcsts(df, win_len):     
    # Calculate DMB ratio:
    # un-weighted:
    df["DMB"] = dmb_calc(df.groupby(by = "ens_mem", dropna = False), window = win_len)
    # weighted DMB:
    df = df.groupby(by = "ens_mem").apply(lambda x:dmb_calc(x, window = win_len, weight =  True))

    # APPLY BIAS CORRECTION FACTOR:
    # new column for un-weighted DMB bias correction: 
    df = df.groupby(by = "ens_mem", dropna = False).     \
        apply(lambda df:df.assign(
            Q_dmb = df["Qout"].values / df["DMB"].shift(periods=1).values )
            ).sort_index()
    # new column for weighted DMB bias correction:
    df = df.groupby(by = "ens_mem", dropna = False).     \
        apply(lambda df:df.assign(
            Q_ldmb = df["Qout"].values / df["LDMB"].shift(periods=1).values )
            ).sort_index()

    return df

# %% Initialization of variables
# for terminal mode:
# rt_dir          = r"./Fcst_data"
# obs_dir         = r"./reanalysis_data"
# for interactive mode:
rt_dir          = r"../Fcst_data"
obs_dir         = r"../reanalysis_data"
site            = "Marsyangdi"
site_comID      = "5020959"
init_date_list  = np.append( 
            pd.date_range(start='20200514', end='20200730').strftime("%Y%m%d").values,
            pd.date_range(start='20200801', end='20201215').strftime("%Y%m%d").values 
            )
ens_members     = [*range(1,53)]
# ens_members     = [*range(1, 5), 52]
# river ids for Naugad in different renditions:
# riv_id    = 25681
# riv_id          = 54302
# river id for Marsyangdi:
riv_id = 34580

# forecast day of interest:
day             = 2
win_len         = 7

# %% Loop through all the files and create a dataframe:
fcst_data = df_creator(rt_dir, init_date_list, riv_id, ens_members)

# %% Add observations:
fcst_data = add_obs(place = site, fcst_df = fcst_data, 
                obs_dir = obs_dir, day = day)

# %% Bias correct the forecasts using DMB and LDMB
t1 = bc_fcsts(df = fcst_data, win_len = win_len )

# %% Add plotting functions
df = t1.reset_index()
# make subplot interface
fig = make_subplots(rows = 3, cols = 1,
                    shared_xaxes = True,
                    shared_yaxes = True,
                    vertical_spacing = 0.09,
                    subplot_titles=("Raw", "DMB", "LDMB"),
                    x_title = "date",
                    y_title = "River discharge (<i>m<sup>3</sup>/s</i>)"    
                    )
#                     
fig.update_layout(
    title_text = "Bias-correction for streamflow forecasts"+
         f"<br> site = {site}, day = {day}, window = {win_len}",
    title_x = 0.5,
    legend_title = "Legend",
    )

# plot raw forecasts:
fig.append_trace(
        go.Box(x = df["date"], y=df["Qout"], line = {"color":"rosybrown"},
        name = "ensemble spread", legendgroup = "ens", showlegend = True), 
        row = 1, col = 1
    )
# plot un-weighted bias corrected forecasts:
fig.append_trace(
        go.Box(x = df["date"], y=df["Q_dmb"], line = {"color":"rosybrown"},
        name = "ensemble spread", legendgroup = "ens", showlegend = False), 
        row = 2, col = 1
    )
# plot linearly weighted bias corrected forecasts:
fig.append_trace(
        go.Box(x = df["date"], y=df["Q_ldmb"], line = {"color":"rosybrown"},
        name = "ensemble spread", legendgroup = "ens", showlegend = False), 
        row = 3, col = 1
    )
# plot high-res
fig.append_trace( 
    go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
               y = df[df["ens_mem"] == 52]["Qout"],
               name = "high res", line = {"color":"blue"},
               legendgroup = "high-res"),
    row = 1, col = 1
    )
# plot high-res dmb
fig.append_trace( 
    go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
               y = df[df["ens_mem"] == 52]["Q_dmb"],
               name = "high res DMB", line = {"color":"blue"},
               legendgroup = "high-res", showlegend = False),
    row = 2, col = 1
    )
# plot high-res linearly-weighted
fig.append_trace( 
    go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
               y = df[df["ens_mem"] == 52]["Q_ldmb"],
               name = "high res LDMB", line = {"color":"blue"},
               legendgroup = "high-res", showlegend = False),
    row = 3, col = 1
    )

# plot ensemble-median:
# raw:
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").median().index,
               y = df.groupby(by = "date").median()["Qout"],
               name = "ensemble median", line = {"color":"orange"},
               legendgroup = "ens-med", showlegend = True),
    row = 1, col = 1
    )
# dmb:
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").median().index,
               y = df.groupby(by = "date").median()["Q_dmb"],
               name = "ensemble median", line = {"color":"orange"},
               legendgroup = "ens-med", showlegend = False),
    row = 2, col = 1
    )
# ldmb
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").median().index,
               y = df.groupby(by = "date").median()["Q_ldmb"],
               name = "ensemble median", line = {"color":"orange"},
               legendgroup = "ens-med", showlegend = False),
    row = 3, col = 1
    )

# plot ENSEMBLE MEAN:
# raw:
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").mean().index,
               y = df.groupby(by = "date").mean()["Qout"],
               name = "ensemble mean", line = {"color":"green"},
               legendgroup = "ens-mean", showlegend = True),
    row = 1, col = 1
    )
# dmb:
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").mean().index,
               y = df.groupby(by = "date").mean()["Q_dmb"],
               name = "ensemble mean", line = {"color":"green"},
               legendgroup = "ens-mean", showlegend = False),
    row = 2, col = 1
    )
# ldmb
fig.append_trace( 
    go.Scatter(x = df.groupby(by = "date").mean().index,
               y = df.groupby(by = "date").mean()["Q_ldmb"],
               name = "ensemble mean", line = {"color":"green"},
               legendgroup = "ens-mean", showlegend = False),
    row = 3, col = 1
    )
# plot observations
fig.append_trace( 
    go.Scatter(x = df["date"], y = df["Obs"], name = "observations", 
                line = {"color":"red"}, mode = "markers", legendgroup = "obs"),
    row = 1, col = 1
    )
for i in [2,3]:
    fig.append_trace(
            go.Scatter(x = df["date"], y=df["Obs"], line = {"color":"red"},
            mode = "markers", legendgroup = "obs", showlegend = False), 
        row = i, col = 1
        )

# Add date slider:
# steps = []
# for i in range(0, len(fig.data), 2):
#     step = dict(
#         method = "restyle",
#         args   = ["visible", [False] * len(fig.data)],
#     )
#     step["args"][1][i:i+2] = [True, True]
#     steps.append(step)


# fig.show()
# render in a browser:
# fig.show(renderer = "browser")
# save as html file locally
fig.show(renderer = "iframe")

# %% implement Nash-Scutliffe efficiency:
# load climatology data:
obs_clim = pd.read_csv( os.path.join(obs_dir, "clim-"+site_comID+".csv"), 
            names = ["month", "Obs_mean"], header=0, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])

df['month'] = df['date'].dt.month 
df = pd.merge(df, obs_clim, on = "month")

# %%
def nse_calc(df):
    # print (df["Obs_mean"])
    NSE = 1 - \
        ( sum( (df["Q_ldmb"].values - df["Obs"].values) **2 ) ) / \
        ( sum( (df["Obs"].values - df["Obs_mean"].values) **2 ) )
    return NSE

# test = df[df["ens_mem"] == 52].sort_values(
#         'date', ascending=False).groupby('month').head(3)
        
# NSE = test.groupby(by = ["month", "Obs_mean"],  dropna = False). \
#         apply(lambda x:nse_calc(x))
        
NSE = df.groupby(by = ["month", "Obs_mean"],  dropna = False). \
        apply(lambda x:nse_calc(x)).reset_index()

# %%
test = pd.read_csv( os.path.join(obs_dir, "MHPS_DISCHARGE-2077"+".csv"),
            header = 0)
test.head()            
test = pd.melt(test, id_vars = 'Days', var_name = "month", value_name = "discharge" )
test.to_csv(os.path.join(obs_dir, "MHPS_DISCHARGE_long-2077"+".csv"))
