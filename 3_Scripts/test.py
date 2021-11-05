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
            # fname       = f"Qout_npl_geoglowsn_{i:d}.nc"
            # for Jorge's data:
            fname       = f"Qout_npl_{i:d}.nc"
            
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
# function to add observations:
def add_obs(place, obs_dir, day, fcst_df):
    # Load the observations csv file and load the dataframe
    # make the data compatible with the fcst dataframe format
    obs = pd.read_csv( os.path.join(obs_dir, place+".txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])

    q70_flo     = obs.quantile(q = 0.7, axis =0, 
            numeric_only = True, interpolation = "linear")[0]
    lo_flo_clim = obs[obs["Obs"] <= q70_flo][["Obs"]].mean().values
    hi_flo_clim = obs[obs["Obs"] > q70_flo][["Obs"]].mean().values

    # merge the forecasts and the observations datasets together. 
    # perform a left join with fcsts being the left parameter:
    df = pd.merge( fcst_df.xs(key = day, level = "day_no")
                    [["Qout","init_date"]],
                    obs, left_index=True, 
                    right_index=True).sort_index()
    return df, q70_flo, lo_flo_clim, hi_flo_clim

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

# function to create mean and median databases used later for monthly verification:
def med_mean (df, obs_dir, site_comID):    
    # load climatology data
    obs_clim = pd.read_csv( os.path.join(obs_dir, "clim-"+site_comID+".csv"), 
                names = ["month", "Obs_mean"], header=0, parse_dates=[0], 
                infer_datetime_format=True, index_col = [0])

    # add climatological values:
    df_med  = pd.merge(df_med, obs_clim, on = "month")
    df_mean = pd.merge(df_mean, obs_clim, on = "month")
    df = pd.merge(df, obs_clim, on = "month") 

    return df_med, df_mean, df

# %% PLOT function
# function for all the plotting happening:
def time_series_plotter(df):
    # make subplot interface
    fig = make_subplots(rows = 3, cols = 1,
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing = 0.09,
                        subplot_titles=("Raw", "DMB", "LDMB"),
                        x_title = "date",
                        y_title = "River discharge (<i>m<sup>3</sup>/s</i>)"    
                        )
    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Bias-correction for streamflow forecasts"+
            f"<br> site = {site}, day = {day}, window = {win_len}",
        title_x = 0.5,
        legend_title = "Legend", 
        yaxis_rangemode = "tozero"
        )
    # loop through the forecast types:
    fcst_types = ["Qout", "Q_dmb", "Q_ldmb"]
    for type in fcst_types:
        legend_decide = True if type == "Qout" else False

        # plot ENSEMBLE SPREAD    
        fig.append_trace(
            go.Box(x = df["date"], y=df[type], line = {"color":"rosybrown"},
            name = "ensemble spread", legendgroup = "ens", showlegend = legend_decide), 
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot HIGH-RES
        fig.append_trace( 
            go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
                    y = df[df["ens_mem"] == 52][type],
                    name = "high res", line = {"color":"blue"},
                    legendgroup = "high-res", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot ENS-MEDIAN
        fig.append_trace( 
            go.Scatter(x = df.groupby(by = "date").median().index,
                    y = df.groupby(by = "date").median()[type],
                    name = "ensemble median", line = {"color":"orange"},
                    legendgroup = "ens-med", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot ENS-MEAN
        fig.append_trace( 
            go.Scatter(x = df.groupby(by = "date").mean().index,
                    y = df.groupby(by = "date").mean()[type],
                    name = "ensemble mean", line = {"color":"green"},
                    legendgroup = "ens-mean", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )
        
        # plot OBS:
        fig.append_trace(
                go.Scatter(x = df[df["ens_mem"] == 52]["date"],
                    y=df[df["ens_mem"] == 52]["Obs"], name = "observed",
                    line = {"color":"red"}, mode = "lines+markers", 
                    legendgroup = "obs", showlegend = legend_decide), 
            row = fcst_types.index(type) + 1, col = 1
        )

    return fig

# plot observations
## ADD FINER DATES AS YOU ZOOM IN ##
def plot_obs(obs_dir):
    # setup the figure layout:
    fig = go.Figure(
        layout = {"title_text": "Observation time series",
                    "title_x": 0.5,
                    "yaxis_title" : "discharge (<i>m<sup>3</sup>/s</i>)",
                    # "yaxis_range" : [0, 1200],
                    "legend_title": "Legend",
                    "xaxis_tickangle": 45,
                    "xaxis_dtick":"M2",
                    "xaxis_tickformat": "%b\n%Y",
                    "yaxis_rangemode":"nonnegative",
                    "xaxis_rangeslider_visible":True,
                    "hovermode":"x"
                }
    )   

    # plot all the obs except Marsyangdi HPP:
    for site_name in pd.read_pickle (r"./Sites_info/sites_tbl.pkl").index.values:
        # Marsyangdi HPP is excluded due to only 7 mos. of data in 2020
        if site_name == "Marsyangdi_HPP":
            break
        # read the observation text files
        obs_df = pd.read_csv( os.path.join(obs_dir, site_name+".txt"), 
                names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
                infer_datetime_format=True, index_col = [0])
        # add the line charts for the observations
        fig.add_trace(
                go.Scatter(x = obs_df.index, y = obs_df["Obs"], 
                name = site_name, showlegend = True, mode = "lines")
            )

    return fig 


# %% Initialization of variables
# site name and associated comID:
site            = "Marsyangdi"
## for terminal mode:
rt_dir          = r"./1_Data/Fcst_data"
obs_dir         = r"./1_Data/obs_data"
site_comID      = pd.read_pickle (r"./3_Scripts/Sites_info/sites_tbl.pkl").loc[site].values[0]
## for interactive mode:
# rt_dir          = r"../1_Data/Fcst_data"
# obs_dir         = r"../1_Data/obs_data"
# site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
# date list of interest:
# init_date_list  = np.append( 
#             pd.date_range(start='20200514', end='20200730').strftime("%Y%m%d").values,
#             pd.date_range(start='20200801', end='20201215').strftime("%Y%m%d").values 
#             )
init_date_list  = pd.date_range(start='20150101', end='20151231').strftime("%Y%m%d").values
ens_members     = [*range(1,53)]

# forecast day of interest:
day             = 2
win_len         = 7

# %% Loop through all the files and create a dataframe:
fcst_data = df_creator(rt_dir, init_date_list, site_comID, ens_members)
t2 = fcst_data
## Add a pickle option here ##

# %% Add observations:
[fcst_data, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

#  %% Plot observation time series:
# fig2 = plot_obs(obs_dir)
# # fig2.show()
# fig2.show(renderer = "iframe")

# %% Bias correct the forecasts using DMB and LDMB
t1 = bc_fcsts(df = fcst_data, win_len = win_len )

# %% Ensemble Mean/Median + Add climatology
df = t1.reset_index()
# [df_med, df_mean, df] = med_mean(df, obs_dir, site_comID)

# %% Calculate NSE 
# NSE = nse_calc(df, df_med, df_mean)
# NSE              

def nse_calc(df, df_med, df_mean):
    
    return NSE
# %%
# function to calculate Nash-Scutliffe efficiency:
def nse_form(df, flo_mean, fcst_type = "Q_ldmb"):
    # formula for NSE
    NSE = 1 - \
        ( np.nansum( (df[fcst_type].values - df["Obs"].values) **2 ) ) / \
        ( np.nansum( (df["Obs"].values - flo_mean) **2 ) )
    return NSE

# %% define ens mean, median and high-res dataframes
# calculate ensemble mean and median 
df_med  = df.groupby(by = "date").median().reset_index() \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
df_mean = df.groupby(by = "date").mean().reset_index() \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
df_highres = df[df["ens_mem"] == 52] \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]

# %% divide dfs into low/high flow categories:
metrics = ["median", "mean", "hi-res"]
lo_flo, hi_flo, col_names = [], [], []
for j in metrics:
    # set the deterministic forecast type:
    if j == "median": df = df_med
    elif j == "mean": df = df_mean
    else : df = df_highres 

    # cycle through the raw and bias-corrected forecasts:
    fcst_type = ["Qout", "Q_dmb", "Q_ldmb"]
    for i in fcst_type:
        print (i)
        NSE = nse_form (df = df[df["Obs"] <= q70_flo], fcst_type=i,
                flo_mean = lo_flo_clim)
        lo_flo.append(NSE)
        print(NSE)
        NSE = nse_form (df = df[df["Obs"] > q70_flo], fcst_type=i,
                flo_mean = hi_flo_clim)
        print(NSE)
        hi_flo.append(NSE)

        col_names.append(j+"_"+ i)

NSE = pd.DataFrame([lo_flo, hi_flo], columns = col_names, 
        index = ['low_flow', 'high_flow']).rename_axis("flow_event")
# %% Create time series plot
# fixes to make:
#   ensure that lower y limit is 0
#   scaling based on 
fig = time_series_plotter(df)
# render in a browser:
fig.show(renderer = "browser")
# save as html file locally
# fig.show(renderer = "iframe") 

# %% create observations vs forecasts plot:
fig3 = go.Figure(
        layout = {"title_text": "forecasts vs observations",
                    "title_x": 0.5,
                    "xaxis_title" : "observations (<i>m<sup>3</sup>/s</i>)",
                    "yaxis_title" : "forecasted discharge (<i>m<sup>3</sup>/s</i>)",
                    "yaxis_range" : [0, 4000],
                    "legend_title": "Legend"
                }
    )    
for _, grouped_df in df.groupby('date'): 
    fig3.add_trace(
            go.Box(x = grouped_df["Obs"], y = grouped_df["Qout"], 
            line = {"color":"sandybrown"}, legendgroup = "ens_mem", 
            showlegend = False), 
        )
# add a y = x line for the metric 
fig3.add_trace(
        go.Scatter(x = np.arange(0,1501), y = np.arange(0, 1501),
                    name = "y = x", line = {"color":"black"})
    )
# fig3.show()
fig3.show(renderer = "iframe")

# %%
test = pd.read_csv( os.path.join(obs_dir, "MHPS_DISCHARGE-2077"+".csv"),
            header = 0)
test.head()            
test = pd.melt(test, id_vars = 'Days', var_name = "month", value_name = "discharge" )
test.to_csv(os.path.join(obs_dir, "MHPS_DISCHARGE_long-2077"+".csv"))
