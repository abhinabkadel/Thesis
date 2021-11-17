# %% 
# Import necessary modules
# read netcdf files:
import xarray as xr
# dataframe and data analysis
import pandas as pd
import numpy as np
# error metric calculations:
from hydrostats import HydroErr
from scipy import stats
# use os commands:
import os 
# make plots:
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# print warnings:
import warnings

#%% Defined functions:
# create single database from 52 ens. mems. across given time period:
def df_creator(rt_dir, date_range, riv_id, ens_members):
    init_date_list  = pd.date_range(start=date_range[0], 
            end=date_range[1]).strftime("%Y%m%d").values
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
    fcst_data = fcst_data.reorder_levels(["ens_mem", "day_no", "date"]).sort_index()
    
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

# function to create determininstic forecasts:
def det_frcsts (df):    
    # ensemble median:
    df_med  = df.groupby(by = "date").median().reset_index() \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
    # ensemble mean:
    df_mean = df.groupby(by = "date").mean().reset_index() \
        [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
    # high-res forecast
    df_highres = df[df["ens_mem"] == 52] \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]

    return df_med, df_mean, df_highres

# function to calculate Nash-Scutliffe efficiency:
def nse_form(df, flo_mean, fcst_type = "Q_dmb"):
    # formula for NSE
    NSE = 1 - \
        ( np.nansum( (df[fcst_type].values - df["Obs"].values) **2 ) ) / \
        ( np.nansum( (df["Obs"].values - flo_mean) **2 ) )
    return NSE

# correlation, bias and flow variability:
def kge_form(df, fcst_type = "Q_dmb"):
    # calculate pearson coefficient:
    correlation = HydroErr.pearson_r(df[fcst_type], df["Obs"])
    # calculate flow variability error or coef. of variability:
    flow_variability = stats.variation(df[fcst_type], nan_policy='omit') / \
                        stats.variation(df["Obs"], nan_policy='omit')
    # calculate bias:
    bias = df[fcst_type].mean() / df["Obs"].mean()
    # calculate KGE
    KGE  = 1 - (
            (correlation - 1)**2 + (flow_variability - 1)**2 + (bias - 1)**2 
        )**0.5
    # KGE using the HydroErr formula:
    # print(HydroErr.kge_2012(df[fcst_type], df["Obs"]))
    
    return pd.DataFrame(np.array([[correlation, flow_variability, bias, KGE]]))

# function that calculates deterministic verification metrics:
def metric_calc(df_det, q70_flo, lo_flo_clim, hi_flo_clim):
    # defines dataframes for low_flow and high_flow values
    df_low  = df_det[df_det["Obs"] <= q70_flo]
    df_high = df_det[df_det["Obs"] > q70_flo]

    # loop through the two dataframes to create:
    for df in [df_low, df_high]:
        flo_mean = lo_flo_clim if df.equals(df_low) else hi_flo_clim
        
        # loop through win len
        # fcst_Day
        data = []
        fcst_type = ["Qout", "Q_dmb", "Q_ldmb"]
        # loop through the raw and bias corrected forecasts:
        for i in fcst_type:
            # NSE:
            NSE = df.groupby(by = "det_frcst").apply(
                    lambda x:nse_form(x, flo_mean, i)
                )
            # KGE:
            kge = df.groupby(by = "det_frcst").apply(
                    lambda x:kge_form(x, i)
                )

            # concatenate and create a dataframe
            verifs = pd.concat([NSE, kge.droplevel(1)], axis = 1).set_axis([
                "NSE", "r", "flo_var", "bias", "KGE"], axis = 1
                )
            # new index with the fcst_type information:
            verifs["fcst_type"] = i

            data.append(verifs)

        # end for along fcst_type

    if flo_mean == lo_flo_clim:
        lo_verif = pd.concat(data)
    else : hi_verif = pd.concat(data)

    lo_verif = lo_verif.set_index(["fcst_type"], append= True
            ).reorder_levels(["fcst_type", "det_frcst"])
    hi_verif = hi_verif.set_index(["fcst_type"], append= True
            ).reorder_levels(["fcst_type", "det_frcst"])

    return lo_verif, hi_verif


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
                    # "xaxis_dtick":"M2",
                    "xaxis_tickformat": "%b\n%Y",
                    "yaxis_rangemode":"nonnegative",
                    "xaxis_rangeslider_visible":True,
                    "hovermode":"x"
                }
    )   

    fig.update_xaxes(
        tickformatstops = [
            dict(dtickrange=[None, 604800000], value="%e. %b"),
            # weekly view:
            dict(dtickrange=[604800000, "M1"], value="%e. %b"),
            # monthly view:
            dict(dtickrange=["M1", "M12"], value="%b '%y"),

            dict(dtickrange=["M12", None], value="%Y Y")
        ]
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
# rt_dir          = r"./1_Data/Fcst_data"
# obs_dir         = r"./1_Data/obs_data"
# site_comID      = pd.read_pickle (r"./3_Scripts/Sites_info/sites_tbl.pkl").loc[site].values[0]
## for interactive mode:
rt_dir          = r"../1_Data/Fcst_data"
obs_dir         = r"../1_Data/obs_data"
site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
# date list of interest:
# init_date_list  = np.append( 
#             pd.date_range(start='20200514', end='20200730').strftime("%Y%m%d").values,
#             pd.date_range(start='20200801', end='20201215').strftime("%Y%m%d").values 
#             )
date_range      = ['20140101', '20141231']
ens_members     = [*range(1,53)]

# forecast day of interest:
day             = 2
win_len         = 7

# %% Loop through all the files and create a dataframe:
try:
    fcst_data = pd.read_pickle("./pickle_dfs/"+site+".pkl")
    
except:
    warnings.warn("Pickle file missing. Creating forecast database using \
            individual ensemble files")
    fcst_data = df_creator(rt_dir, date_range, site_comID, ens_members)

t2 = fcst_data
# 
fcst_data = fcst_data.sort_index().loc(axis=0)[
    (slice(None), slice(None), slice(date_range[0], date_range[1]))
]
# %% Add observations:
[fcst_data, q70_flo, lo_flo_clim, hi_flo_clim] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

# %% Bias correct the forecasts using DMB and LDMB
t1 = bc_fcsts(df = fcst_data, win_len = win_len )

# %% Separate dataframes for deterministic forecasts:
df = t1.reset_index()
[df_med, df_mean, df_highres] = det_frcsts(df)

# %% Calculate NSE 
NSE = nse_calc(df_med, df_mean, df_highres, q70_flo, lo_flo_clim, hi_flo_clim)

# %% concatenate the 3 deterministic forecast matrices to create 
# a single deterministic dataframe:
df_det = pd.concat([df_med, df_mean, df_highres], keys=["median", "mean", "high-res"])
df_det = df_det.droplevel(1)
df_det.index.names = ["det_frcst"]

# %%
lo_verif, hi_verif = metric_calc(
    df_det, q70_flo, lo_flo_clim, hi_flo_clim)
# %% test the KGE calculation:
correlation, flow_variability, bias, KGE = kge_form(df = df_med)

# %% Create another bulky database:
frames = [df_med, df_mean, df_highres]
result = pd.concat(frames, keys=["median", "mean", "high-res"])

# %% calculate the CRPS:


# %% Create time series plot
# fixes to make:
#   ensure that lower y limit is 0
#   scaling based on 
fig = time_series_plotter(df = t1.reset_index())
# fig.show(renderer = "browser")
fig.show(renderer = "iframe") 

#  %% Plot observation time series for all sites:
# fig2 = plot_obs(obs_dir)
# fig2.show(renderer = "browser")
# # fig2.show(renderer = "iframe")


# %% create observations vs forecasts plot:
# very big file 
fig3 = make_subplots(
        rows = 3, cols = 1,
        shared_xaxes = True,
        shared_yaxes = True,
        vertical_spacing = 0.09,
        subplot_titles=("Raw", "DMB", "LDMB"),
        x_title = "observations (<i>m<sup>3</sup>/s</i>)",
        y_title = "forecasted discharge (<i>m<sup>3</sup>/s</i>)"    
    )

fcst_types = ["Qout", "Q_dmb", "Q_ldmb"]
for type in fcst_types:
    
    legend_decide = True if type == "Qout" else False
    
    for date, grouped_df in df.groupby('date'): 
        if type == "Qout" and date == df["date"][0]:
            legend_decide = True
        else : legend_decide = False

        # # ENS spread
        # fig3.append_trace(
        #     go.Box(x = grouped_df["Obs"], y = grouped_df[type], 
        #     line = {"color":"sandybrown"}, legendgroup = "ens_mem",
        #     name = "ens spread", showlegend = legend_decide),
        # row = fcst_types.index(type) + 1, col = 1
        # )

        # add y = x line
        fig3.append_trace(
            go.Scattergl(x = np.arange(0, max(df.Qout.max(),df.Obs.max())), 
                    y = np.arange(0, max(df.Qout.max(),df.Obs.max())),
                    name = "y = x", line = {"color":"black"}),
        row = fcst_types.index(type) + 1, col = 1
        )

        # ENS-MEAN:
        fig3.add_trace(
            go.Scattergl(x = df_mean["Obs"], y = df_mean[type],
                    name = "ens mean", mode = 'markers',
                    marker = {"color":"green"}),
        row = fcst_types.index(type) + 1, col = 1
        )

        # ENS-MEDIAN:
        fig3.add_trace(
            go.Scattergl(x = df_med["Obs"], y = df_med[type],
                    name = "ens mean", mode = 'markers',
                    marker = {"color":"blue"}),
        row = fcst_types.index(type) + 1, col = 1
        )

        # High-Res:
        fig3.add_trace(
            go.Scattergl(x = df_highres["Obs"], y = df_highres[type],
                    name = "ens mean", mode = 'markers',
                    marker = {"color":"red"}),
        row = fcst_types.index(type) + 1, col = 1
        )

fig3.update_layout(
        title_text = f"forecasts vs observations for {site}",
        title_x = 0.5,
        legend_title = "Legend", 
        yaxis_rangemode = "tozero",
        yaxis_range = [0, df.Qout.max()],
        xaxis_range = [0, df.Obs.max()],
    )

# fig3.show()
# fig3.show(renderer = "browser")
fig3.show(renderer = "iframe")
# %%
