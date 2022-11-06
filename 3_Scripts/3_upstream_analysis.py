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
## Analysis for Marsyangdi:
# 56504 in the watershed, 56382: tributary, 56381: up stream 

date_range  = ['20140401', '20140715']
# date_range  = ['20140301', '20140310']
rt_dir      = r"F:/Toolkit/Thesis_files/Fcst_data"
obs_dir     = r"../1_Data/obs_data"


#%% 
"""
####### Subplots for topmost watersheds ######
"""

site_comIDs = [55543, 55536, 55702, 55534, 55693, 55535]

# load observations:
obs = pd.read_csv( os.path.join(obs_dir, "Marsyangdi.txt"), 
        names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
        infer_datetime_format=True, index_col = [0]).loc[
            slice('20140301', '20140630')    
        ]

# make subplot interface
fig = make_subplots(rows = 4, cols = 2,
                    shared_xaxes = False,
                    shared_yaxes = False,
                    horizontal_spacing = 0.02,
                    vertical_spacing = 0.1,
                    y_title = "River discharge (<i>m<sup>3</sup>/s</i>)",    
                    subplot_titles = [55543, " ", 55536, 55702, 55534, 55693, 55535]
                    )

row_vals  = iter(range(2,5))
row = 1; col = 1

# loop through the upstream COMIDs
for site_comID in site_comIDs:

    legend_decide = True if row == 1 and col == 1 else False

    print(site_comID)
    print(row, col)
    # load the forecast dataset:
    fcst_data = df_creator(rt_dir, date_range, site_comID, [*range(1,53)])
    test_data = fcst_data.xs(key = 2, level = "day_no")

    # add ensemble spread 
    fig.add_trace(
            go.Box(x = test_data.reset_index('date')['date'], 
                y = test_data['Q_raw'], 
                name = 'ens spread', line = {"color":"rosybrown"},
                showlegend = legend_decide
            ), row = row, col = col
        ) 
    
    # add ensemble median:
    fig.add_trace( 
        go.Scatter(x = test_data.groupby(by = "date").median().index,
                y = test_data.groupby(by = "date").median()["Q_raw"],
                name = "ens median", line = {"color":"cyan"},
                legendgroup = "ens-med", showlegend = legend_decide),
        row = row, col = col 
    )

    if row == 2 and col == 1 :
        col = 2
    elif row == 3 and col == 1:
        col = 2
    elif site_comID == 55535:    
        break
    else: 
        row = next(row_vals)
        col = 1
    
# set the legend inside the figure:
fig.update_layout(
    title_text = "<b> Forecasted day 2 hydrographs at source for Marsyangdi",
    title_x = 0.5,
    font_size   = 18,
    margin_r    = 10,
    legend=dict(
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=0.8
    )
)

# adjust location of the y-axis label
# fig['layout']['annotations'][-1]['x'] = 0.01
# change font size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=18)

fig.show()    
save_pth = f'./iframe_figures/Marsyangdi-day1-topmost_hydrographs.html' 
fig.write_html(save_pth)


# %%
"""
####### Intermediate Watersheds ######
"""
# main river segments:
site_comIDs = [55702, 55693, 55747, 55748, 
    55985, 55986, 56168, 56169, 
    56287, 56288, 56471]

site_comIDs = [55747, 56471]

# load observations:
obs = pd.read_csv( os.path.join(obs_dir, "Marsyangdi.txt"), 
        names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
        infer_datetime_format=True, index_col = [0]).loc[
            slice('20140301', '20140630')    
        ]

# make subplot interface
fig = make_subplots(rows = 2, cols = 1,
                    # shared_xaxes = True,
                    # shared_yaxes = True,
                    horizontal_spacing = 0.05,
                    vertical_spacing = 0.1,
                    # x_title = "date",
                    y_title = "<b> River discharge (<i>m<sup>3</sup>/s</i>)",    
                    subplot_titles = site_comIDs
                    )

# row_vals  = iter(range(2,3))
row = 1; col = 1

# loop through the upstream COMIDs
for site_comID in site_comIDs:

    legend_decide = True if row == 2 and col == 1 else False

    print(site_comID)
    print(row, col)
    # load the forecast dataset:
    fcst_data = df_creator(rt_dir, date_range, site_comID, [*range(1,53)])
    test_data = fcst_data.xs(key = 1, level = "day_no")

    # add ensemble spread 
    fig.add_trace(
            go.Box(x = test_data.reset_index('date')['date'], 
                y = test_data['Q_raw'], 
                name = f"{site_comID:d} ensemble spread", line = {"color":"rosybrown"},
                showlegend = legend_decide
            ), row = row, col = col
        ) 
    
    # add ensemble median:
    fig.add_trace( 
        go.Scatter(x = test_data.groupby(by = "date").median().index,
                y = test_data.groupby(by = "date").median()["Q_raw"],
                name = "ens median", line = {"color":"cyan"},
                legendgroup = "ens-med", showlegend = legend_decide),
        row = row, col = col 
    )
    
    # add observations    
    fig.add_trace( 
        go.Scatter( x = obs.index, y = obs["Obs"], 
            line = {"color":"red"}, name = "5647 Obs", 
            legendgroup = "Obs", showlegend = legend_decide )
        , row = 2, col = 1 
    )

    if col % 2 == 0:            
        # row = next(row_vals)
        row = 1
    else: 
        row = 2          

# set the legend inside the figure:
fig.update_layout(
    title_text = "<b> Forecasted day 2 hydrographs at source for Marsyangdi",
    title_x = 0.5,
    font_size   = 25,
    margin_r    = 10,
    legend=dict(
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=0.8
    )
)

# adjust location of the y-axis label
# fig['layout']['annotations'][-1]['x'] = 0.01
# change font size
for i in fig['layout']['annotations']:
    i['font'] = dict(size=25)


fig.show()    
save_pth = f'./iframe_figures/Marsyangdi-day1-upstream_hydrographs.html' 
fig.write_html(save_pth)


# %%
def runoff_data_creator(site, date_range):
# Downscales runoff forecasts from the gridded values to catchment scale. 

    # Downscaling weights for each of the sites were calculated once and saved into csv:
    wt_df = pd.read_csv("./pickle_dfs/wts_" + site + ".csv")

    rt_dir      = r"F:/Toolkit/Thesis_files/runoff_forecasts"
    
    init_date_list = pd.date_range(start = date_range[0], 
                        end = date_range[1]).strftime("%Y%m%d").values
    runoff_data = []
    for init_date in init_date_list:
        # loop through the ensemble members:
        for ens_mem in np.arange(1,52): 
            # forecast filter points for high/low res forecasts:
            filtr_pts = wt_df

            # load the forecast files:
            fname       = f'runoff_{ens_mem:d}.nc'    
            file_pth    = os.path.join(rt_dir, init_date, fname)    
            data        = xr.open_dataset(file_pth)

            runoff_vals = []

            # resample the forecasts to daily 
            test = data.RO.resample(time = '1D').mean()
            
            # loop through the forecast grids that intersect with the 
            # catchment:
            for i in range(len(filtr_pts)):

                # substitution to make code readable:
                easy_var    = test.sel(lon = filtr_pts.lon[i],
                            lat = filtr_pts.lat[i],
                            method= 'nearest')

                easy_var[:] = easy_var * filtr_pts.weight[i]/100 \
                    * filtr_pts.grid_area[i]

                runoff_vals.append(easy_var)

            # sum the runoff values to produce total runoff time series 
            # for the catchment:
            catch_RO = np.sum(runoff_vals, axis = 0)
            df = pd.DataFrame(
                {
                    'runoff': catch_RO
                },
                index = test.time.values
            )
            df.index.name = 'date'

            # set the ensemble value based on the range index
            df['ens_mem'] = ens_mem

            # add in information on initial date:
            df["init_date"] = init_date

            # # specify the day of the forecast
            df["day_no"] = 1 + (df.index.get_level_values('date') -  
                            pd.to_datetime(init_date, 
                                format = '%Y%m%d')).days 

            runoff_data.append(df)

        # end for ensemble list
    # end for forecast run dates

    runoff_data = pd.concat(runoff_data)
    runoff_data.set_index(['ens_mem', 'day_no'] , append = True, inplace = True )
    runoff_data = runoff_data.reorder_levels(["ens_mem", "day_no", "date"]).sort_index()

    runoff_data.to_pickle("./pickle_dfs/" + site + "_runoff.pkl")

    return runoff_data

# %%
site        = "55535"
date_range  = ['20140401', '20140630']
runoff_data = runoff_data_creator(site, date_range)

day = 2
plt_data = runoff_data.xs(day, level = 'day_no').reset_index()
if day == 11:
    plt_data = plt_data.drop(52)
# plt_data = runoff_data.xs(3, level = 'day_no').loc(axis=0
#     )[(slice(None), slice('20150103', '20150105'))].reset_index()
fig = time_series_individual(plt_data, site, day, fcst_type = 'runoff')
fig.update_layout(
    title_text = f"<b> Day {day} runoff forecast time series"+
                                f"<br> site = {site} </b>" 
)
fig.show()

save_pth = f'../4_Results/07-runoff-time_series/{site}-day_{day}-runoff.html' 
fig.write_html(save_pth)
