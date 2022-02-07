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

date_range  = ['20140301', '20140630']
rt_dir      = r"../1_Data/Fcst_data"
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
                    # shared_xaxes = True,
                    # shared_yaxes = True,
                    horizontal_spacing = 0.05,
                    vertical_spacing = 0.05,
                    x_title = "date",
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
    test_data = fcst_data.xs(key = 1, level = "day_no")

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
    
    # add observations    
    fig.add_trace( 
        go.Scatter( x = obs.index, y = obs["Obs"], 
            line = {"color":"red"}, name = "5647 Obs", 
            legendgroup = "Obs", showlegend = legend_decide )
        , row = row, col = col 
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
    title_text = "Upstream hydrographs for Marsyangdi",
    title_x = 0.5,
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=-0.1
    )
)

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

# load observations:
obs = pd.read_csv( os.path.join(obs_dir, "Marsyangdi.txt"), 
        names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
        infer_datetime_format=True, index_col = [0]).loc[
            slice('20140301', '20140630')    
        ]

# make subplot interface
fig = make_subplots(rows = 6, cols = 2,
                    # shared_xaxes = True,
                    # shared_yaxes = True,
                    horizontal_spacing = 0.05,
                    vertical_spacing = 0.05,
                    x_title = "date",
                    y_title = "River discharge (<i>m<sup>3</sup>/s</i>)",    
                    subplot_titles = site_comIDs
                    )

row_vals  = iter(range(2,7))
row = 1; col = 1

# loop through the upstream COMIDs
for site_comID in site_comIDs:

    legend_decide = True if row == 1 and col == 1 else False

    print(site_comID)
    print(row, col)
    # load the forecast dataset:
    fcst_data = df_creator(rt_dir, date_range, site_comID, [*range(1,53)])
    test_data = fcst_data.xs(key = 1, level = "day_no")

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
    
    # add observations    
    fig.add_trace( 
        go.Scatter( x = obs.index, y = obs["Obs"], 
            line = {"color":"red"}, name = "5647 Obs", 
            legendgroup = "Obs", showlegend = legend_decide )
        , row = row, col = col 
    )

    if col % 2 == 0:            
        row = next(row_vals)
        col = 1
    else: 
        col = 2          

# set the legend inside the figure:
fig.update_layout(
    title_text = "Upstream hydrographs for Marsyangdi",
    title_x = 0.5,
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="left",
        x=-0.1
    )
)

fig.show()    
save_pth = f'./iframe_figures/Marsyangdi-day1-upstream_hydrographs.html' 
fig.write_html(save_pth)
