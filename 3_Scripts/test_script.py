# %%
from datetime import date
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *

## to implement:
# calculate performance of individual ensemble members
# What to do when horizon interested is 11-15

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
site = "Marsyangdi";  date_range = ['20150101', '20151231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:


# %%
site_comID      = 56381
date_range      = ['20140101', '20151231']
rt_dir          = r"../1_Data/Fcst_data"
obs_dir         = r"../1_Data/obs_data"
ens_members     = [*range(1,53)]

print(site_comID)

# create the forecast database:
fcst_data = df_creator(rt_dir, date_range, site_comID, ens_members)

# add observations
[fcst_data_day, clim_vals] = add_obs(place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = 1)

# plot the time series
fig = time_series_individual(fcst_data_day.reset_index(), site, 1, "Q_raw")
fig.show()


# %%

# compare the results for the two bias correction approaches:
# Check the difference:
"""
####### Current implementation ######
"""


# %% Plot the observation time series for each of the sites:

# load the observations
sites = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli", "Naugadh"]

# make subplot interface
fig = make_subplots(cols         = 3,
                    rows         = 2, 
                    shared_xaxes = False,
                    shared_yaxes = False,
                    vertical_spacing    = 0.09,
                    horizontal_spacing  = 0.03, 
                    subplot_titles      = 
                        ["Tumlingtar", "Balephi", "", "Marsyangdi", "Trishuli", "Naugadh"],
                    x_title = None,
                    y_title = "River discharge (<i>m<sup>3</sup>/s</i>)"
                    )

# Add figure and legend title                  
fig.update_layout(
    title_text = "<b> Observation time series </b>",
    title_x = 0.50,
    font_size = 18,
    margin_l = 100
    )

# update y axes:
fig.update_yaxes(
    rangemode = "tozero",
    automargin = True,
    )

# adjust horizontal positioning of yaxis title:
fig.layout.annotations[-1]["xshift"] = -55

# increase size of subplot titles:
for i in fig['layout']['annotations']:
    i['font'] = dict(size=20)

colors   = iter(pc.qualitative.D3) 

row = 1; col = 1
for site in sites:

    legend_decide = True if site == "Marsyangdi" else False

    obs_data = pd.read_csv( os.path.join(obs_dir, site+".txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])

    # calculate the q60 flow value
    q60_flo  = obs_data.quantile(q = 0.6, axis = 0, 
            numeric_only = True, interpolation = "linear")[0]

    # add the observations
    fig.add_trace( 
        go.Scatter(
                x = obs_data.index,
                y = obs_data["Obs"],
                name = "observation", line_color = next(colors),
                legendgroup = "obs", showlegend = False 
            ),
        row = row, col = col 
    )

    # add the Q70 flow line:
    fig.add_hline(y = q60_flo, row = row, col = col, )

    # calculate the q95 flow value
    q95_flo  = obs_data.quantile(q = 0.95, axis = 0, 
            numeric_only = True, interpolation = "linear")[0]
    # add the Q95 line:
    fig.add_hline(y = q95_flo, row = row, col = col, line_color = "red")
    
    if site == "Balephi":
        row = 2; col = 1
    else : col += 1

save_pth = f'../4_Results/Obs_time_series/Obs_time-all_sites.html' 
fig.write_html(save_pth)

fig.show(
    
)



    
    
    
