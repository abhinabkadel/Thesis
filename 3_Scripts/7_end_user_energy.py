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
site = "Marsyangdi";  date_range = ['20150101', '20151231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:


# %%
"""
####### Compile overall bias corrected dataset ######
"""
days            = range(1,11)
win_len         = 2
complete_data   = []
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
    
    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df, prob_df =  post_process(
        fcst_data_day, win_len, clim_vals)

    print (day)

    bc_df["day"]   = day

    complete_data.append(bc_df)

# combine all 10 days of data
complete_data   = pd.concat(complete_data)
complete_data   = complete_data.set_index('init_date', append=True
    ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
        [['Q_raw', 'Q_dmb', 'Obs']]

# %%
"""
####### Extract a random date for illustration ######
"""
# implement only dates from dry season:

test = complete_data.xs('20150105', level = 'init_date').sort_index()

# Plot the end user received inflow forecast:
fig = go.Figure(
        layout = {
            "xaxis_title"   : "date",
            "yaxis_title"   : "River discharge (<i>m3/s</i>)",    
            "title"         : "10 day streamflow forecast",
            "title_x"       : 0.5
        }  
    )


fig.update_layout(legend=dict(
    yanchor="top",
    y=1.4,
    xanchor="left",
    x=-0.1
))

# plot the ensemble tracers
for i in range(1,53):
    fig.add_trace(
        go.Scatter(
            x = test.xs(i, level = 'ens_mem').index, 
            y = test.xs(i, level = 'ens_mem')["energy_yield"],
            name = "forecast", legendgroup = "forecast",
            showlegend = True if i == 1 else False,
            marker_opacity = 0,
            line = dict(
                width = 2, shape = 'spline', color = 'blue'
            )
        )
    )

# add ensemble median:
fig.add_trace( 
    go.Scatter(x = test.groupby(by = "date").median().index,
            y = test.groupby(by = "date").median()['Q_dmb'],
            name = "ensemble median", legendgroup = "ens-med",
            line = {"color" : "cyan", "shape" : "spline"}
    )
)

fig.show()

# %%
"""
####### Calculate Energy Yield ######
"""
rated_discharge = 30.5 

outage      = 0
efficiency  = 0.9
net_head    = 90.5
# calculate the energy yield (in MWh)
test['energy_yield'] = test['Q_dmb'] * 9.81 * \
    net_head * efficiency * (24 - outage) / 1000
test.loc[test.energy_yield > 1872, 'energy_yield'] = 1872

# plot the ensemble energy generation:
fig = go.Figure(
        layout = {
            "xaxis_title"   : "date",
            "yaxis_title"   : "Energy yield (<i>MWh</i>)",    
            "title"         : "10 day energy yield",
            "title_x"       : 0.5
        }  
    )

fig.update_layout(legend=dict(
    yanchor="top",
    y=1.4,
    xanchor="left",
    x=-0.1
))

# plot the ensemble tracers
for i in range(1,53):
    fig.add_trace(
        go.Scatter(
            x = test.xs(i, level = 'ens_mem').index, 
            y = test.xs(i, level = 'ens_mem')["energy_yield"],
            name = "forecast", legendgroup = "forecast",
            showlegend = True if i == 1 else False,
            marker_opacity = 0,
            line = dict(
                width = 2, shape = 'spline', color = 'blue'
            )
        )
    )

# add ensemble median:
fig.add_trace( 
    go.Scatter(x = test.groupby(by = "date").median().index,
            y = test.groupby(by = "date").median()['energy_yield'],
            name = "ensemble median", legendgroup = "ens-med",
            line = {"color" : "cyan", "shape" : "spline"}
    )
)
fig.show()
