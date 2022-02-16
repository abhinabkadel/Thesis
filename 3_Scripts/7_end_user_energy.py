# %%
import pandas as pd
import warnings

from sympy import det
# import all functions:
from calc_funcs import *
from plt_funcs import *
import random

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
# days            = [1,2]
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
        [['Q_raw', 'Q_dmb', 'Obs', 'day']]

# %% bias correction getting perfect information:
vat = bc_df.loc[slice(1,2), slice("20150101" , "20150106"),: ]



# %%
"""
####### Get monthly climatological data ######
"""
obs_data =  pd.read_csv( os.path.join(obs_dir, site+".txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])
obs_mon = obs_data.resample('M').mean()
obs_mon["month"] = obs_mon.index.month
obs_mon = obs_mon.groupby("month").min()



# %%
"""
####### Extract a random date for illustration ######
"""
# list of low flow dates:
q70_flo     = obs_data.quantile(q = 0.7, axis =0, 
            numeric_only = True, interpolation = "linear")[0]
poss_dates  = obs_data[obs_data.Obs < q70_flo].loc[slice("20150101", "20151220")].index
rand_sample = random.choice(poss_dates.strftime('%Y%m%d'))

rand_sample = "20151127"
# extract random date:
test            = complete_data.xs(rand_sample, level = 'init_date').sort_index()
test['month']   = test.index.get_level_values(1).month
test            = test.set_index("month", append=True)
# join the monthly climatological flow data:
test            = test.join(obs_mon, how = "inner", rsuffix= "_mean"
                    ).droplevel("month").drop(["day"], axis  = 1)
# calculate e-flow:                   
test["eflow"]   = test["Obs_mean"] * 0.1
test

#%%
"""
####### Plot the inflow forecast for the random date retrieved ######
"""
# Plot the end user received inflow forecast:
fig = go.Figure(
        layout = {
            "xaxis_title"   : "date",
            "yaxis_title"   : "River discharge (<i>m3/s</i>)",    
            "title"         : "10 day streamflow forecast <br>" + rand_sample,
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
            y = test.xs(i, level = 'ens_mem')["Q_dmb"],
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

# add raw forecast flow:
fig.add_trace( 
    go.Scatter(x = test.groupby(by = "date").median().index,
            y = test.groupby(by = "date").median()['Q_raw'],
            name = "Raw ensemble median", 
            line = {"color" : "green", "shape" : "spline"}
    )
)

# add actual flow:
fig.add_trace( 
    go.Scatter(x = test.xs(key = 1, level = "ens_mem").index,
            y = test.xs(key = 1, level = "ens_mem")['Obs'],
            name = "actual flow", legendgroup = "obs",
            line = {"color" : "red", "shape" : "spline"}
    )
)

fig.show()

# %%
"""
####### Calculate Energy Yield ######
"""
rated_discharge = 30.5 
turbines        = 3
outage      = 0
efficiency  = 0.87
net_head    = 90.5
max_energy  = rated_discharge * turbines * 9.81 * net_head * efficiency * 24/1000 

# implement self consumption plus water cooling usage. dad: 0.2m3/s

# calculate estimated energy yield (in MWh)
# in W : net_flow * gravity + net_head * plant_eff * 
#           hrs_operation * specfic gravity 
test['fcst_energy'] = ( test['Q_dmb'] - test['eflow'] ) * 9.81 * \
    net_head * efficiency * (24 - outage) / 1000
test.loc[test.fcst_energy > max_energy, 'energy_yield'] = max_energy

test['perf_info'] = ( test['Obs'] - test['eflow'] ) * 9.81 * \
    net_head * efficiency * (24 - outage) / 1000
test.loc[test.perf_info > max_energy, 'energy_yield'] = max_energy

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

# add perfect knowledge trace
fig.add_trace( 
    go.Scatter(x = test.xs(key = 33, level = "ens_mem").index,
            y = test.xs(key = 33, level = "ens_mem")['perf_info'],
            name = "perfect knowledge", 
            line = {"color" : "red", "shape" : "spline"}
    )
)

fig.show()

#%% day ahead financial analysis
train_dat  = complete_data[complete_data["Obs"] <= clim_vals["q70_flo"]]

day_ahead  = train_dat.set_index("day", append=True).xs(1, level = "day")
day_ahead  = day_ahead.droplevel("init_date").reorder_levels(["date", "ens_mem"]).sort_index()

# get unique set of observations:
obs         = day_ahead.groupby("date").mean()[["Obs"]]

# calculate the persistance forecast
# (yesterday's observation = forecast for tomorrow)
pers_frcst = obs[["Obs"]].shift(periods=2)
pers_frcst = pers_frcst.rename(columns={'Obs': 'pers_frcst'})

# persistance forecast
day_ahead = day_ahead.join(pers_frcst, how="inner")
day_ahead

# %% deterministic comparison:
det_data   = day_ahead.groupby("date").median()
outage      = 0
efficiency  = 0.9
net_head    = 90.5

det_data['month']   = det_data.index.month
det_data            = det_data.set_index("month", append=True)
# join the monthly climatological flow data:
det_data            = det_data.join(obs_mon, how = "inner", rsuffix= "_mean"
                    ).droplevel("month")
# calculate e-flow:                   
det_data["eflow"]   = det_data["Obs_mean"] * 0.1
det_data


# %% Energy dataframe
det_energy = det_data[["pers_frcst", 'Q_dmb', "Obs"]].sub(det_data["eflow"], axis = 0) \
                * 9.81 * net_head * efficiency * (24 - outage) / 1000
det_energy[det_energy > max_energy] = max_energy

#%% Revenue dataframe
ppa_rate = 8.30
revenue = det_energy

revenue["pers_fine"] = np.where(det_energy["Obs"] < 0.8 * det_energy["pers_frcst"], True, False)
revenue["fcst_fine"] = np.where(det_energy["Obs"] < 0.8 * det_energy["Q_dmb"], True, False)

revenue["pers_half"] = np.where(det_energy["Obs"] > det_energy["pers_frcst"], True, False)
revenue["fcst_half"] = np.where(det_energy["Obs"] > det_energy["Q_dmb"], True, False)

