# %%
from datetime import date
import pandas as pd
import warnings
# import all functions:
from calc_funcs import *
from plt_funcs import *

## to implement:
# calculate performance of individual ensemble members
# run the optimization by location
# What to do when horizon interested is 11-15

# %% 
"""
####### Initialization parameters and get frcst data ######
"""
def get_fcst_data (date_range):
    # site name and associated comID:
    site            = "Marsyangdi"
    ## set file path:
    rt_dir          = r"../1_Data/Fcst_data"
    obs_dir         = r"../1_Data/obs_data"
    site_comID      = pd.read_pickle (r"./Sites_info/sites_tbl.pkl").loc[site].values[0]
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

    return site, obs_dir, fcst_data

# %% ######################################### %% #
############# Load Forecast Data ################

site, obs_dir, fcst_data = get_fcst_data ( date_range = ['20150101', '20151231'])
# fcst_data is the original df used for all later calculations:

# -----------------******--------------------- #

# %%
def dmb_calc_variations(df, window, variation = "dmb"):
    
    # implementation based on Dominique:
    if variation == "ldmb":
        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the dmb parameter:
        wt_DMB = np.vstack(Q_wins) * wts * np.reciprocal(np.vstack(Obs_wins))
        # add padding and sum the array
        df[variation] = np.pad(np.sum(wt_DMB, axis = 1), 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)
        return df

    # Vaariation of the original implementation, to make it align with original
    # un-weighted dmb implementation
    # take ratio of the sum of the weighted forecasts and observations
    if variation == "ldmb-var":

        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.sum( np.vstack(Q_wins) * wts, axis = 1) / \
                np.sum(np.vstack(Obs_wins) * wts, axis = 1)
        # add padding and sum the array
        df[variation] = np.pad(wt_DMB, 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)
        return df
        
    # take the sum of the ratios 
    if variation == "dmb-var":

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.sum ( np.vstack(Q_wins)/ \
                np.vstack(Obs_wins) , axis = 1)
        # add padding and sum the array
        df[variation] = np.pad(wt_DMB, 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)

        return df

    # dmb original implementation based on McCollor & Stull, Bourdin 2013:
    # ratio of the sums:
    else:
        df[variation] =  df.Q_raw.rolling(window).sum().values / \
            df.Obs.rolling(window).sum().values
        return df


def bc_fcsts_variations(df, win_len):     
    
    dmb_vars = ["dmb", "ldmb", "dmb-var", "ldmb-var"]
    
    for variation in dmb_vars:
        
        # Calculate dmb ratio:
        df = df.groupby(by = "ens_mem").apply(
            lambda x:dmb_calc_variations(x, window = win_len, variation = variation)
        )    

        # APPLY BIAS CORRECTION FACTOR:
        df = df.groupby(by = "ens_mem", dropna = False).     \
            apply(lambda df:df.assign(
                new_val = df["Q_raw"].values / df[variation].shift(periods=1).values )
            ).rename(
                columns = {'new_val':"Q_"+variation}
            ).sort_index()
    
    return df

# %% Test different bias correction approaches:

day     = 2
win_len = 3
fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]

# add observations:
[fcst_data_day, clim_vals] = add_obs(
place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

bc_df   = bc_fcsts_variations(fcst_data_day, win_len)   
df_det  = det_frcsts(bc_df.reset_index(),
        fcst_types)

lo_verif, hi_verif = metric_calc(df_det, clim_vals, fcst_types)

# %% time series plot for the dmb:
dmb_vars    = ["dmb", "ldmb", "dmb-var", "ldmb-var"]
fig = dmb_vars_plttr (bc_df, dmb_vars)
fig.show()
                
#%% check the runoff gridded data:
rt_dir      = r"../1_Data/runoff_forecasts/May/"
init_date   = '20150510'
fname       = f'runoff_5.nc'
file_pth    = os.path.join(rt_dir, init_date, fname)

data  = xr.open_dataset(file_pth)
untouched = data

# %%
# data array
test = data.RO
lores_filtr = pd.DataFrame(
    {
    'lat' : [27.903, 28.044, 27.903, 28.044],
    'lon' : [84.375, 84.516, 84.516, 84.375],
    'weight' : [17.82, 24.59, 34.78, 58.02],
    'grid_area': [215664872, 215389905, 215664872, 215389905]
    }
)
    
    
# %%

# test = data.RO


for ens_mem in np.arange(1,52): 
    # forecast filter points to for high/low res forecasts:
    filtr_pts = hires_filtr if ens_mem == 52 else lores_filtr

    # load the forecast files:
    fname       = f"Qout_npl_{i:d}.nc"
    
    file_pth    = os.path.join(rt_dir, init_date, fname)    

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

    # set the ensemble value based on the range index
    df['ens_mem'] = i

# # add in information on initial date:
# df["init_date"] = init_date

# # specify the day of the forecast
# df["day_no"] = 1 + (df.index.get_level_values('date') -  
#                 pd.to_datetime(init_date, 
#                     format = '%Y%m%d')).days 
# %% plot of score vs forecast type


# ----------------- ******************* --------------------- #







# %% ######################################### %% #
##### Calibration for multiple horizons ###########
# calibrator runs across multiple window lengths for dmb correction:

fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb", "Q_dmb-var", "Q_ldmb-var"]
# days = [5, 7, 9, 10]
days = [2]
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
    # optimum calibration:
    lo_verif, hi_verif, prob_verif = fcst_calibrator (fcst_data_day, clim_vals)

    flo_conditions = ["low", "high"]

    for flo_con in flo_conditions:
        fig = calibrtn_plttr (hi_verif, lo_verif, prob_verif, site, day, 
                flo_con)     

        fig.show(renderer = 'iframe')
      
# %% ######################################### %% #
######### Verification Set Forecast Analysis ############

# for multiple horizons add the required day no. as an array element:
# days = [1,3,5,7]
days  = range(1,11)
# implement flexible window bias correction approach:
win_len = 3
lo_verif    = []
hi_verif    = []
prob_verif  = []
for day in days:
    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df, prob_df =  post_process(
        fcst_data_day, win_len, clim_vals)

    lo_df["day"]   = day
    hi_df["day"]   = day
    prob_df["day"] = day

    # one large df with day information
    lo_verif.append(lo_df) 
    hi_verif.append(hi_df)
    prob_verif.append(prob_df)

# create a single large dataframe for low and high flow seasons:
lo_verif    = pd.concat(lo_verif)
hi_verif    = pd.concat(hi_verif)
prob_verif  = pd.concat(prob_verif)

lo_verif["flow_clim"]   = "low"
hi_verif["flow_clim"]   = "high"

det_verif   = pd.concat([lo_verif, hi_verif])

det_verif    = det_verif.set_index(["day", "flow_clim", "fcst_type"], append= True
                ).reorder_levels(["day", "flow_clim", "fcst_type", "det_frcst"]).sort_index()
prob_verif  = prob_verif.set_index(["day"], append= True
                ).reorder_levels(["day", "flow_clim", "fcst_type"]).sort_index()



# %% Forecasts vs observations plot:
scatter_plttr(df_det, bc_df, clim_vals, day, site)

# %% Approach to create probabilistic forecast:
# implement the Kernel density estimation
from sklearn.neighbors import KernelDensity
_, _, calib_datset = get_fcst_data ( date_range = ['20140101', '20141231'])

# %% Bandwidth calculation:
# Question: what is the optimum bandwidth?
# Approach 1: best member appraoch:
#       For each horizon:
#           Calculate the smallest absolute error
#       10 time steps:
#           Calculate the variance = bandwidth

# 1 time step = 1 day
days = [*range(1,11)]
win_len = 3
day = 2
# days = [5, 7, 9, 10]
# for day in days:
    
# add observations:
[calib_dat, clim_vals] = add_obs(
    place = site, fcst_df = calib_datset, obs_dir = obs_dir, day = day)

# bias correct and return 2 deterministic forecast outputs and 
# also the overall bias corrected dataframe:
lo_df, hi_df, calib_dat, prob_verif =  post_process(calib_dat, win_len, 
                        clim_vals)

# calculate the absolute error for each timestep
test = calib_dat.groupby("ens_mem").apply(
        lambda x:(x["Q_dmb"] - x["Obs"]).abs()
    ).droplevel(0).groupby(
        "date"
    ).agg('min')

lo_times = calib_dat.loc[52, :].index[calib_dat.loc[52, :]["Obs"] <= q70_flo ]
hi_times = calib_dat.loc[52, :].index[calib_dat.loc[52, :]["Obs"] > q70_flo ]


# %%
# calculate variance:
# used bandwidth = 28.36
bandwidth = test[test.index.isin(lo_times)].var()
bandwidth = 12

rand_date = np.random.choice(calib_dat.index.get_level_values("date").values)
# rand_date = "20140912"

rand_sample = calib_dat.reorder_levels(["date", "ens_mem"]).sort_index().loc[
        rand_date, :
    ]

kde = KernelDensity(kernel="gaussian",
            bandwidth=bandwidth).fit(
                np.reshape(rand_sample["Q_dmb"].values, (-1,1))
            )

minimum = 0.9 * min(rand_sample["Q_dmb"].values) 
maximum = 1.1 * max(rand_sample["Q_dmb"].values)
X_plot = np.linspace(minimum, maximum, 100)[:,np.newaxis]
y_vals = kde.score_samples(X_plot)

fig = go.Figure(
        layout = {
            "xaxis_title" : "River discharge forecast (<i>m<sup>3</sup>/s</i>)",
            "yaxis_title" : "density"    
        }  
    )


# Add figure and legend title                  
fig.update_layout(
    title_text = "Guassian kernel dressing" + 
    f"<br> bandwidth = {bandwidth:.2f}, forecast horizon = {day}," +  
    "date = { pd.to_datetime(str(rand_date)).strftime('%Y%m%d') }",
    title_x = 0.5,
    legend_title = "Legend"
    )

 # add the kernel:
fig.add_trace(
        go.Scatter(x = X_plot[:,0], y=np.exp(y_vals), 
        name = "dressed-kernel", line = {"color":"blue"},
        )
    )

# add the individual members
fig.add_trace(
        go.Scatter(x = rand_sample["Q_dmb"].values, 
            y = np.zeros_like(rand_sample["Q_dmb"].values),
            mode = "markers", 
            name = "ens members", line = {"color":"black"},
        )
    )

# %% ######################################### %% #
############## All Plot Functions #################
# 3 in 1 subplot:
fig = time_series_plotter(bc_df.reset_index(), site, day, win_len)
fig.show(renderer = "iframe")

# %% Individual bias correction only:
type = "Q_raw"
fig1 = time_series_individual(bc_df.reset_index(), site, day, win_len, type)
fig1.show(renderer = "iframe")

# %% Forecast skill vs horizon
fig = det_skill_horizon_plttr(det_verif, site)
fig.show(
    renderer = "iframe"
    )

# %% Forecast skill vs horizon (CRPS)
fig = crps_horizon_plttr(prob_verif, site)
fig.show()
