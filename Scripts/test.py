# %%
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt
import os 

#%% Defined functions:
# create single database from 52 ens. mems. across given time period:
def df_creator(rt_dir, init_date_list, riv_id, ens_members):
    fcst_data = []
    for init_date in init_date_list:
        for i in ens_members:
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
# rt_dir          = r"./Fcst_data"
# obs_dir         = r"./reanalysis_data"
rt_dir          = r"../Fcst_data"
obs_dir         = r"../reanalysis_data"
site            = "Naugad"
init_date_list  = pd.date_range(start='20140101', end='20140110').strftime("%Y%m%d").values
ens_members     = [*range(1, 5), 52]
# river ids for Naugad in different renditions:
# riv_id    = 25681
riv_id          = 54302
# forecast day of interest:
day             = 2
win_len         = 5

# %% Loop through all the files and create a dataframe:
fcst_data = df_creator(rt_dir, init_date_list, riv_id, ens_members)

# %% Add observations:
fcst_data = add_obs(place = "Naugad", fcst_df = fcst_data, 
                obs_dir = obs_dir, day = day)

# %% Bias correct the forecasts using DMB and LDMB
t1 = bc_fcsts(df = fcst_data, win_len = win_len )

# %% Add plotting functions
df = t1
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
fig.suptitle("day 2 forecasts initialised for different dates for 2014 January", 
            y = 0.96 )
fig.text(0.02, 0.5, "Flow ($m^3/s$)", va = "center", rotation = "vertical", 
            fontsize = "large")
fig.subplots_adjust(left = 0.12, hspace = 0.3)

sn.set(style = "darkgrid")
# plot the high-resolution forecast:
p1 = sn.scatterplot(x = "init_date", y = "Qout", data = df[df["ens_mem"] == 52], 
                color = "black", ax = ax[0], label = "high-res", legend = False)
sn.scatterplot(x = "init_date", y = "Q_bc", data = df[df["ens_mem"] == 52], 
                color = "black", ax = ax[1])
# plot the observations:
ax[0].plot(df.groupby("init_date")['Obs'].mean(), "ro", label = "observations")
ax[1].plot(df.groupby("init_date")['Obs'].mean(), "ro")

# plot raw forecasts
sn.violinplot(x = "init_date", y = "Qout", data = df, ax = ax[0], 
                color = "skyblue", width = 0.5 , linewidth = 2)
# plot bias corrected forecasts:
sn.boxplot(x = "init_date", y = "Q_bc", data = df, ax = ax[1], 
                color = "skyblue", width = 0.5)

# aesthetic changes:
ax[0].set_xlabel("")
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[1].set_xlabel("initial date")
ax[0].set_title("Raw forecasts")
ax[1].set_title("bias corrected forecasts")

# add a legend:
fig.legend(loc = "center right",
            title = "Legend")

plt.show()
