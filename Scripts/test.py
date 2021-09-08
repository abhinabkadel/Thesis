# %%
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as mdates  
import matplotlib.pyplot as plt
import seaborn as sn
import os 

#%% Defined functions:
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
    
# function to calculate the DMB ratiomend
def dmb_calc(df):
    return sum (df.Qout) / sum (df.Obs)

# %% Initialization of variables
# rt_dir          = r"/Users/akadel/Documents/Kadel/Thesis/Fcst_data"
rt_dir          = r"../Fcst_data"
obs_dir         = r"../reanalysis_data"
site            = "Naugad"
init_date_list  = pd.date_range(start='20140101', end='20140105').strftime("%Y%m%d").values
ens_members     = [*range(1, 5), 52]
# river ids for Naugad in different renditions:
# riv_id    = 25681
riv_id          = 54302
# forecast day of interest:
day             = 2

## ************************************** ##
## Build dataframe of raw forecasts
##

# %% Loop through all the files and create a dataframe:
fcst_data = df_creator(rt_dir, init_date_list, riv_id, ens_members)

# %% Check the head of data:
fcst_data.head()
# %% Check the tail of data:
fcst_data.tail()

# %% 
## ************************************** ##
## Load the observation ##
## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ##

# %% Load observation:
# make the data compatible with the fcst dataframe format
obs = pd.read_csv(os.path.join(rt_dir, "Naugad.csv"))
obs.set_index("Dates", inplace = True)
obs.index.names = ['date']
obs.index   = obs.index.map(lambda x:x.split()[0])
obs.index   = pd.to_datetime(obs.index, format = "%d/%m/%Y")
obs_range   = obs.loc[pd.date_range("20140101", "20150131")]
obs.columns = ["Obs"]

# %% try merging the forecasts and the observations datasets together. 
# perform a left join with fcsts being the left parameter:

t1 = pd.merge(fcst_data.xs(key = day, level = "day_no")[["Qout","init_date"]],
                obs, left_index=True, right_index=True)
# calculate the DMB for the calibration year
t1 = (t1.groupby(by = "ens_mem").
        apply(lambda x: x.assign( DMBn = dmb_calc ) )
        ).sort_index()

# %% Set up verification data:
# set verification dates to be 1 year after the calibration date:
verif_date_list = pd.to_datetime(init_date_list).shift(365, freq = "d").strftime("%Y%m%d").values
verif_data      = df_creator(rt_dir, verif_date_list, riv_id, ens_members)
verif_data = verif_data.xs(key = day, level = "day_no")[["Qout","init_date"]].sort_index()
verif_data = pd.merge(verif_data, obs, left_index=True, right_index=True)

# %%
test        = t1[["DMBn"]]
test.index  = test.index.set_levels(test.index.levels[1].shift(365, freq = "d"), 
                level = 1) 

# %%
test = pd.merge(verif_data, test, left_index=True, right_index=True)
test["Q_bc"] = test.Qout / test.DMBn

# %% reorder columns
# cols = list(test)
# cols[0], cols[1] = cols[1] , cols[0]
# t1 = t1.loc[:,cols]

# %% Reset all the indices
# reset index:
df = (test.reset_index().sort_index())
df

# %% prepare plot
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

# %%
# df.loc[lambda x : ( x["ens_mem"] == 1 ) | ( x["ens_mem"] == 2 ), :]

# %%
