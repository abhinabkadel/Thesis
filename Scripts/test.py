# %%
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as mdates  
import os 

#%% Defined functions:
def df_creator(file_pth, riv_id, fileindex, init_date):
    data  = xr.open_dataset(file_pth)
    Qfcst = data.sel(rivid = riv_id)

    # resample the data to daily 
    df    = Qfcst.Qout.to_dataframe()
    df    = df.resample('D').mean()
    df.index.names = ['date']

    # set the ensemble value based on the range index
    df['ens_mem'] = fileindex
    df.set_index('ens_mem' , append = True, inplace = True )

    # add in information on initial date:
    df["init_date"] = init_date

    # specify the day of the forecast
    df["day_no"] = 1 + (df.index.get_level_values('date') -  
                    pd.to_datetime(init_date, 
                        format = '%Y%m%d')).days 

    return df

# function to calculate the DMB ratio
def dmb_calc(df):
    return sum (df.Qout) / sum (df.Obs)

# function to 



# %% Initialization of variables
rt_dir          = r"D:\Masters\Thesis\Test_downloads"
obs_dir         = r"D:\Masters\Thesis\reanalysis_data"
site            = "Naugad"
init_date_list  = pd.date_range(start='20140101', end='20140105').strftime("%Y%m%d").values
# init_date       = "20140101"
# river ids for Naugad in different renditions:
# riv_id    = 25681
riv_id          = 54302


# %% Loop through all the files:
fcst_data = []
for init_date in init_date_list:
    for i in range(1, 3):
        fname       = f"Qout_south_asia_mainland_{i:d}.nc"
        file_pth    = os.path.join(rt_dir, init_date,fname)
        # mini_df contains data of individual ens member
        mini_df = df_creator(file_pth, riv_id, i, init_date)
        # df_holder.append(mini_df)
        fcst_data.append(mini_df)
    # end for ensemble list
# end for montlhly file list

# %% Concatenate the mini dfs to create a large dataset
fcst_data = pd.concat(fcst_data)

# %% post processsing and make beautiful
# add a new column with the forecast day
# fcst_data["day_no"] = 1 + (fcst_data.index.get_level_values('date') -  
#     pd.to_datetime(fcst_data["init_date"], format = '%Y%m%d')).days 
fcst_data.set_index('day_no' , append = True, inplace = True )
fcst_data = fcst_data.reorder_levels(["ens_mem", "day_no", "date"])

# %% Check the head of data:
fcst_data.head()

# %% Check the tail of data:
fcst_data.tail()

## ************************************** ##
## Dataframe creation is complete ##
##

# %% Bias correct each nth day forecast.
# select forecast day:
day = 2

test = fcst_data.xs(key = day, level = "day_no")[["Qout","init_date"]]

# %% Load observation:
# make the data compatible with the fcst dataframe format
obs = pd.read_csv(os.path.join(rt_dir, "Naugad.csv"))
obs.set_index("Dates", inplace = True)
obs.index.names = ['date']
obs.index   = obs.index.map(lambda x:x.split()[0])
obs.index   = pd.to_datetime(obs.index, format = "%d/%m/%Y")
obs_range   = obs.loc[pd.date_range("20140101", "20140117")]
obs.columns = ["Obs"]

# %% try merging the forecasts and the observations datasets together. 
# perform a left join with fcsts being the left parameter:

t1 = pd.merge(test, obs, left_index=True, right_index=True)
t1 = (t1.groupby(by = "ens_mem").
        apply(lambda x: x.assign( DMBn = dmb_calc ) )
        ).sort_index()
t1["Q_bc"] = t1.Qout / t1.DMBn
cols = list(t1)
cols[0], cols[1] = cols[1] , cols[0]
t1 = t1.loc[:,cols]

# %% Plot the data
# reset index:
(t1.reset_index(level=1).sort_index()).index
test.boxplot(column = ["Qout", "Obs", "Q_bc"], by = "date", )

# prepare plot
fig, ax = plt.subplots(2,1, sharex=True, sharey=False)
df.boxplot(column = ["Qout", "Q_bc"], by = "init_date", rot = 45, ax=ax)
ax[0].set_xlabel("")
ax[1].set_xlabel("initial date")
fig.suptitle("")
fig.text(0.03, 0.5, "Flow m3/s", va = "center", rotation = "vertical", fontsize = "large")
# ax[0].set_ylabel("Flow m3/s")
fig.subplots_adjust(left = 0.12, hspace = 0.5)
df.loc[lambda x : ( x["ens_mem"] == 1 ) | ( x["ens_mem"] == 2 ), :]