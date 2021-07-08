# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt
import matplotlib.dates as mdates  
import os 
from scipy import stats as st
# %%
rt_dir      = r"D:\Masters\Thesis\Test_downloads"
init_date   = "20140101"
# river ids for Naugad in different renditions:
# riv_id      = 25681
riv_id      = 54302

#%%
# figure set up:
fig, ax = plt.subplots()
plt.xlabel ('time'); plt.ylabel ('Qout')
steps = mdates.DayLocator(interval = 1)
ax.xaxis.set_major_locator(steps)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %H"))
plt.xticks (rotation = 90)

fcst = []

for i in range(1, 53):
    # fname = f"Qout_npl_geoglowsn_{i:d}.nc"
    fname = f"Qout_south_asia_mainland_{i:d}.nc"
    file_pth = os.path.join(rt_dir, init_date,fname)

    # load data
    data = xr.open_dataset(file_pth)

    # subsetting data of specific river id and extract time data
    Qfcst   = data.sel(rivid = riv_id)
    # extract Timestamp object from the np.dateTime64 object:
    t_ax    = Qfcst.time.values.astype('datetime64[s]')
    t_ax    = t_ax.astype(dt.datetime)
    t_labels = [date.strftime("%y-%m-%d %Hh") for date in t_ax]

    fcst.append(Qfcst)

    if i == 52:
        line = ax.plot( t_ax, Qfcst.Qout.values, 'k' )
    else:
        ax.plot( t_ax, Qfcst.Qout.values, linewidth = 0.5 , alpha = 0.7 )
plt.legend( line , ['high-res'] )
plt.show()

# %% Prepare data to fit distributions:
fcst_vals = np.zeros(52)
for i in range(len(fcst)):
    # print(i)
    fcst_vals[i] = fcst[i].sel(time = '2020-04-05T18').Qout.values

# %%
# try some bias correction:



# %% create Gaussian fit
mean, var   = st.distributions.norm.fit(fcst_vals)
x = np.linspace(0, 5.25, len(fcst_vals))

fitted_data = st.distributions.norm.pdf(x, mean, var)

plt.hist(fcst_vals, bins = 20, density = True, alpha = 0.5), 
plt.plot(x, fitted_data, 'r-')
plt.title("Guassian distribution on ensemble output")
plt.xlabel("Forecasted streamflow in (cu. m/sec) for 2020-04-05T18")
plt.ylabel("Probability?")

# %% create Gamma fit
mean, var   = st.gamma.fit(fcst_vals)
x = np.linspace(0, 5.25, len(fcst_vals))

fitted_data = st.distributions.norm.pdf(x, mean, var)

plt.hist(fcst_vals, bins = 20, density = True, alpha = 0.5), 
plt.plot(x, fitted_data, 'r-')
plt.title("Guassian distribution on ensemble output")
plt.xlabel("Forecasted streamflow in (cu. m/sec) for 2020-04-05T18")
plt.ylabel("Probability?")



    
## Add a check to see that the river id corresponds to the desired location


# # %% INDIVIDUAL PLOTS ONLY
# # load data
# data = xr.open_dataset(file_pth)
# # subsetting data of specific river id and extract time data
# Qfcst   = data.sel(rivid = riv_id)
# # extract Timestamp object from the np.dateTime64 object:
# t_ax    = Qfcst.time.values.astype('datetime64[s]')
# t_ax    = t_ax.astype(dt.datetime)
# t_labels = [date.strftime("%y-%m-%d %Hh") for date in t_ax]
# # create a single plot 
# fig, ax = plt.subplots()
# plt.xlabel ('time'); plt.ylabel ('Qout')
# # steps = mdates.DayLocator(interval = 1)
# steps = mdates.DayLocator(interval = 1)
# ax.xaxis.set_major_locator(steps)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d %H"))

# ax.plot( t_ax, Qfcst.Qout.values )

# plt.xticks (rotation = 90)
# plt.show()

# %%
