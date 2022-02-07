
from calc_funcs import *
from plt_funcs import *
import pandas as pd

date_range  = ['20150101', '20150930']
site        = 'Tumlingtar'
try:
    runoff_data = pd.read_pickle("./pickle_dfs/" + site + "_runoff.pkl")
except:
    runoff_data = runoff_data_creator(site, date_range)

# %% plot the time series
day = 3
plt_data = runoff_data.xs(day, level = 'day_no').reset_index()
# plt_data = runoff_data.xs(3, level = 'day_no').loc(axis=0
#     )[(slice(None), slice('20150103', '20150105'))].reset_index()
fig = time_series_individual(plt_data, site, day, type = 'runoff')
fig.show('iframe')