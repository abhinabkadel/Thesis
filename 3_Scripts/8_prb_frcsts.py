# %%
from datetime import date
from sklearn.neighbors import KernelDensity
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


"""
####### Dominique's approach ######
"""
# %% 
site = "Marsyangdi";  date_range = ['20150101', '20151231']
obs_dir, calib_datset = get_fcst_data ( date_range, site)

days = [*range(1,11)]
win_len = 2
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

# focus on low flow season only:
df_low  = calib_dat[calib_dat["Obs"] <= clim_vals["q70_flo"]]
df_low = df_low[["Q_dmb", "Obs"]]


# %% Step 1: plot a Talagand diagram

# test = df_low.loc(axis=0)[
#         (slice(1,4), slice("20140410", "20140419"))
#     ].reorder_levels(["date", "ens_mem"]).sort_index()
test = df_low
# compute individual ranks:     
test = test.groupby(by = "date").apply(
    lambda x: x.assign(
        rank = sum( x["Q_dmb"] < x["Obs"].unique()[0] ) + 1
    )
)  

# plot a histogram:
import plotly.express as px
fig = px.histogram(test, x = "rank", histnorm="percent",
    labels = {'percent' : 'count'},
    title = "Rank histogram")
fig.update_layout( yaxis_title_text='frequency', title_x = 0.5 )
fig.show()


# %% Step 2: plot distribution of the errors:

# Calculate the ensemble median:
ens_med = test.groupby(by = "date").median()
# Error of the ensemble median:
error       = ens_med["Obs"] - ens_med["Q_dmb"]
error_ln    = np.log(ens_med["Obs"]) - np.log(ens_med["Q_dmb"])

import plotly.figure_factory as ff
# Plot the empirical distributions of ensemble median errors and 
# the theoretical gaussian distribution:
fig = ff.create_distplot( [error.dropna().values], ['Gaussian'], 
    bin_size=.5, curve_type='normal', show_hist = False, show_rug= False)
fig.add_trace(
        go.Histogram( x = error, histnorm='probability density')
    )
fig.update_xaxes(
    range = [-80, 80]
)
fig.show()

# Plot the empirical distributions of log ensemble median errors and 
# the theoretical gaussian distribution:
fig = ff.create_distplot( [error_ln.dropna().values], ['Gaussian'], 
    bin_size=.5, curve_type='normal', show_hist = False, show_rug= False)
fig.add_trace(
        go.Histogram( x = error_ln, histnorm='probability density')
    )
# fig.update_xaxes(
#     range = [-80, 80]
# )
fig.show()


# %% Errors are Gaussian shape roughly. get the gaussian function:
sample_date = test.xs(key = "20150419", level = "date")

sample_date["Q_dmb"].median()
sample_date["Q_dmb"].std()

# %% training dataset for the EMOS statistics:
site = "Marsyangdi";  date_range = ['20140101', '20141231']
obs_dir, train_datset = get_fcst_data ( date_range, site)

day = 2

# add observations:
[train_dat, clim_vals] = add_obs(
    place = site, fcst_df = train_datset, obs_dir = obs_dir, day = day)

# bias correct and return 2 deterministic forecast outputs and 
# also the overall bias corrected dataframe:
lo_df, hi_df, train_dat, prob_verif =  post_process(train_dat, win_len, 
                        clim_vals)

# focus on low flow season only:
train_dat  = train_dat[train_dat["Obs"] <= clim_vals["q70_flo"]]
train_dat = train_dat[["Q_dmb", "Obs"]]



# %%

"""
####### Kernel Density Estimation: ######
"""



# %% Bandwidth calculation:
# Question: what is the optimum bandwidth?
# Approach 1: best member appraoch:
#       For each horizon:
#           Calculate the smallest absolute error
#       10 time steps:
#           Calculate the variance = bandwidth

# 1 time step = 1 day
days = [*range(1,11)]
win_len = 2
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

# focus on low flow season only:
df_low  = calib_dat[calib_dat["Obs"] <= clim_vals["q70_flo"]]


# %%
# calculate the absolute error for each timestep
best_members = df_low.groupby("ens_mem").apply(
        lambda x:(x["Q_dmb"] - x["Obs"]).abs()
    ).droplevel(0).groupby(
        "date"
    ).agg('min')

# calculate variance:
# bandwidth = best_members.var()
bandwidth = 4.35

# rand_date = np.random.choice(best_members.index.values)
rand_date = "20141102"

rand_sample = df_low.reorder_levels(["date", "ens_mem"]). \
    sort_index().loc[rand_date, :]

kde = KernelDensity(kernel="gaussian",
            bandwidth=bandwidth).fit(
                np.reshape(rand_sample["Q_dmb"].values, (-1,1))
            )


minimum = 0.8 * min(rand_sample["Q_dmb"].values) 
maximum = 1.2 * max(rand_sample["Q_dmb"].values)

f           = np.linspace(minimum, maximum, 1000)
f_ensmem    = 162.71

bw_term  = 1 / ( np.sqrt(2) * bandwidth )
gaus_fun = ( bw_term / np.sqrt(np.pi) ) * np.exp(
        - ( bw_term * (f - f_ensmem) )**2
    )

import plotly.express as px
px.line(x=f, y=gaus_fun,
     title='Single kernel')
# %%

import matplotlib.pyplot as plt


X_plot = np.linspace(minimum, maximum, 100)[:,np.newaxis]
y_vals = kde.score_samples(X_plot)

# figure title
fig = go.Figure(
        layout = {
            "xaxis_title" : "River discharge forecast (<i>m<sup>3</sup>/s</i>)",
            "yaxis_title" : "density",
            "title"       : "Guassian kernel dressing" + 
                f"<br> bandwidth = {bandwidth:.2f}, "
                f"forecast horizon = {day}," +  
                f'date = {np.datetime_as_string(rand_date)[:10]}',
            "title_x"     : 0.5,
            "legend_title": "Legend"
        }  
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








