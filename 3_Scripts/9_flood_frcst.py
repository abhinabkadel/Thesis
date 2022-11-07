# %%
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
# Post-processing:
days  = range(1,11)

complete_data   = []
det_data        = []

for day in days:

    win_len = 2
    approach = "" 
    # Approach list:
    if site == "Tumlingtar" and day == 11:
        approach = "common_DMB"
        win_len  = 3          
    elif site == "Balephi" and day == 6 : approach = "common_DMB"
    elif site == "Marsyangdi" : approach = "common_DMB"

    # add observations:
    [fcst_data_day, clim_vals] = add_obs(
    place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

    clim_vals["q95_flo"]

    # bias correct and return 2 deterministic forecast outputs and 
    # also the overall bias corrected dataframe:
    lo_df, hi_df, bc_df, prob_df, df_det =  post_process(
        fcst_data_day, win_len, clim_vals, approach)

    # rename column name for approach 2:
    if np.isin("Q_dmb_2", bc_df.columns.values) == True:
        bc_df = bc_df.rename(columns={"Q_dmb_2": "Q_dmb"})

    bc_df["day"]    = day
    df_det["day"]   = day
    
    complete_data.append(bc_df)
    det_data.append(df_det)

# create a single large dataframe for low and high flow seasons:
complete_data   = pd.concat(complete_data)
complete_data   = complete_data.set_index(['day'], append=True
    ).reorder_levels(["day", "ens_mem", "date"]).sort_index() \
        [['Q_raw', 'Q_dmb', 'Obs', 'init_date']]

det_data   = pd.concat(det_data)
det_data   = det_data.set_index(['day', 'date'], append=True
    ).reorder_levels(["day", "det_frcst", "date"]).sort_index() \
        [['Q_raw', 'Q_dmb', 'Obs']]


# %%
# take ensemble median as it denotes 50% probability of exceedence
ens_med     = det_data.xs("median", level = "det_frcst")
# actual flood events:
obs_vals    = obs_vals = df_det.loc["median"].set_index("date")[["Obs"]]
obs_floods  = obs_vals[obs_vals["Obs"] > clim_vals["q95_flo"]]

# forecasted flood events 
fcst_flood = ens_med[ens_med["Q_dmb"] > clim_vals["q95_flo"]]

print(site)
# extract events for each day:
# identify when at least 50 % of the forecast member predict floods:
days = range(5,8)
for day in days:
    test        = fcst_flood.xs(day, level = "day")
    # frcst Y, obs Y
    hits        = (test.isin(obs_floods)["Obs"]).sum() 
    # frcst Y, obs N
    false_alrm  = (~test.isin(obs_floods)["Obs"]).sum()
    # frcst N, obs Y
    miss        = (~obs_floods.isin(test)["Obs"]).sum()
    no_event    = 365 - (hits + false_alrm + miss)
    print(f"day = {day}")
    print(hits, miss, false_alrm, no_event)

    crrct_prtion = (hits + no_event) / 365
    dmb_lck = ((hits + miss)/365) * ((hits + false_alrm)/365) + \
                    ((no_event + miss)/365) * ((no_event + false_alrm)/365)
    # Heidke Skill score
    hs = (crrct_prtion - dmb_lck) / (1 - dmb_lck)

    # % of the correct flood forecasts:
    # higher is better
    hit_rate = hits / (hits + miss)
    
    # % of the false alarms:
    # lower is better
    false_alrm_ratio = false_alrm / (hits + false_alrm)

    print (hit_rate, false_alrm_ratio)
# %%
hi_res      = df_det.loc["high-res"]
hi_res[hi_res["Q_dmb"] > clim_vals["q95_flo"]]

#%%

print(bc_df["Obs"][bc_df["Obs"] > clim_vals["q95_flo"]].unique())

# %%
clim_vals["q95_flo"]

# %%
obs = pd.read_csv( os.path.join(obs_dir, site+".txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])

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

#%%
day = 3; win_len = 2

# add observations:
[fcst_data_day, clim_vals] = add_obs(
place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

clim_vals["q95_flo"]

# bias correct and return 2 deterministic forecast outputs and 
# also the overall bias corrected dataframe:
lo_df, hi_df, bc_df, prob_df, df_det =  post_process(
    fcst_data_day, win_len, clim_vals)
