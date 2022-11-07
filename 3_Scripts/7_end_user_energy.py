# %%
import pandas as pd
import warnings

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

# inflow forecasts received by end-user:
def end_usr_inflow (obs_data, complete_data, plot = True):

    # list of low flow dates:
    q60_flo     = obs_data.quantile(q = 0.7, axis =0, 
                numeric_only = True, interpolation = "linear")[0]
    poss_dates  = obs_data[obs_data.Obs < q60_flo].loc[slice("20150101", "20151220")].index
    rand_sample = random.choice(poss_dates.strftime('%Y%m%d'))

    rand_sample = "20151010"
    # extract random date:
    rndm_sam_data            = complete_data.xs(rand_sample, level = 'init_date').sort_index()
    rndm_sam_data['month']   = rndm_sam_data.index.get_level_values(1).month
    rndm_sam_data            = rndm_sam_data.set_index("month", append=True)
    # join the monthly climatological flow data:
    rndm_sam_data            = rndm_sam_data.join(obs_mon, how = "inner", rsuffix= "_mean"
                        ).droplevel("month").drop(["day"], axis  = 1)
    # calculate e-flow:                   
    rndm_sam_data["eflow"]   = rndm_sam_data["Obs_mean"] * 0.1
    rndm_sam_data

    # calculate estimated energy yield (in MWh)
    # in W : net_flow * gravity + net_head * plant_eff * 
    #           hrs_operation * specfic gravity 
    rndm_sam_data['fcst_energy'] = ( rndm_sam_data['Q_dmb'] - rndm_sam_data['eflow'] ) * 9.81 * \
        net_head * efficiency * (24 - outage) / 1000
    rndm_sam_data.loc[rndm_sam_data.fcst_energy > max_energy, 'fcst_energy'] = max_energy

    # perfect information energy yield:
    rndm_sam_data['perf_info'] = ( rndm_sam_data['Obs'] - rndm_sam_data['eflow'] ) * 9.81 * \
        net_head * efficiency * (24 - outage) / 1000
    rndm_sam_data.loc[rndm_sam_data.perf_info > max_energy, 'perf_info'] = max_energy


    if plot == True:
        # Plot the end user received inflow forecast:
        fig = go.Figure(
                layout = {
                    "yaxis"         : {
                            "title"          : "Discharge (<i>m<sup>3</sup>/s</i>)",      
                            "title_standoff" : 0
                            },
                    "title"         : "<b> 15 day streamflow forecast <br> </b>",
                    "title_x"       : 0.5,
                    "title_y"       : 0.93,
                    "font_size"     : 18,
                    "margin"        : {
                                        "r" : 10,
                                        "t" : 50,
                                        "b" : 10
                                    },
                    "legend"        : {
                                        "yanchor"   : "top",
                                        "y"         : 0.27,
                                        "xanchor"   : "left",
                                        "x"         : 0.,
                                        "font_size" : 14
                                    }
                }   
            )

        # plot the ensemble tracers
        for i in range(1,53):
            fig.add_trace(
                go.Scatter(
                    x = rndm_sam_data.xs(i, level = 'ens_mem').index, 
                    y = rndm_sam_data.xs(i, level = 'ens_mem')["Q_dmb"],
                    name = "ensemble forecasts", legendgroup = "forecast",
                    showlegend = True if i == 1 else False,
                    marker_opacity = 0,
                    line = dict(
                        width = 2, shape = 'spline', color = 'blue'
                    )
                )
            )

        # add ensemble median:
        fig.add_trace( 
            go.Scatter(x = rndm_sam_data.groupby(by = "date").median().index,
                    y = rndm_sam_data.groupby(by = "date").median()['Q_dmb'],
                    name = "ensemble median", legendgroup = "ens-med",
                    line = {"color" : "cyan", "shape" : "spline"}
            )
        )

        # add raw forecast flow:
        # fig.add_trace( 
        #     go.Scatter(x = rndm_sam_data.groupby(by = "date").median().index,
        #             y = rndm_sam_data.groupby(by = "date").median()['Q_raw'],
        #             name = "Raw ensemble median", showlegend = False, 
        #             line = {"color" : "green", "shape" : "spline"}
        #     )
        # )

        # add actual flow:
        fig.add_trace( 
            go.Scatter(x = rndm_sam_data.xs(key = 1, level = "ens_mem").index,
                    y = rndm_sam_data.xs(key = 1, level = "ens_mem")['Obs'],
                    name = "actual flow", legendgroup = "obs",
                    line = {"color" : "red", "shape" : "spline"}
            )
        )

        return fig

    return rndm_sam_data

def bc_frcst_data (days, fcst_data, obs_dir, site):
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
    complete_data   = complete_data.set_index('init_date', append=True
        ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
            [['Q_raw', 'Q_dmb', 'Obs', 'day']]

    # get monthly climatology data
    obs_data =  pd.read_csv( os.path.join(obs_dir, site+".txt"), 
                names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
                infer_datetime_format=True, index_col = [0])
    obs_mon = obs_data.resample('M').mean()
    obs_mon["month"] = obs_mon.index.month
    obs_mon = obs_mon.groupby("month").min()

    return complete_data, obs_mon, clim_vals

def single_horizon (complete_data, clim_vals, day):

    day_ahead  = complete_data[complete_data["Obs"] <= clim_vals["q60_flo"]]. \
                    set_index("day", append=True).xs(day, level = "day").  \
                        droplevel("init_date").reorder_levels(["date", "ens_mem"]). \
                            sort_index()
    
    # calculate the persistence forecasts    
    obs         = day_ahead.groupby("date").mean()[["Obs"]]
    # (yesterday's observation = forecast for tomorrow)
    pers_frcst  = obs[["Obs"]].shift(periods=2)
    pers_frcst  = pers_frcst.rename(columns={'Obs': 'pers_frcst'})

    day_ahead = day_ahead.join(pers_frcst, how="inner").dropna()

    return day_ahead

# energy yield from the ensemble median:
def med_enrgy_yield (day_ahead, obs_mon, site):

    ## related data:
    # Marsyangdi
    if site == "Marsyangdi":
        rated_discharge = 30.5 
        turbines        = 3
        outage          = 0
        efficiency      = 0.87
        net_head        = 90.5
    elif site == "Trishuli":
        # Trishuli
        rated_discharge = 7.8
        turbines        = 7
        outage          = 0
        efficiency      = 0.87
        net_head        = 51.4
    else :
        rated_discharge = 23.5
        turbines        = 3
        outage          = 0
        efficiency      = 0.92
        net_head        = 201.7
    
    max_energy      = rated_discharge * turbines * 9.81 * net_head * efficiency * 24/1000 

    ## deterministic flow information:
    det_data   = day_ahead.groupby("date").median()

    det_data['month']   = det_data.index.get_level_values('date').month
    det_data            = det_data.set_index("month", append=True)
    # join the monthly climatological flow data:
    det_data            = det_data.join(obs_mon, how = "inner", rsuffix= "_mean"
                        ).droplevel("month")

    
    # add the minimum and maximum flow information:
    det_data["Q_max"]   = day_ahead[["Q_dmb"]].groupby("date").max()
    det_data["Q_min"]   = day_ahead[["Q_dmb"]].groupby("date").min()
    # calculate e-flow:                   
    det_data["eflow"]   = det_data["Obs_mean"] * 0.1
    
    det_energy = det_data[["pers_frcst", 'Q_min', 'Q_dmb', 'Q_max', "Obs"]].sub(det_data["eflow"], axis = 0) \
                * 9.81 * net_head * efficiency * (24 - outage) / 1000
    det_energy[det_energy > max_energy] = max_energy

    # energy calculation for entire ensemble
    ens_energy = day_ahead.copy(deep = True)
    ens_energy['month']   = ens_energy.index.get_level_values("date").month
    ens_energy            = ens_energy.set_index("month", append=True)
    # join the monthly climatological flow data:    
    ens_energy            = ens_energy.join(obs_mon, how = "inner", rsuffix= "_mean"
                            ).droplevel("month")
    ens_energy["eflow"]   = ens_energy["Obs_mean"] * 0.1
    ens_energy = ens_energy[["pers_frcst", 'Q_dmb', "Obs"]].sub(ens_energy["eflow"], axis = 0) \
                * 9.81 * net_head * efficiency * (24 - outage) / 1000
    ens_energy[ens_energy > max_energy] = max_energy    
    ens_energy

    return det_data, det_energy, ens_energy

# revenue calculation function:
def revenue_calc(df, fcst_type, ppa_rate):

    generation = df["Obs"]
    bid_amt    = df[fcst_type] 
    # normal conditions
    gen_revenue = generation * ppa_rate
    
    # add fine amount:
    if generation < 0.8*bid_amt:
        fine_amt = (generation - 0.8*bid_amt) * ppa_rate
        gen_revenue = gen_revenue + fine_amt
    
    # excess generation 
    elif generation > bid_amt :
        gen_revenue = ( bid_amt + (generation - bid_amt)/2 ) * ppa_rate  
    
    return gen_revenue

# cases of fine, normal, excessive production days:
def revenue_df(ens_energy, det_energy):

    # ppa rate NRs per kWh | convert to MWh
    ppa_rate = 8.30 * 1000
    fine_half = det_energy.copy(deep = True)

    fine_half["pers_fine"] = np.where(det_energy["Obs"] < 0.8 * det_energy["pers_frcst"], True, False)
    fine_half["fcst_fine"] = np.where(det_energy["Obs"] < 0.8 * det_energy["Q_dmb"], True, False)

    fine_half["pers_half"] = np.where(det_energy["Obs"] > det_energy["pers_frcst"], True, False)
    fine_half["fcst_half"] = np.where(det_energy["Obs"] > det_energy["Q_dmb"], True, False)

    revenue = det_energy.copy(deep = True)
    revenue["pers_revenue"] = revenue.apply( lambda df:revenue_calc(df, "pers_frcst", ppa_rate), axis = 1 )
    revenue["fcst_revenue"] = revenue.apply( lambda df:revenue_calc(df, "Q_dmb", ppa_rate), axis = 1 )
    revenue["act_revenue"]  = revenue["Obs"] * ppa_rate

    ens_energy["pers_revenue"] = ens_energy.apply( lambda df:revenue_calc(df, "pers_frcst", ppa_rate), axis = 1 )
    ens_energy["fcst_revenue"] = ens_energy.apply( lambda df:revenue_calc(df, "Q_dmb", ppa_rate), axis = 1 )
    ens_energy["act_revenue"]  = ens_energy["Obs"] * ppa_rate
    ens_energy

    quarts = ens_energy.groupby("date")[["Q_dmb", "Obs"]].quantile([0.25, 0.75])
    quarts["fcst_revenue"] = quarts.apply( lambda df:revenue_calc(df, "Q_dmb", ppa_rate), axis = 1 )
    quarts.index.rename(['date','Q'], inplace = True)

    return revenue, fine_half, ens_energy, quarts


# %% ######################################### %% #
############# Get Forecast Data ################
site = "Tumlingtar";  date_range = ['20150101', '20151231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:

"""
####### Compile overall bias corrected dataset ######
"""
days                    = range(1,16)
complete_data, obs_mon, clim_vals  = bc_frcst_data (days, fcst_data, obs_dir, site)
# complete_data

# %%
"""
####### Financial analysis for day ahead scenario ######
"""
days = [1]
for day in days:
    print(day)
    # ensemble forecasts for a single horizon:
    day_ahead = single_horizon (complete_data, clim_vals, day = day)
    day_ahead

    # get the ensemble median forecasts and respective energy yield:
    det_data, det_energy, ens_energy = med_enrgy_yield (day_ahead, obs_mon, site)

    # calculate the revenue:
    revenue, fine_half, ens_energy, quarts = revenue_df(ens_energy, det_energy)

    # calculate the fines and fine_halfs: 
    fine_half[["pers_fine", "fcst_fine", "pers_half", "fcst_half"]].sum()
    # revenue

    #
    max_rev = ens_energy[["fcst_revenue"]].groupby("date").max().sum()
    min_rev = ens_energy.groupby("date").apply(
                lambda x:x[ x["fcst_revenue"] > 0 ][["fcst_revenue"]].min()
            ).sum()

    # calculate overall revenue
    print(revenue[["pers_revenue", "fcst_revenue", "act_revenue"]].sum())
    print("\n min and max possible revenue")
    print(min_rev.values, max_rev.values)

    print(quarts[["fcst_revenue"]].groupby("Q").sum())

#%%
"""
####### Create Rank histogram ######
"""

# test = df_low.loc(axis=0)[
#         (slice(1,4), slice("20140410", "20140419"))
#     ].reorder_levels(["date", "ens_mem"]).sort_index()
# test = df_low
# compute individual ranks:     
test = complete_data
test = day_ahead
test = test[test["day"] == 1].groupby(by = "date").apply(
    lambda x: x.assign(
        rank = sum( x["Q_dmb"] < x["Obs"].unique()[0] ) + 1
    )
)  

# plot a histogram:
import plotly.express as px
fig = px.histogram(test, x = "rank", histnorm="percent",
    labels = {'percent' : 'count'},
    title = "<b> Rank histogram")
fig.update_layout(
    margin = {  
                't': 70,
                'b': 20,
                'r': 10
            },
    font_size = 18, 
    yaxis_title = {
                'text'      : 'frequency',
                'standoff'  : 5
            },
    xaxis_title_standoff = 0,
    title = {
                'x': 0.5,
                'y': 0.95
    }           
)
fig.show()
save_pth = f'../4_Results/{site}_rank.jpg' 
fig.write_image(save_pth, scale=1, width=900, height=550 )



# %%
"""
####### Ensemble scenario ######
"""
# first test the cases where the ensemble forecasts did not capture the 
# observation at all:
day_ahead

# %%
# extract the min, max, Q1 and Q3 for the day ahead forecasts
test = day_ahead["Q_dmb"].groupby("date").agg( 
            [ min, lambda x:x.quantile(0.25), lambda x:x.quantile(0.75), max ]
        ).rename(columns = {"<lambda_0>":"Q1", "<lambda_1>":"Q3"})
# add the observations and persistence forecast information:
test = test.join(pd.concat([obs, pers_frcst], axis = 1, join = "inner")).dropna()

test["within_limits"] = np.where( np.logical_and (test["min"] <= test["Obs"] , test["max"] > test["Obs"]), True, False)

test["within_50"] = np.where( np.logical_and (test["Q1"] <= test["Obs"] , test["Q3"] > test["Obs"]), True, False)
test


# %%
# %% bias correction getting perfect information:
vat = bc_df.loc[slice(1,2), slice("20150103" , "20150109"),: ]

vat = bc_df.loc[slice(1,2), slice("20150103" , "20150110"),: ]
vat
