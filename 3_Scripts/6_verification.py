#%%
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

def verif_data_bc(days, fcst_data, site, obs_dir):
    # creates post-processed dataset for verification based on the input results
    lo_verif        = []
    hi_verif        = []
    prob_verif      = []
    complete_data   = []

    for day in days:
        # add observations:
        [fcst_data_day, clim_vals] = add_obs(
        place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

        win_len = 2
        approach = "" 
        # Approach list:
        if site == "Tumlingtar" and day == 11:
            approach = "common_DMB"
            win_len  = 3          
        elif site == "Balephi" and day == 6 : approach = "common_DMB"
        elif site == "Marsyangdi" : approach = "common_DMB"

        # bias correct and return the deterministic forecast outputs and 
        # also the overall bias corrected dataframe:
        lo_df, hi_df, bc_df, prob_df, df_det =  post_process(
            fcst_data_day, win_len, clim_vals, approach)

        # rename column name for approach 2:
        if np.isin("Q_dmb_2", bc_df.columns.values) == True:
            bc_df = bc_df.rename(columns={"Q_dmb_2": "Q_dmb"})

        lo_df["day"]   = day
        hi_df["day"]   = day
        prob_df["day"] = day
        bc_df["day"]   = day
        
        # one large df with day information
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)
        prob_verif.append(prob_df)

        complete_data.append(bc_df)
        
        # plot the time_series for different window length:
        # fig         = time_series_plotter(bc_df.reset_index(), site, day)
        # save_pth    = f'../4_Results/09-verif-time_series/{site}-day_{day}-time_series.html' 
        # # fig.show()
        # fig.write_html(save_pth)

        # # Plot the DMB:
        # fig = dmb_vars_plttr (bc_df, ['dmb'], site, day)
        # save_pth = f'../4_Results/10-DMB_time_series/{site}-day_{day}-DMB_series.jpg' 
        # fig.show()
        # fig.write_image(save_pth)
        # fig.write_html(save_pth)

    # create a single large dataframe for low and high flow seasons:
    lo_verif        = pd.concat(lo_verif)
    hi_verif        = pd.concat(hi_verif)
    prob_verif      = pd.concat(prob_verif)
    complete_data   = pd.concat(complete_data)

    # create single deterministic verification dataframe from lo_verif and hi_verif:
    lo_verif["flow_clim"]   = "low"
    hi_verif["flow_clim"]   = "high"
    det_verif   = pd.concat([lo_verif, hi_verif])

    # set indices and reorder levels:
    det_verif       = det_verif.set_index(["day", "flow_clim", "fcst_type"], append= True
                        ).reorder_levels(["day", "flow_clim", "fcst_type", "det_frcst"]).sort_index()
    prob_verif      = prob_verif.set_index(["day"], append= True
                        ).reorder_levels(["day", "flow_clim", "fcst_type"]).sort_index()

    det_verif       = det_verif.rename(index = {'Q_dmb_2':"Q_dmb"}, level= "fcst_type")
    prob_verif      = prob_verif.rename(index = {'Q_dmb_2':"Q_dmb"}, level= "fcst_type")

    complete_data   = complete_data.set_index('init_date', append=True
        ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
            [['Q_raw', 'Q_dmb', 'Obs', 'day']]

    return det_verif, prob_verif, complete_data

"""
####### Best and worst members by flow season ######
"""

def best_wrst_barplot(period, best_df, wrst_df):
    # make subplot interface for the deterministic metrics
    fig = make_subplots(cols         = 2,
                        rows         = 2, 
                        shared_xaxes = False,
                        shared_yaxes = False,
                        vertical_spacing    = 0.09,
                        horizontal_spacing  = 0.03, 
                        subplot_titles      = 
                            ["best NSE", "worst NSE", "best KGE", "worst KGE" ],
                        y_title = "<b> count </b>"
                        )

    # Add figure and legend title                  
    fig.update_layout(
        title_text  = f"<b> Which member is most frequently the best/worst? ", 
                            # + f"<br> {period} flow times </b>",
        title_x     = 0.50,
        title_y     = 0.96,
        font_size   = 18,
        margin_l    = 60,
        margin_r    = 10,
        margin_t    = 120,
        margin_b    = 10, 
        showlegend  = False,
        legend      = {
                'x': 0.80,
                'y': 1.,
                'itemwidth':40, 
            },
        )

    # update y axes:
    fig.update_yaxes(
        rangemode   = "tozero",
        automargin  = False,
        title_standoff    = 10
        )

    # adjust location of the y-axis label
    fig['layout']['annotations'][-1]['x'] = 0.01

    # increase size of subplot titles:
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=20)

    for metric in ["NSE", "KGE"]:

        row = 1 if metric == "NSE" else 2
        
        # plot the bars for the best values:
        count_df = best_df[metric].value_counts().rename_axis('ens_mem').to_frame('count')
        count_df = count_df[count_df["count"] > 1 ]
        fig.add_trace(
            go.Bar(
                x = count_df.rename(index = {52: "high-res"}).index.values.astype(np.str),
                y = count_df["count"]
            ),
            row = row, col = 1
        )

        # plot the bars for worst values:
        count_df = wrst_df[metric].value_counts().rename_axis('ens_mem').to_frame('count')
        count_df = count_df[count_df["count"] > 1 ]
        fig.add_trace(
            go.Bar(
                x = count_df.rename(index = {52: "high-res"}).index.values.astype(np.str),
                y = count_df["count"]
            ),
            row = row, col = 2
        )

    fig.show()
    save_pth = f'../4_Results/04-Verification-best_mem-table/best-wrst_freq-{period}.jpg'
    fig.write_image(save_pth, scale=1, width=1000, height=1150 )

    return fig 


# %% ######################################### %% #
############# Load Forecast Data ################
site = "Marsyangdi";  date_range = ['20150101', '20151231']
obs_dir, fcst_data = get_fcst_data ( date_range, site)
# fcst_data is the original df used for all later calculations:

# for multiple horizons add the required day no. as an array element:
days  = range(1,16)
# calculate verification metrics
det_verif, prob_verif, complete_data = verif_data_bc(days, fcst_data, site, obs_dir)

# %%
"""
####### Plot the verification results ######
"""
# plot the deterministic metrics vs time horizon:
for det_frcst in ['high-res', 'mean', 'median'] :
    fig = det_skill_horizon_plttr (det_verif, site, det_frcst)
    save_pth = f'../4_Results/13-Verification-metrics/{site}-{det_frcst}.html' 
    fig.write_html(save_pth)
    fig.show()

# %%
# plot crps vs time horizon:
fig  = crps_horizon_plttr(prob_verif, site)
fig.show()
save_pth = f'../4_Results/12-Verification-CRPS/{site}.jpg' 
fig.write_image(save_pth, scale=1, width=900, height=550 )

# %%
"""
####### Identify best and worst members for each metric ######
"""
def best_wrst_count(det_verif, best_df, wrst_df, flow_clim):
    # best member for each horizon for the site
    best_mems   = det_verif.reset_index("day").groupby(by = "day"). \
                    agg(lambda x: x.idxmax())[["NSE", "KGE"]]
    # worst member for each horizon for the site
    wrst_mems   = det_verif.reset_index("day").groupby(by = "day"). \
                    agg(lambda x: x.idxmin())[["NSE", "KGE"]]

    best_mems["site"] = site                    
    wrst_mems["site"] = site   

    if flow_clim == "high" or flow_clim == "low":
        best_mems["flow_clim"] = flow_clim                    
        wrst_mems["flow_clim"] = flow_clim                    

    best_df.append(best_mems)                  
    wrst_df.append(wrst_mems)

    return best_df, wrst_df

sites   = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli"]

best_df = []; wrst_df = []
for site in sites:

    # select date range:
    date_range = ['20150101', '20151231']

    # load data:
    obs_dir, fcst_data = get_fcst_data ( date_range, site)

    det_verif, prob_verif, complete_data = verif_data_bc(
        range(1,16), fcst_data, site, obs_dir)

    # remove the for loop and the subset line for entire year
    for flow_clim in ["high", "low"]:
        # subset only the low/high flow values:
        df = det_verif.xs("Q_dmb", level = "fcst_type").xs(flow_clim, level = "flow_clim")
        best_df, wrst_df = best_wrst_count(df, best_df, wrst_df, flow_clim)

    # # for entire dataset:
    # df=det_verif.reset_index(level = "flow_clim").xs("Q_dmb", level = "fcst_type")   
    # best_df, wrst_df = best_wrst_count(df, best_df, wrst_df, "")

best_df = pd.concat(best_df)
best_df = best_df.set_index("site", append=True).reorder_levels(["site", "day"]).sort_index()
wrst_df = pd.concat(wrst_df)
wrst_df = wrst_df.set_index("site", append=True).reorder_levels(["site", "day"]).sort_index()

# %%
period = "high"
fig = best_wrst_barplot(period, best_df, wrst_df)
save_pth = f'../4_Results/11-Veri-best_wrst-2015/entire.jpg' 
fig.show()
fig.write_image(save_pth, scale=1, width=1000, height=1150 )

# %%
for period in ["low", "high"]:
    # Subplots:
    fig = best_wrst_barplot(period, best_df[best_df["flow_clim"] == period], 
            wrst_df[wrst_df["flow_clim"] == period])
    save_pth = f'../4_Results/11-Veri-best_wrst-2015/{period}.jpg' 
    fig.show()
    fig.write_image(save_pth, scale=1, width=1000, height=1150 )

