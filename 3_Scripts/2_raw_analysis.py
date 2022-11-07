# %%
from matplotlib.pyplot import title
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

# %% 
"""
####### Other functions used: ######
"""
def scatter_plt (det_df, bc_df, fcst_type = "Q_raw") :
    # Creates forecast vs observation scatter plots. 
    # Individual scatter plot created for a forecast type, 
    # forecast horizon, site and flow conditions
    
    # Setup figure
    fig = go.Figure(
        layout = {
            "xaxis_title"       : "observations (<i>m<sup>3</sup>/s</i>)",
            "yaxis_title"       : "forecasted discharge (<i>m<sup>3</sup>/s</i>)",    
            "yaxis_range"       : [0, 170],
            # "xaxis_range"       : [0, 370],
            "xaxis_rangemode"   : "tozero",
            "font_size"         : 18,
            "title"             : f"forecast horizon = {bc_df.day.unique()[0]}",
            "title_x"           : 0.5,
            "showlegend"        : False,
            "title_yanchor"     : "bottom",
            "title_y"           : 0.92,
            "margin_t"          : 60,
            "margin_r"          : 10, 
            "legend"            : {
                                    "yanchor"   : "top",
                                    "y"         : 0.98,
                                    "xanchor"   : "left",
                                    "x"         : 0.01,
                                    "font_size" : 18
                                }
        }
    )

    ####
    # Plot all forecasts vs observations
    fig.add_trace(
        go.Scattergl(
            x = bc_df['Obs'], y = bc_df[fcst_type], 
            mode = "markers", showlegend = True, 
            name = "all fcst/obs pairs", marker = {"color":"grey"})
    )

    ####
    # add y = x line
    fig.add_trace(
        go.Scattergl(
            x = np.arange(
                    bc_df.Obs.min()*0.95, 
                    min( bc_df.Q_raw.max(),bc_df.Obs.max() ) * 1.05
                ), 
            y = np.arange(
                    min( bc_df.Q_raw.min(),bc_df.Obs.min() ), 
                    min( bc_df.Q_raw.max(),bc_df.Obs.max() )
                ),
                name = "y = x", line = {"color":"black"})
    )


    ####
    # Deterministic forecast plots
    for det_type, det_dat in det_df.groupby(by = "det_frcst"):        

        # colour scheme for the deterministic forecasts
        if det_type == "median":
            colr_val = "cyan"
        elif det_type == "mean" : 
            continue
            # colr_val = "green" 
        else :
            colr_val = "blue"     

        # plot the trace
        fig.add_trace(
            go.Scattergl(x = det_dat["Obs"], y = det_dat[fcst_type],
                name = det_type, mode = 'markers', legendgroup = det_type,
                marker = {"color": colr_val}, showlegend = True
            )                      
        )    

    return fig

def metric_calc_all_data(det_df, bc_df, clim_vals, fcst_types = ["Q_raw"]):
# function to calculate verification metrics for the forecasts
# calculates NSE and KGE

    # loop through the two dataframes to create:
    flo_mean = (clim_vals['lo_flo_clim'] + clim_vals['hi_flo_clim'])/2
        
    data    = []    

    # calculate for the given forecast typ:
    for i in fcst_types:
        # NSE:
        nse = det_df.groupby(by = "det_frcst").apply(
                lambda x:nse_form(x, flo_mean, i)
            )
        # NSE for individual ensemble members as det frcsts:
        nse_all_mem = bc_df.groupby(by = "ens_mem").apply(
                lambda x:nse_form(x, flo_mean, i)
            )

        # KGE:
        kge = det_df.groupby(by = "det_frcst").apply(
                lambda x:kge_form(x, i)
            )
        # KGE for individual ensemble members as det frcsts:
        kge_all_mem = bc_df.groupby(by = "ens_mem").apply(
                lambda x:kge_form(x, i)
            )

        # concatenate and create a dataframe
        verifs = pd.concat(
                    [ pd.concat( [ nse, nse_all_mem ] ), 
                        pd.concat( [ kge.droplevel(1), kge_all_mem.droplevel(1)] )]
                    , axis = 1).set_axis(
                        ["NSE", "r", "flo_var", "bias", "KGE"], axis = 1
                )
        # new index with the fcst_type information:
        verifs["fcst_type"] = i

        data.append(verifs)

    # end for along fcst_types

    verif_data = pd.concat(data)
    verif_data.index.rename("det_frcst", inplace = True)

    return verif_data

# %%
def prob_metric_all_data(df, fcst_type = "Q_raw"):
# calculates the probabilistic performance  CRPS
#    
    # CRPS xskillscore 
    ds           = df.xs(key = 1)[["Obs"]].to_xarray()
    ds['frcsts'] = df.reorder_levels(["date", "ens_mem"]) \
                    .sort_index().to_xarray()[fcst_type]
    ds      = ds.rename_dims({"ens_mem":"member"})
    crps    = ds.xs.crps_ensemble('Obs', 'frcsts').values

    return crps

# compile overall data set for verification analysis
def verif_data_total(days, fcst_data, site, obs_dir, 
    fcst_type = "Q_raw", plt_type = "time"):
    
    det_verif       = []
    complete_data   = []
    prob_verif      = []

    # loop through forecast horizons
    for day in days:
        # add observations:
        [fcst_data_day, clim_vals] = add_obs(
        place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)
        
        # create special deterministic forecasts:
        det_df = det_frcsts (fcst_data_day.reset_index(), fcst_types = [fcst_type])
        
        # add day as index 
        fcst_data_day["day"]   = day
        # append complete_data, the list of bias corrected dataframes.
        complete_data.append(fcst_data_day)

        # calculate deterministic verification metrics:
        verif_data          = metric_calc_all_data(det_df, fcst_data_day, clim_vals)
        verif_data["day"]   = day
        det_verif.append(verif_data)

        # calculate probabilistic verification metrics:
        crps = prob_metric_all_data(df = fcst_data_day)
        prob_verif.append(crps)

        # create the scatter plot
        if plt_type == "scatter":
            fig = scatter_plt (det_df, fcst_data_day, fcst_type = "Q_raw")
            # save as html:
            # save_pth = f'../4_Results/01-Scatter_raw/{site}-day_{day}-raw-scatter.html' 
            # fig.write_html(save_pth)
            # save as jpg:
            save_pth = f'../4_Results/01-Scatter_raw/{site}-day_{day}-raw-scatter.jpg' 
            fig.write_image(save_pth)
            # show on screen:
            fig.show()

        # create the time series:
        elif plt_type == "time":
            fig = time_series_individual(
                    fcst_data_day.reset_index(), site, day, fcst_type
                )
            # save as html:
            save_pth = f'../4_Results/05-Time_series-raw/{site}-day_{day}-raw-time.html' 
            fig.write_html(save_pth)
            # save as jpg:
            # save_pth = f'../4_Results/05-Time_series-raw/{site}-day_{day}-raw-time.jpg' 
            # fig.write_image(save_pth, scale=1, width=1000, height=500 )
            # show on screen:
            fig.show()            

    # combine all 15 days of data
    det_verif   = pd.concat(det_verif)
    det_verif       = det_verif.set_index(["day"], append= True
                        ).reorder_levels(["day", "det_frcst"]).sort_index()
    # create a dataframe of CRPS values across entire horizon
    prob_verif = pd.DataFrame(
        {"day":np.array(days), 
        "crps":np.array(prob_verif),
        "fcst_type" : fcst_type }
    )
    prob_verif = prob_verif.set_index("day", 
            append= True).droplevel(0).sort_index()

    # compile the overall data again
    complete_data   = pd.concat(complete_data)
    complete_data   = complete_data.set_index('init_date', append=True
        ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
            [['Q_raw', 'Obs', 'day']]

    return det_verif, prob_verif, complete_data

# Verification plots for entire dataset
def verif_plot_total(metric):
    ###### Setup Figure Layout ######
    if metric == "crps": 
        # Setup figure for crps
        fig_crps = go.Figure(
                layout = {
                    "xaxis_title"       : "forecast horizon (day)",
                    "yaxis_title"       : "CRPS (<i>m<sup>3</sup>/s</i>)",    
                    "font_size"         : 18,
                    "title"             : f"<b> CRPS score for entire dataset </b>",
                    "title_x"           : 0.5,
                    "title_yanchor"     : "bottom",
                    "title_y"           : 0.95,
                    "margin_t"          : 60,
                    "margin_r"          : 10, 
                    "legend"            : {
                                            "yanchor"   : "top",
                                            "y"         : 0.98,
                                            "xanchor"   : "left",
                                            "x"         : 0.15,
                                            "font_size" : 18
                                        }
                }
            )
        # color scheme for the CRPS traces:
        colors   = iter(pc.qualitative.D3) 

    else:

        # make subplot interface for the deterministic metrics
        fig = make_subplots(cols         = 3,
                            rows         = 2, 
                            shared_xaxes = False,
                            shared_yaxes = False,
                            vertical_spacing    = 0.09,
                            horizontal_spacing  = 0.03, 
                            subplot_titles      = 
                                ["Tumlingtar", "Balephi", "", "Marsyangdi", "Trishuli", "Naugadh"],
                            x_title = "Forecast horizon (day)",
                            y_title = metric + " score"
                            )

        # Add figure and legend title                  
        fig.update_layout(
            title_text  = f"<b> Total {metric} </b>",
            title_x     = 0.50,
            font_size   = 18,
            margin_l    = 100,
            legend      = {
                    'x': 0.80,
                    'y': 1.,
                    'itemwidth':40, 
                },
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
   
    ###### Create the traces ######

    sites   = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli", "Naugadh"]
    # sites   = ["Marsyangdi"]

    row = 1; col = 1
    for site in sites:

        # select date range:
        date_range = ['20140101', '20141231'] if site == "Naugadh" \
            else ['20140101', '20151231']  
        # load data:
        obs_dir, fcst_data = get_fcst_data ( date_range, site)

        # verification data for entire year
        [ det_verif, prob_verif, complete_data ] = verif_data_total(
                                    range(1,16), fcst_data, site, obs_dir, plt_type = "")

        legend_decide = True if site == "Marsyangdi" else False

        # plot CRPS aka the probabilistic metric
        if metric == "crps": 
            fig_crps.add_trace( 
                go.Scatter(
                    x = prob_verif.index,
                    y = prob_verif['crps'], 
                    line = dict(
                        color = next(colors), width=4,
                        shape = 'spline'
                        ),
                    name = site,
                    ),
            )

        else:
            # add the NSE/KGE value of best member:
            fig.add_trace( 
                go.Scatter(
                        y = det_verif.groupby("day").max()[metric],
                        x = det_verif.groupby("day").max().index.values,
                        name = "best value", line_color = "magenta",
                        legendgroup = "best-val", showlegend = legend_decide 
                    ),
                row = row, col = col 
            )

            # add the NSE/KGE value of worst member:
            fig.add_trace( 
                go.Scatter(
                        y = det_verif.groupby("day").min()[metric],
                        x = det_verif.groupby("day").min().index.values,
                        name = "worst value", line_color = "purple",
                        legendgroup = "worst-val", showlegend = legend_decide 
                    ),
                row = row, col = col 
            )

            # add the 3 standard deterministic forecasts:
            for det_type in ["mean", "median", "high-res"]:
                
                # colour scheme for the deterministic forecasts
                if det_type == "median":
                    colr_val = "cyan"
                elif det_type == "mean" : 
                    colr_val = "green" 
                else :
                    colr_val = "blue"    

                # add the NSE/KGE for the deterministic forecasts 
                fig.add_trace( 
                    go.Scatter(
                            y = det_verif.xs( det_type, level = "det_frcst")[metric],
                            x = det_verif.xs( det_type, level = "det_frcst").index.values,
                            name = det_type, line_color = colr_val,
                            legendgroup = det_type, showlegend = legend_decide 
                        ),
                    row = row, col = col 
                )

            # move to the next site
            if site == "Balephi":
                row = 2; col = 1
            else : col += 1

    ######## Show the created plots #########
    if metric == "crps":
        # save information for crps:
        save_pth = f'../4_Results/02-Verification-raw-entire_data/CRPS.html' 
        fig_crps.write_html(save_pth)
        fig_crps.show()
    else :
        # save information for verification metrics
        save_pth = f'../4_Results/02-Verification-raw-entire_data/{metric}.html' 
        fig.write_html(save_pth)
        fig.show()

# compile verification data by season (dry/wet):
def verif_data_seasonal(days, fcst_data, site, obs_dir, win_len = 2, fcst_type = "Q_raw"):
    # initialize with empty lists:
    lo_verif        = []
    hi_verif        = []
    prob_verif      = []
    complete_data   = []
    # loop through the forecast horizons:
    for day in days:

        [fcst_data_day, clim_vals] = add_obs(
            place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

        lo_df, hi_df, bc_df, prob_df, df_det =  post_process(
            fcst_data_day, win_len, clim_vals, fcst_types = [fcst_type])

        lo_df["day"]   = day
        hi_df["day"]   = day
        prob_df["day"] = day
        bc_df["day"]   = day
        
        # one large df with day information
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)
        prob_verif.append(prob_df)
        complete_data.append(bc_df)

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
    det_verif       = det_verif.set_index(["day", "flow_clim", "fcst_type"], 
                        append= True).reorder_levels(["day", "flow_clim", "fcst_type", 
                            "det_frcst"]).droplevel("fcst_type").sort_index()
    prob_verif      = prob_verif.set_index(["day"], append= True).reorder_levels(
                        ["day", "flow_clim", "fcst_type"]).droplevel("fcst_type").sort_index()
    complete_data   = complete_data.set_index('init_date', append=True
        ).reorder_levels(["init_date", "ens_mem", "date"]).sort_index() \
            [['Q_raw', 'Obs', 'day']]

    return det_verif, prob_verif, complete_data

def verif_plot_seasonal(metric, flo_con):
    
    ###### Setup Figure Layout ######
    if metric == "crps": 
        # Setup figure for crps
        fig_crps = go.Figure(
                layout = {
                    "xaxis_title"       : "forecast horizon (day)",
                    "yaxis_title"       : "CRPS (<i>m<sup>3</sup>/s</i>)",    
                    "font_size"         : 18,
                    "title"             : f"<b> CRPS score for {flo_con} flow </b>",
                    "title_x"           : 0.5,
                    "title_yanchor"     : "bottom",
                    "title_y"           : 0.95,
                    "margin_t"          : 60,
                    "margin_r"          : 10, 
                    "legend"            : {
                                            "yanchor"   : "top",
                                            "y"         : 0.98,
                                            "xanchor"   : "left",
                                            "x"         : 0.15,
                                            "font_size" : 18
                                        }
                }
            )
        # color scheme for the CRPS traces:
        colors   = iter(pc.qualitative.D3) 

    else:

        # make subplot interface for the deterministic metrics
        fig = make_subplots(cols         = 3,
                            rows         = 2, 
                            shared_xaxes = False,
                            shared_yaxes = False,
                            vertical_spacing    = 0.09,
                            horizontal_spacing  = 0.03, 
                            subplot_titles      = 
                                ["Tumlingtar", "Balephi", "", "Marsyangdi", "Trishuli", "Naugadh"],
                            x_title = "Forecast horizon (day)",
                            y_title = metric + " score"
                            )

        # Add figure and legend title                  
        fig.update_layout(
            title_text  = f"<b> {metric} for {flo_con} flow</b>",
            title_x     = 0.50,
            font_size   = 18,
            margin_l    = 100,
            legend      = {
                    'x': 0.80,
                    'y': 1.,
                    'itemwidth':40, 
                },
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
        
    ###### Create the traces ######

    # Make year round KGE plots:
    sites   = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli", "Naugadh"]
    # sites   = ["Marsyangdi"]

    row = 1; col = 1
    for site in sites:

        # select date range:
        date_range = ['20140101', '20141231'] if site == "Naugadh" \
            else ['20140101', '20151231']  
        # load data:
        obs_dir, fcst_data = get_fcst_data ( date_range, site)

        # verification data for entire year
        [ det_verif, prob_verif, complete_data ] = verif_data_seasonal(
                                    range(1,16), fcst_data, site, obs_dir)

        legend_decide = True if site == "Marsyangdi" else False

        # plot CRPS aka the probabilistic metric
        if metric == "crps": 
            prob_verif = prob_verif.xs(flo_con, level = "flow_clim")
            fig_crps.add_trace( 
                go.Scatter(
                    x = prob_verif.index,
                    y = prob_verif['crps'], 
                    line = dict(
                        color = next(colors), width=4,
                        shape = 'spline'
                        ),
                    name = site,
                    ),
            )

        # plot the deterministic metrics:
        else:

            det_verif = det_verif.xs(flo_con, level = "flow_clim")
            # add the NSE/KGE value of best member:
            fig.add_trace( 
                go.Scatter(
                        y = det_verif.groupby("day").max()[metric],
                        x = det_verif.groupby("day").max().index.values,
                        name = "best value", line_color = "magenta",
                        legendgroup = "best-val", showlegend = legend_decide 
                    ),
                row = row, col = col 
            )

            # add the NSE/KGE value of worst member:
            fig.add_trace( 
                go.Scatter(
                        y = det_verif.groupby("day").min()[metric],
                        x = det_verif.groupby("day").min().index.values,
                        name = "worst value", line_color = "purple",
                        legendgroup = "worst-val", showlegend = legend_decide 
                    ),
                row = row, col = col 
            )

            # add the 3 standard deterministic forecasts:
            for det_type in ["mean", "median", "high-res"]:
                
                # colour scheme for the deterministic forecasts
                if det_type == "median":
                    colr_val = "cyan"
                elif det_type == "mean" : 
                    colr_val = "green" 
                else :
                    colr_val = "blue"    

                # add the NSE/KGE for the deterministic forecasts 
                fig.add_trace( 
                    go.Scatter(
                            y = det_verif.xs( det_type, level = "det_frcst")[metric],
                            x = det_verif.xs( det_type, level = "det_frcst").index.values,
                            name = det_type, line_color = colr_val,
                            legendgroup = det_type, showlegend = legend_decide 
                        ),
                    row = row, col = col 
                )

            # move to the next site
            if site == "Balephi":
                row = 2; col = 1
            else : col += 1

    ######## Show the created plots #########
    if metric == "crps":
        # save information for crps:
        save_pth = f'../4_Results/03-Verification-raw-entire_data/CRPS_{flo_con}.html' 
        fig_crps.write_html(save_pth)
        fig_crps.show()
    else :
        # save information for verification metrics
        save_pth = f'../4_Results/03-Verification-raw-entire_data/{metric}_{flo_con}.html' 
        fig.write_html(save_pth)
        fig.show()


    return fig 

# %% 
"""
####### Load data for the chosen site: ######
"""
site = "Balephi";  date_range = ['20140101', '20141231']
# fcst_data is the original df used for all later calculations:
obs_dir, fcst_data = get_fcst_data ( date_range, site)

day = 2; win_len = 2

[fcst_data_day, clim_vals] = add_obs(
            place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

# Bias correct the forecasts using DMB and LDMB
bc_df = bc_fcsts(df = fcst_data_day, win_len = win_len )

# Separate dataframes for deterministic forecasts:
# df = t1.reset_index()
df_det = det_frcsts(bc_df.reset_index(), ["Q_raw"])

# calculate the metrics:
lo_verif, hi_verif = metric_calc(df_det, bc_df, clim_vals, ["Q_raw"])

print(hi_verif.sort_values(["KGE", "NSE"],ascending=False).head(3))
print(lo_verif.sort_values(["KGE", "NSE"],ascending=False).head(3))



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

sites   = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli", "Naugadh"]

best_df = []; wrst_df = []
for site in sites:

    # select date range:
    date_range = ['20140101', '20141231'] if site == "Naugadh" \
        else ['20140101', '20151231']  

    # load data:
    obs_dir, fcst_data = get_fcst_data ( date_range, site)

    # verification data for entire year
    [ det_verif, prob_verif, complete_data ] = verif_data_total(
                            range(1,16), fcst_data, site, obs_dir, plt_type = " ")

    # [ det_verif, prob_verif, complete_data ] = verif_data_seasonal (
    #         range(1,16), fcst_data, site, obs_dir)

    # remove the for loop and the subset line for entire year
    # for flow_clim in ["high", "low"]:
    #     # subset only the low/high flow values:
    #     df = det_verif.xs(flow_clim, level = "flow_clim")
    
    # for entire dataset:
    df=det_verif    
    best_df, wrst_df = best_wrst_count(df, best_df, wrst_df, "")

best_df = pd.concat(best_df)
best_df = best_df.set_index("site", append=True).reorder_levels(["site", "day"]).sort_index()
wrst_df = pd.concat(wrst_df)
wrst_df = wrst_df.set_index("site", append=True).reorder_levels(["site", "day"]).sort_index()

# %%
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

# %%

# period : low, high, entire
for period in ["low", "high"]:
    # Subplots:
    fig = best_wrst_barplot(period, best_df[best_df["flow_clim"] == period], 
            wrst_df[wrst_df["flow_clim"] == period])

# %%
period = "entire"
fig = best_wrst_barplot(period, best_df, wrst_df)

#%%
"""
####### Verification results by climatological cutoff ######
"""
# NSE, KGE, or crps
metric = "NSE"; flo_con = "low"
print(metric, flo_con)
fig = verif_plot_seasonal(metric, flo_con)

# %% Plot for total annual verification:
"""
####### Verification results of entire dataset ######
"""
# CRPS / KGE /NSE
metric      = "crps"
verif_plot_total(metric)

# %%
"""
####### Runoff analysis ######
"""
date_range  = ['20140401', '20140715']
site        = 'Marsyangdi'
try:
    runoff_data = pd.read_pickle("./pickle_dfs/" + site + "_runoff.pkl")
except:
    runoff_data = runoff_data_creator(site, date_range)

# %% plot the time series
day = 2
plt_data = runoff_data.xs(day, level = 'day_no').reset_index()
if day == 11:
    plt_data = plt_data.drop(52)
# plt_data = runoff_data.xs(3, level = 'day_no').loc(axis=0
#     )[(slice(None), slice('20150103', '20150105'))].reset_index()
fig = time_series_individual(plt_data, site, day, fcst_type = 'runoff')
fig.update_layout(
    title_text = f"<b> Day {day} runoff forecast time series"+
                                f"<br> site = {site} </b>" 
)
fig.show()

save_pth = f'../4_Results/07-runoff-time_series/{site}-day_{day}-runoff.html' 
fig.write_html(save_pth)

# %%
"""
####### Verification results of entire dataset ######
"""
# load the upstream and the downstream datasets:
obs_up = pd.read_csv( os.path.join(obs_dir, "DFL_439.35.txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])
# dwnstream station:
obs_dwn = pd.read_csv( os.path.join(obs_dir, "Marsyangdi.txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])


x_axis = slice('20140101', '20141231')
# make plotly plot:
fig = go.Figure(
    data = [
        # go.Scatter(x = obs_up[x_axis].index, y = obs_up[slice('20110101', '20111231')].Obs, name = "upstream 2011"),
        # go.Scatter(x = obs_up[x_axis].index, y = obs_up[slice('20120101', '20121231')].Obs, name = "upstream 2012",
        #             line = dict(color = "blue", width=2, shape = 'spline', dash = "solid")),
        # go.Scatter(x = obs_up[x_axis].index, y = obs_up[slice('20130101', '20131231')].Obs, name = "upstream 2013"),
        # go.Scatter(x = obs_up[x_axis].index, y = obs_up[x_axis].Obs, name = "upstream 2014"),
        go.Scatter(x = obs_up[x_axis].index, y = obs_up[slice('20150101', '20151231')].Obs, name = "upstream 2015",
                    line = dict(color = "blue", width=2, shape = 'spline', dash = "solid")),

        # dwnstream locations
        # go.Scatter(x = obs_dwn[x_axis].index, y = obs_dwn[slice('20120101', '20121231')].Obs, name = "downstream 2012",
        #             line = dict(color = "red", width=2, shape = 'spline', dash = "solid")),
        # go.Scatter(x = obs_dwn[x_axis].index, y = obs_dwn[slice('20130101', '20131231')].Obs, name = "downstream 2013"),
        # go.Scatter(x = obs_dwn[x_axis].index, y = obs_dwn[x_axis].Obs, name = "downstream 2014"),
        go.Scatter(x = obs_dwn[x_axis].index, y = obs_dwn[slice('20150101', '20151231')].Obs, name = "downstream 2015",                     
                    line = dict(color = "red", width=2, shape = 'spline', dash = "solid")),
    ],
    layout = {
        "yaxis_title"       : "Observed Discharge (<i>m<sup>3</sup>/s</i>)",    
        "xaxis_tickformat"  : "%b",
        "font_size"         : 18,
        "title"             : f"<b>Downstream vs Upstream observation comparison",
        "title_x"           : 0.5,
        "title_yanchor"     : "bottom",
        # "showlegend"        : True,
        "title_y"           : 0.92,
        "margin_t"          : 60,
        "margin_r"          : 10, 
        "legend"            : {
                                "yanchor"   : "top",
                                "y"         : 0.98,
                                "xanchor"   : "left",
                                "x"         : 0.01,
                                "font_size" : 18,
                            }
    }
)
# fig.update_traces(marker=dict(size=12,
#                               line=dict(width=2,
#                                         color='DarkSlateGrey')),
#                   selector=dict(mode='markers'))
fig.show("iframe")

# %% Individual NSE plots
fig = go.Figure(
        layout = {
            "xaxis_title"       : "forecast horizon (day)",
            "yaxis_title"       : "NSE",    
            "font_size"         : 35,
            "title_yanchor"     : "bottom",
            "title_y"           : 0.95,
            "margin_t"          : 10,
            "margin_r"          : 10, 
            "legend"            : {
                                    "yanchor"   : "top",
                                    "y"         : 0.4,
                                    "xanchor"   : "left",
                                    "x"         : 0.28,
                                    "font_size" : 30,
                                    "traceorder": "normal"
                                }
        }
    )

# color scheme for the CRPS traces:
colors   = iter(pc.qualitative.D3)

fig.add_hline(y = 0, )
fig.add_hline(y = 0.7, line_dash = "dash")

sites   = ["Tumlingtar", "Balephi", "Marsyangdi", "Trishuli", "Naugadh"]
for site in sites:

    print (site)
    col_val = next(colors)
    print (col_val)

    date_range = ['20140101', '20141231'] if site == "Naugadh" \
        else ['20140101', '20151231']  

    # load data:
    obs_dir, fcst_data = get_fcst_data ( date_range, site)

    # verification data for entire year
    [ det_verif, prob_verif, complete_data ] = verif_data_total(
                            range(1,16), fcst_data, site, obs_dir, plt_type = " ")


    # add the NSE value of best member:
    fig.add_trace( 
        go.Scatter(
                y = det_verif.groupby("day").max()["NSE"],
                x = det_verif.groupby("day").max().index.values,
                name = "best value", line_color = col_val,
                legendgroup = site, showlegend = False
            ),
    )

    # worst member
    fig.add_trace( 
        go.Scatter(
                y = det_verif.groupby("day").min()["NSE"],
                x = det_verif.groupby("day").min().index.values,
                name = site, line_color = col_val, 
                fill = "tonexty", legendgroup = site, 
                showlegend = True
            ),
    )


save_pth = f'../4_Results/02-Verification-raw-entire_data/NSE-all.html' 
fig.write_html(save_pth)
fig.show()




