from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.colors as pc

## TIME SERIES PLOTS (Flow vs Time):
# time series for all 3 bias correction options:
def time_series_plotter(df, site, day):
    # make subplot interface
    fig = make_subplots(rows = 2, cols = 1,
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing = 0.09,
                        subplot_titles=("Raw", "bias-corrected (equal weight)", 
                            "bias-corrected (linearly-weighted)"),
                        x_title = "date",
                        y_title = "River discharge (<i>m<sup>3</sup>/s</i>)"    
                        )
    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Bias-correction for streamflow forecasts"+
            f"<br> site = {site}, horizon = {day}",
        title_x = 0.5,
        legend_title = "Legend", 
        yaxis_rangemode = "tozero"
        )
    # loop through the forecast types:
    fcst_types = ["Q_raw", "Q_dmb"]
    for type in fcst_types:
        legend_decide = True if type == "Q_raw" else False

        # plot ENSEMBLE SPREAD    
        fig.append_trace(
            go.Box(x = df["date"], y=df[type], line = {"color":"rosybrown"},
            name = "ensemble spread", legendgroup = "ens", showlegend = legend_decide), 
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot HIGH-RES
        fig.append_trace( 
            go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
                    y = df[df["ens_mem"] == 52][type],
                    name = "high res", line = {"color":"blue"},
                    legendgroup = "high-res", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot ENS-MEDIAN
        fig.append_trace( 
            go.Scatter(x = df.groupby(by = "date").median().index,
                    y = df.groupby(by = "date").median()[type],
                    name = "ensemble median", line = {"color":"cyan"},
                    legendgroup = "ens-med", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )

        # plot ENS-MEAN
        fig.append_trace( 
            go.Scatter(x = df.groupby(by = "date").mean().index,
                    y = df.groupby(by = "date").mean()[type],
                    name = "ensemble mean", line = {"color":"green"},
                    legendgroup = "ens-mean", showlegend = legend_decide),
            row = fcst_types.index(type) + 1, col = 1
        )
        
        # plot OBS:
        fig.append_trace(
                go.Scatter(x = df[df["ens_mem"] == 52]["date"],
                    y=df[df["ens_mem"] == 52]["Obs"], name = "observed",
                    line = {"color":"red"}, mode = "lines+markers", 
                    legendgroup = "obs", showlegend = legend_decide), 
            row = fcst_types.index(type) + 1, col = 1
        )

    return fig

# time series for individual bias correction option:
def time_series_individual(df, site, day, type):

    yaxis_txt = "catchment runoff (<i>m<sup>3</sup></i>)" if type == 'runoff' \
        else "River discharge (<i>m<sup>3</sup>/s</i>)"

    fig = go.Figure(
        layout = {
            "xaxis_title" : "date",
            "yaxis_title" : "River discharge (<i>m<sup>3</sup>/s</i>)"    
        }  
    )

    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Bias-correction for streamflow forecasts"+
            f"<br> site = {site}, forecast horizon = {day}",
        title_x = 0.5,
        legend_title = "Legend", 
        yaxis_rangemode = "tozero"
        )

    # bc_df = bc_df.reset_index()
    # add ENSEMBLE SPREAD    
    fig.add_trace(
        go.Box(x = df["date"], y=df[type], line = {"color":"rosybrown"},
        name = "ensemble spread", legendgroup = "ens")
    )

    # plot HIGH-RES
    fig.add_trace( 
        go.Scatter(x = df[df["ens_mem"] == 52]["date"], 
                y = df[df["ens_mem"] == 52][type],
                name = "high res", line = {"color":"blue"},
                legendgroup = "high-res")
    )

    # plot ENS-MEDIAN
    fig.add_trace( 
        go.Scatter(x = df.groupby(by = "date").median().index,
                y = df.groupby(by = "date").median()[type],
                name = "ensemble median", line = {"color":"cyan"},
                legendgroup = "ens-med")
    )

    # plot ENS-MEAN
    fig.add_trace( 
        go.Scatter(x = df.groupby(by = "date").mean().index,
                y = df.groupby(by = "date").mean()[type],
                name = "ensemble mean", line = {"color":"green"},
                legendgroup = "ens-mean")
    )

    if type != 'runoff':
        # plot OBS:
        fig.add_trace(
                go.Scatter(x = df[df["ens_mem"] == 52]["date"],
                    y=df[df["ens_mem"] == 52]["Obs"], name = "observed",
                    line = {"color":"red"}, mode = "lines+markers", 
                    legendgroup = "obs")
        )

    return fig 

# observations time series:


## CALIBRATION PLOTS (Skill vs Window Length)
def calibrtn_plttr (hi_verif, lo_verif, prob_verif, site, day, 
        flo_con = "high", fcst_types = ["Q_dmb", "Q_ldmb"]):

    if flo_con == "high" : 
        df_big = hi_verif
    elif flo_con == "low" :
        df_big = lo_verif
    else : print ("wrong input")

    # number of columns for the subplot
    cols    = max ( len(fcst_types) // 2 , 1 )
    rows    = len(fcst_types) if cols == 1 \
        else max ( len(fcst_types) - cols, 1 )

    # make subplot interface
    fig = make_subplots(cols         = cols,
                        rows         = rows, 
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing    = 0.09,
                        subplot_titles      = fcst_types,
                        x_title = "window length (days)",
                        y_title = "Score",
                        specs   = [ [{"secondary_y": True} for 
                            c in range(cols)] for r in range(rows)]
                        )
    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Verification scores for different window lengths"+
            f"<br> site = {site}, day = {day}, flow season = {flo_con} ",
        title_x = 0.5,
        legend_title = "Legend", 
        )
    # update both y axes:
    fig.update_yaxes(
        rangemode = "tozero",
        range = [0.5, 1.1]
        )

    # only extract the ensemble median
    df_big = df_big.xs('median', level = 2)

    row  = 1;  col = 1
    # loop through the forecast types:
    for type in fcst_types:
        
        # slice through the dataframe for individual forecast types:
        df = df_big.xs(type, level = 1)
        
        # show only one legend entry per verification metric:
        legend_decide = True if type == "Q_dmb" else False
        
        # define color to be used:
        colors   = iter(pc.qualitative.D3)

        # deterministic metrics:
        metrics = ["NSE", "r", "flo_var", "bias", "KGE"]
        for metric in metrics:     
            # plot different metrics:
            fig.append_trace( 
                go.Scatter(x = df.index, 
                        y = df[metric], 
                        name = metric,
                        legendgroup = metric,
                        marker_color = next(colors),
                        showlegend = legend_decide
                        ),
                row = row, col = col
            )

        # probabilistic metric:
        df = prob_verif.xs(key = flo_con, level = 1). \
            xs(key = type, level = 1)
        
        # plot CRPS
        fig.add_trace( 
            go.Scatter(x            = df.index, 
                    y               = df['crps'], 
                    name            = 'crps',
                    legendgroup     = 'crps',
                    marker_color    = next(colors),
                    showlegend      = legend_decide
                    ),
            row = row, col = col,
            secondary_y = True
        )

        print(row, col)
        print(type)
        # iterate row and cols:
        col = col + 1
        if col == cols + 1:
            col = 1; row = row + 1

    return fig

## POSTPROCESSING SKILL PLOTS (Skill vs Postprocess technique)


## SKILL Evolution (Skill vs Forecast Horizon)
# plot for deterministic forecasts only:
def det_skill_horizon_plttr (det_verif, site):
    # make subplot interface
    fig = make_subplots(cols         = 2,
                        rows         = 2, 
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing    = 0.05,
                        horizontal_spacing    = 0.05,
                        subplot_titles      = [
                            "raw (low flow)",
                            "raw (high flow)",
                            "bias-corrected (low flow)",
                            "bias-corrected (high flow)"
                            ],
                        x_title = "forecast horizon (day)",
                        y_title = "Skill metric value",
                        specs   = [ [{"secondary_y": True} for 
                            c in range(2)] for r in range(2)]
                        )
                        
    fig.update_layout(
        title_text = "<b> deterministic forecast skill across" + 
                "different forecast horizons" + 
                f"<br> site = </b> {site}",
        title_x    =  0.5,
        legend     = {
            'x': 0.95,
            'y': 1,
            'itemwidth':40, 
            # 'sizey':0.5
        },
        margin     = {
            'b' : 50,
            't' : 70
        } 
    )

    row = 1

    # loop through the forecast types:
    fcst_types = ["Q_raw", "Q_dmb"]
    for fcst_type in fcst_types:
        
        col = 1

        # loop through the flow conditions:
        for flow_con in ['low', 'high'] :

            # different line plot types for the different
            # deterministic forecasts
            dash    = iter(['solid', 'longdash', 'dot'])

            # plot the deterministic forecasts:
            for det_frcst in ['high-res', 'mean', 'median'] :

                # define color and dash option to be used:
                colors  = iter(pc.qualitative.D3)            
                dash_opt = next(dash)

                # deterministic metrics:
                metrics = ["NSE", "r", "flo_var", "bias", "KGE"]
                for metric in metrics:

                    # show only 1 legend entry for the det frcst type:
                    legend_decide = True if fcst_type == 'Q_raw' and \
                        flow_con == 'low' and metric == "NSE" else False

                    # add a dummy trace for legend entries:
                    fig.append_trace(
                        go.Scatter(
                            x = [1], y = [1],
                            marker = {
                                'size'      : 10,
                                'opacity'   : 0
                            },
                            line = dict(
                                color = 'black', width=2,
                                dash  = dash_opt
                                ),                        
                            name =  det_frcst, 
                            legendgroup = det_frcst, 
                            hoverinfo = 'skip', 
                            showlegend = legend_decide                           
                        ),row = row, col = col
                    )

                    # plot the deterministic metrics:
                    fig.add_trace(
                        go.Scatter(
                            x = det_verif.index.get_level_values("day")
                                    .unique().values,
                            y = det_verif.xs(fcst_type, level = "fcst_type")
                                    .xs(flow_con, level = "flow_clim")
                                    .xs(det_frcst, level = "det_frcst")[metric],
                            line = dict(
                                color = next(colors), width=4,
                                dash = dash_opt, shape = 'spline'
                                ), 
                            name =  det_frcst, 
                            legendgroup = det_frcst, 
                            showlegend = False
                        ), 
                        row = row, col = col
                    )
            
            # change the column
            col = col + 1
        
        # change the row 
        row = row + 1

    # Plotly does not allow 2 different legends for a plot. Hence, this approach:
    # add color chart for different metrics as an image:
    from base64 import b64encode
    image_filename      = '../5_Images/det_metrics_image.png'
    det_legend_items    = b64encode(open(image_filename, 'rb').read())
    # add deterministic_forecasts lists as image
    fig.add_layout_image(
        dict(
            source  = 'data:image/png;base64,{}'.format(det_legend_items.decode()),
            xref    = "paper", yref = "paper",
            x = 1.06, y = 0.1,
            sizex=0.5, sizey=0.5,
            xanchor="right", yanchor="bottom"
        )
    )

    return fig

# plot for probabilistic metric (CRPS):
def crps_horizon_plttr(prob_verif, site):
    
    fig = go.Figure(
        layout = {
            "xaxis_title"   : "forecast horizon (day)",
            "yaxis_title"   : "CRPS",    
            "title"         : "<b> CRPS across different forecast horizons </b> <br> " + 
                            "site = " + site,
            "title_x"       : 0.5        
        }
    )

    fcst_types = ["Q_raw", "Q_dmb"]
    for fcst_type in fcst_types:
        dash = 'dashdot' if fcst_type == 'Q_raw' else 'solid'

        colors  = iter(['green', 'red'])   
        # loop through the flow conditions:
        for flow_con in ['low', 'high'] :

            # plot CRPS aka the probabilistic metric
            fig.add_trace( 
                go.Scatter(
                    x = prob_verif.index.get_level_values("day")
                                    .unique().values, 
                    y = prob_verif.xs(fcst_type, level = "fcst_type")
                                .xs(flow_con, level = "flow_clim")['crps'], 
                    line = dict(
                        color = next(colors), width=4,
                        dash = dash, shape = 'spline'
                        ),
                    name            = fcst_type + ' '+ flow_con,
                    )
            )

    return fig

def kge_crps_plttr(det_verif, prob_verif, site, fcst_types):
    
    # make subplot for KGE
    fig = make_subplots(cols         = 2,
                        rows         = 1, 
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing    = 0.05,
                        horizontal_spacing    = 0.05,
                        subplot_titles      = [
                            "low flow",
                            "high flow"
                            ],
                        x_title = "forecast horizon (day)",
                        y_title = "KGE"
                        )
                        
    # update layout for KGE                    
    fig.update_layout(
        title_text = "<b> KGE for different bias correction " +
            "approaches across the 10 day forecast horizon" + 
                f"<br> site = </b> {site}",
        title_x    =  0.5,
        legend     = {
            'x': 0.95,
            'y': 1,
            'itemwidth':40, 
            # 'sizey':0.5
        },
        margin     = {
            'b' : 50,
            't' : 70
        } 
        )

    # make subplot for CRPS:
    fig_crps = make_subplots(cols         = 2,
                        rows         = 1, 
                        shared_xaxes = True,
                        shared_yaxes = False,
                        vertical_spacing    = 0.05,
                        horizontal_spacing    = 0.05,
                        subplot_titles      = [
                            "low flow",
                            "high flow"
                            ],
                        x_title = "forecast horizon (day)",
                        y_title = "CRPS (<i>m<sup>3</sup>/s</i>)"
                        )

    # update layout for CRPS
    fig_crps.update_layout(
        title_text = "<b> CRPS for different bias correction " +
                    "approaches across the 10 day forecast horizon " + 
                    "<br> site = </b>" + site,
        title_x    =  0.5,
        legend     = {
            'x': 0.95,
            'y': 1,
            'itemwidth':40, 
            # 'sizey':0.5
        },
        margin     = {
            'b' : 50,
            't' : 70
        } 
        )

    # choose the deterministic forecast
    det_frcst = 'median'
    col_val = 1
    # loop through the flow conditions:
    for flow_con in ['low', 'high'] :

        # legend decide:
        legend_decide = True if flow_con == 'low' else False
        
        # define color option to be used:
        colors  = iter(pc.qualitative.D3)            

        # loop through the forecast types:
        for fcst_type in fcst_types:
            
            # plot KGE:
            fig.add_trace(
                go.Scatter(
                    x = det_verif.index.get_level_values("day")
                            .unique().values,
                    y = det_verif.xs(fcst_type, level = "fcst_type")
                            .xs(flow_con, level = "flow_clim")
                            .xs(det_frcst, level = "det_frcst")['KGE'],
                    line = dict(
                        color = next(colors), width=4,
                        shape = 'spline'
                        ), 
                    name =  fcst_type, 
                    legendgroup = fcst_type, 
                    showlegend = legend_decide
                ), 
                row = 1, col = col_val
            )

            # plot CRPS aka the probabilistic metric
            fig_crps.add_trace( 
                go.Scatter(
                    x = prob_verif.index.get_level_values("day")
                                    .unique().values, 
                    y = prob_verif.xs(fcst_type, level = "fcst_type")
                                .xs(flow_con, level = "flow_clim")['crps'], 
                    line = dict(
                        color = next(colors), width=4,
                        shape = 'spline'
                        ),
                    name = fcst_type,
                    legendgroup = fcst_type, 
                    showlegend = legend_decide
                    ),
                row = 1, col = col_val
            )

        col_val = 2

    return fig, fig_crps



## FORECAST VS OBSERVATION PLOTS (Flow vs Flow):
def scatter_plttr (df_det, bc_df, clim_vals, day, site,
                fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"],
                renderer = "") :

    # Creates forecast vs observation scatter plots. 
    # Individual scatter plot created for a forecast type, 
    # forecast horizon, site and flow conditions
    for fcst_type in fcst_types:
        flo_events = ["high", "low"]
        for flo_event in flo_events:
            
            # Plot the forecast vs observation for each initialised date 
            fig = go.Figure(
                layout = {
                    "xaxis_title"   : "observations (<i>m<sup>3</sup>/s</i>)",
                    "yaxis_title"   : "forecasted discharge (<i>m<sup>3</sup>/s</i>)",    
                    "title"         : "<b> Forecasts vs Observations Scatter </b> <br> " + 
                                    "<sup> forecat type = " + fcst_type[2:] +
                                    " | forecast horizon = " + str(day) + 
                                    " | site = " + site +
                                    "<br>flow conditions = " + flo_event,
                    "title_x"       : 0.5        
                }
            )

            # possibility to change the axes to logarithmic
            # fig.update_xaxes(type="log")
            # fig.update_yaxes(type="log", range = [2.2,3.56], dtick = "L200")

            if flo_event == "high":
                df  = bc_df[bc_df["Obs"] > clim_vals["q70_flo"]]
            else : df  = bc_df[bc_df["Obs"] < clim_vals["q70_flo"]]

            # add y = x line
            fig.add_trace(
                go.Scattergl(
                    x = np.arange(
                            df.Obs.min()*0.95, 
                            min( df.Q_raw.max(),df.Obs.max() ) * 1.05
                        ), 
                    y = np.arange(
                            min( df.Q_raw.min(),df.Obs.min() ), 
                            min( df.Q_raw.max(),df.Obs.max() )
                        ),
                        name = "y = x", line = {"color":"black"})
            )

            # Ensemble spread for each date
            for date, grouped_df in df.groupby('date'): 
                # only one legend entry for the multiple box plots
                legend_decide = True if date == df.index.get_level_values(1)[0] \
                    else False

                fig.add_trace(
                    go.Box(x = grouped_df["Obs"], y = grouped_df[fcst_type], 
                    line = {"color":"rosybrown"}, legendgroup = "ens_mem",
                    name = "ens spread", showlegend = legend_decide)
                )

            # Deterministic forecasts 
            if flo_event == "high":
                df  = df_det[df_det["Obs"] > clim_vals["q70_flo"]] 
            else : 
                df  = df_det[df_det["Obs"] < clim_vals["q70_flo"]] 

            # matrix of colors for individual deterministic forecast types:
            colors = iter(["cyan", "green", "blue"])
            
            for det_type, group in df.groupby(by = "det_frcst"):
                # one color for each forecast type
                colr_val = next(colors)

                # loop through each date:
                for date, grouped_df in df.groupby('date'): 
                    # only one legend entry for the multiple box plots
                    legend_decide = True if date == df['date'][0] \
                        else False
                    fig.add_trace(
                        go.Scattergl(x = grouped_df["Obs"], y = grouped_df[fcst_type],
                            name = det_type, mode = 'markers', legendgroup = det_type,
                            marker = {"color": colr_val}, showlegend = legend_decide
                        )                      
                    )

            # show the figure
            fig.show(renderer= renderer)

    return None


## PROBABILITY DISTRIBUTION PLOTS (Probability vs Flow):
#


## DMB variation (DMB vs Time)
def dmb_vars_plttr (bc_df, dmb_vars):
    fig = go.Figure(
        layout = {
            "xaxis_title"   : "date",
            "yaxis_title"   : "DMB ratio",    
            "title"         : "Time series progression of DMB values",
            "title_x"       : 0.5
        }  
    )

    color       = pc.qualitative.D3

    for name in dmb_vars:
        # plot all DMB tracers
        # fig.add_trace(
        #     go.Scatter(x = bc_df.reset_index()["date"], 
        #                 y = bc_df[name],
        #                 name = name, opacity = 0.6

        #         )
        # )

        group_df = bc_df.groupby("date").agg(['min', 'max'])[name]

        # add the maximum DMB values
        fig.add_trace(
            go.Scatter(x = group_df.reset_index()["date"], 
                        y = group_df["max"],
                        name = name, opacity = 0.5, 
                        marker_color = color[dmb_vars.index(name)] , 
                        legendgroup = name, showlegend = False
            )
        )    

        # add the minimum DMB values and fill in between 
        fig.add_trace(
            go.Scatter(x = group_df.reset_index()["date"], 
                        y = group_df["min"],
                        name = name, opacity = 0.5, 
                        marker_color = color[dmb_vars.index(name)], 
                        legendgroup = name, fill = "tonexty"
                )
        )    
        
    return fig

## 