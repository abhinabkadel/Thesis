from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

## TIME SERIES PLOTS (Flow vs Time):
# time series for all 3 bias correction options:
def time_series_plotter(df, site, day, win_len):
    # make subplot interface
    fig = make_subplots(rows = 3, cols = 1,
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
            f"<br> site = {site}, horizon = {day}, window = {win_len}",
        title_x = 0.5,
        legend_title = "Legend", 
        yaxis_rangemode = "tozero"
        )
    # loop through the forecast types:
    fcst_types = ["Qout", "Q_dmb", "Q_ldmb"]
    for type in fcst_types:
        legend_decide = True if type == "Qout" else False

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
def time_series_individual(df, site, day, win_len, type):
    fig = go.Figure(
        layout = {
            "xaxis_title" : "date",
            "yaxis_title" : "River discharge (<i>m<sup>3</sup>/s</i>)"    
        }  
    )

    # Add figure and legend title                  
    fig.update_layout(
        title_text = "Bias-correction for streamflow forecasts"+
            f"<br> site = {site}, forecast horizon = {day}, window = {win_len}",
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
# deterministic forecasts only:
# add option for CRPS:
def calibrtn_plttr (hi_verif, lo_verif, site, day, flo_con = "high"):
    if flo_con == "high" : 
        df_big = hi_verif
    elif flo_con == "low" :
        df_big = lo_verif
    else : print ("wrong input")

    df_big = df_big.xs('median', level = 2)

    # make subplot interface
    fig = make_subplots(rows = 2, cols = 1,
                        shared_xaxes = True,
                        shared_yaxes = True,
                        vertical_spacing = 0.09,
                        subplot_titles=("DMB", "LDMB"),
                        x_title = "window length (days)",
                        y_title = "Score",
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

    # loop through the forecast types:
    fcst_types = ["Q_dmb", "Q_ldmb"]
    for type in fcst_types:
        df = df_big.xs(type, level = 1)
        
        # show only one legend entry per verification metric:
        legend_decide = True if type == "Q_dmb" else False
        
        # define color to be used:
        color   = pc.qualitative.D3
        # metrics to plot:
        metrics = ["NSE", "r", "flo_var", "bias", "KGE"]
        
        for metric in metrics:     
            # plot different metrics:
            fig.append_trace( 
                go.Scatter(x = df.index, 
                        y = df[metric], 
                        name = metric,
                        legendgroup = metric,
                        marker_color = color[metrics.index(metric)],
                        showlegend = legend_decide
                        ),
                row = fcst_types.index(type) + 1, col = 1
            )

    return fig



## FORECAST VS OBSERVATION PLOTS (Flow vs Flow):
#

#

#

## PROBABILITY DISTRIBUTION PLOTS (Probability vs Flow):
#


