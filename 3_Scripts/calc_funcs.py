"""
    This module contains all the functions used to set up
    the post-processing of the streamflow forecasts:
"""
######### Dependednt Modules ###########
# Import necessary modules
# read netcdf files:
import xarray as xr
# dataframe and data analysis
import pandas as pd
import numpy as np
# error metric calculations:
from hydrostats import HydroErr
from scipy import stats
import xskillscore as xs
# use os commands:
import os 


# create single pandas dataframe from 52 ens. mems. across given 
# time period:
def df_creator(rt_dir, date_range, riv_id, ens_members):
    init_date_list  = pd.date_range(start=date_range[0], 
            end=date_range[1]).strftime("%Y%m%d").values
    fcst_data = []
    for init_date in init_date_list:
        for i in ens_members:
            # for ICIMOD archives:
            # fname       = f"Qout_npl_geoglowsn_{i:d}.nc"
            # for Jorge's data:
            fname       = f"Qout_npl_{i:d}.nc"
            
            file_pth    = os.path.join(rt_dir, init_date, fname)

            # Create a mini-dataframe:
            # mini_df contains data of individual ens member
            data  = xr.open_dataset(file_pth)
            Qfcst = data.sel(rivid = riv_id)

            # resample the data to daily 
            df    = Qfcst.Qout.to_dataframe()
            df    = df.resample('D').mean()
            df.index.names = ['date']

            # set the ensemble value based on the range index
            df['ens_mem'] = i

            # add in information on initial date:
            df["init_date"] = init_date

            # specify the day of the forecast
            df["day_no"] = 1 + (df.index.get_level_values('date') -  
                            pd.to_datetime(init_date, 
                                format = '%Y%m%d')).days 

            # append the fcst_data with mini_df
            fcst_data.append(df)
        # end for ensemble list
    # end for montlhly file list

    # concatanate individual mini-dfs to create a dataframe for the time period
    fcst_data = pd.concat(fcst_data)
    fcst_data.set_index(['ens_mem', 'day_no'] , append = True, inplace = True )
    fcst_data = fcst_data.reorder_levels(["ens_mem", "day_no", "date"]).sort_index()
    
    return fcst_data

# Add observations to the dataframe:
def add_obs(place, obs_dir, day, fcst_df):
    # Load the observations csv file and load the dataframe
    # make the data compatible with the fcst dataframe format
    obs = pd.read_csv( os.path.join(obs_dir, place+".txt"), 
            names = ["date", "Obs"], skiprows=2, parse_dates=[0], 
            infer_datetime_format=True, index_col = [0])

    # calculate Q70 flow, low and high season mean flow values:
    q70_flo     = obs.quantile(q = 0.7, axis =0, 
            numeric_only = True, interpolation = "linear")[0]
    lo_flo_clim = obs[obs["Obs"] <= q70_flo][["Obs"]].mean().values
    hi_flo_clim = obs[obs["Obs"] > q70_flo][["Obs"]].mean().values

    # merge the forecasts and the observations datasets together. 
    # perform a left join with fcsts being the left parameter:
    df = pd.merge( fcst_df.xs(key = day, level = "day_no")
                    [["Qout","init_date"]],
                    obs, left_index=True, 
                    right_index=True).sort_index()
    return df, q70_flo, lo_flo_clim, hi_flo_clim

# calculate Degree of Mass Balance (DMB):
def dmb_calc(df, window, weight = False):
    if weight == True:
        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Qout'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.vstack(Q_wins) * wts * np.reciprocal(np.vstack(Obs_wins))
        # add padding and sum the array
        df["LDMB"] = np.pad(np.sum(wt_DMB, axis = 1), 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)
        return df

    else:
        return df.Qout.rolling(window).sum().values / df.Obs.rolling(window).sum().values

# Bias correct forecasts (each ensemble member is independent) :
def bc_fcsts(df, win_len):     
    # Calculate DMB ratio:
    # un-weighted:
    df["DMB"] = dmb_calc(df.groupby(by = "ens_mem", dropna = False), window = win_len)
    # weighted DMB:
    df = df.groupby(by = "ens_mem").apply(lambda x:dmb_calc(x, window = win_len, weight =  True))

    # APPLY BIAS CORRECTION FACTOR:
    # new column for un-weighted DMB bias correction: 
    df = df.groupby(by = "ens_mem", dropna = False).     \
        apply(lambda df:df.assign(
            Q_dmb = df["Qout"].values / df["DMB"].shift(periods=1).values )
            ).sort_index()
    # new column for weighted DMB bias correction:
    df = df.groupby(by = "ens_mem", dropna = False).     \
        apply(lambda df:df.assign(
            Q_ldmb = df["Qout"].values / df["LDMB"].shift(periods=1).values )
            ).sort_index()

    return df

# function to create determininstic forecasts (mean, median and high-res):
def det_frcsts (df):    
    # ensemble median:
    df_med  = df.groupby(by = "date").median().reset_index() \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
    # ensemble mean:
    df_mean = df.groupby(by = "date").mean().reset_index() \
        [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]
    # high-res forecast
    df_highres = df[df["ens_mem"] == 52] \
    [["date","Obs","Qout","Q_dmb", "Q_ldmb"]]

    # concatenate the 3 deterministic forecast matrices to create 
    # a single deterministic dataframe:
    df_det = pd.concat([df_med, df_mean, df_highres], keys=["median", "mean", "high-res"])
    df_det = df_det.droplevel(1)
    df_det.index.names = ["det_frcst"]

    return df_det

# calculate Nash-Scutliffe efficiency:
def nse_form(df, flo_mean, fcst_type = "Q_dmb"):
    # formula for NSE
    NSE = 1 - \
        ( np.nansum( (df[fcst_type].values - df["Obs"].values) **2 ) ) / \
        ( np.nansum( (df["Obs"].values - flo_mean) **2 ) )
    return NSE

# KGE + correlation, bias and flow variability:
def kge_form(df, fcst_type = "Q_dmb"):
    # calculate pearson coefficient:
    correlation      = HydroErr.pearson_r(df[fcst_type], df["Obs"])
    # calculate flow variability error or coef. of variability:
    flow_variability = stats.variation(df[fcst_type], nan_policy='omit') / \
                        stats.variation(df["Obs"], nan_policy='omit')
    # calculate bias:
    bias = df[fcst_type].mean() / df["Obs"].mean()
    # calculate KGE
    KGE  = 1 - (
            (correlation - 1)**2 + (flow_variability - 1)**2 + (bias - 1)**2 
        )**0.5
    # KGE using the HydroErr formula:
    # print(HydroErr.kge_2012(df[fcst_type], df["Obs"]))
    
    return pd.DataFrame(np.array([[correlation, flow_variability, bias, KGE]]))

# calculate deterministic verification metrics:
def metric_calc(df_det, q70_flo, lo_flo_clim, hi_flo_clim):
    # defines dataframes for low_flow and high_flow values
    df_low  = df_det[df_det["Obs"] <= q70_flo]
    df_high = df_det[df_det["Obs"] > q70_flo]

    # loop through the two dataframes to create:
    for df in [df_low, df_high]:
        flo_mean = lo_flo_clim if df.equals(df_low) else hi_flo_clim
        
        # loop through win len
        # fcst_Day
        data = []
        fcst_type = ["Qout", "Q_dmb", "Q_ldmb"]
        # loop through the raw and bias corrected forecasts:
        for i in fcst_type:
            # NSE:
            NSE = df.groupby(by = "det_frcst").apply(
                    lambda x:nse_form(x, flo_mean, i)
                )
            # KGE:
            kge = df.groupby(by = "det_frcst").apply(
                    lambda x:kge_form(x, i)
                )

            # concatenate and create a dataframe
            verifs = pd.concat([NSE, kge.droplevel(1)], axis = 1).set_axis([
                "NSE", "r", "flo_var", "bias", "KGE"], axis = 1
                )
            # new index with the fcst_type information:
            verifs["fcst_type"] = i

            data.append(verifs)

        # end for along fcst_type

        if flo_mean == lo_flo_clim:
            lo_verif = pd.concat(data)
            # lo_verif = lo_verif.set_index(["fcst_type"], append= True
            #     ).reorder_levels(["fcst_type", "det_frcst"])

        else : 
            hi_verif = pd.concat(data)
            # hi_verif = hi_verif.set_index(["fcst_type"], append= True
            #     ).reorder_levels(["fcst_type", "det_frcst"])

    return lo_verif, hi_verif

# Integrate overall bias correction process in the function :
# input df already contains observations as well
def post_process(fcst_data, win_len, q70_flo, lo_flo_clim, hi_flo_clim):

    # Bias correct the forecasts using DMB and LDMB
    bc_df = bc_fcsts(df = fcst_data, win_len = win_len )

    # Separate dataframes for deterministic forecasts:
    # df = t1.reset_index()
    df_det = det_frcsts(bc_df.reset_index())

    # calculate the metrics:
    lo_verif, hi_verif = metric_calc(df_det, q70_flo, lo_flo_clim, hi_flo_clim)

    return lo_verif, hi_verif, bc_df

# forecast calibration:
def fcst_calibrator (fcst_data, q70_flo, lo_flo_clim, hi_flo_clim):

    windows = [2, 3, 5, 7, 10, 15, 20, 30]
    lo_verif = []
    hi_verif = []

    for win_len in windows:
        lo_df, hi_df, bc_df =  post_process(fcst_data, win_len, 
                            q70_flo, lo_flo_clim, hi_flo_clim)
        lo_df["win_length"] = win_len
        hi_df["win_length"] = win_len 
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)

    # create a single large dataframe for low and high flow seasons:
    lo_verif = pd.concat(lo_verif)
    hi_verif = pd.concat(hi_verif)
    lo_verif = lo_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    hi_verif = hi_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()

    return lo_verif, hi_verif
