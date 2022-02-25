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
from hydrostats.ens_metrics import ens_crps
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
            df = df.rename(columns={"Qout":"Q_raw"})

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
    # end for forecast run dates

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

    # calculate Q60 flow, low and high season mean flow values:
    q60_flo     = obs.quantile(q = 0.6, axis = 0, 
            numeric_only = True, interpolation = "linear")[0]
    lo_flo_clim = obs[obs["Obs"] <= q60_flo][["Obs"]].mean().values
    hi_flo_clim = obs[obs["Obs"] > q60_flo][["Obs"]].mean().values

    # merge the forecasts and the observations datasets together. 
    # perform a left join with fcsts being the left parameter:
    df = pd.merge( fcst_df.xs(key = day, level = "day_no")
                    [["Q_raw","init_date"]],
                    obs, left_index=True, 
                    right_index=True).sort_index()

    # create a python dictionary with the observed values:
    clim_vals = {
        "q60_flo"       : q60_flo,
        "lo_flo_clim"   : lo_flo_clim,
        "hi_flo_clim"   : hi_flo_clim
    }

    return df, clim_vals

# calculate Degree of Mass Balance (DMB):
def dmb_calc(df, window, variation = "dmb"):
    
    # implementation based on Dominique:
    if variation == "ldmb":
        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.vstack(Q_wins) * wts * np.reciprocal(np.vstack(Obs_wins))
        # add padding and sum the array
        df[variation] = np.pad(np.sum(wt_DMB, axis = 1), 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)

    # Unweighted Degree of Mass Balance:
    else:
        df[variation] =  df.Q_raw.rolling(window).sum().values / \
            df.Obs.rolling(window).sum().values
    
    return df

# Bias correct forecasts (each ensemble member is independent) :
def bc_fcsts(df, win_len ):     

    dmb_vars = ["dmb", "ldmb"]

    for variation in dmb_vars:
        
        # Calculate dmb ratio:
        df = df.groupby(by = "ens_mem").apply(
            lambda x:dmb_calc(x, window = win_len, variation = variation)
        )    

        # APPLY BIAS CORRECTION FACTOR:
        df = df.groupby(by = "ens_mem", dropna = False).     \
            apply(lambda df:df.assign(
                new_val = df["Q_raw"].values / df[variation].shift(periods=1).values )
            ).rename(
                columns = {'new_val':"Q_"+variation}
            ).sort_index()   

    return df

# function to create determininstic forecasts (mean, median and high-res):
def det_frcsts (df, fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"]):    
    # ensemble median:
    df_med  = df.groupby(by = "date").median().reset_index() \
        [["date","Obs"] + fcst_types]
    # ensemble mean:
    df_mean = df.groupby(by = "date").mean().reset_index() \
        [["date","Obs"] + fcst_types]
    # high-res forecast
    df_highres = df[df["ens_mem"] == 52] \
        [["date","Obs"] + fcst_types]

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
def metric_calc(df_det, clim_vals, 
    fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"]):
    
    # defines dataframes for low_flow and high_flow values
    df_low  = df_det[df_det["Obs"] <= clim_vals["q60_flo"]]
    df_high = df_det[df_det["Obs"] > clim_vals["q60_flo"]]

    # loop through the two dataframes to create:
    for df in [df_low, df_high]:
        flo_mean = clim_vals["lo_flo_clim"] if df.equals(df_low) \
            else clim_vals["hi_flo_clim"]
        
        # loop through win len
        # fcst_Day
        data = []
        
        # loop through the raw and bias corrected forecasts:
        for i in fcst_types:
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

        # end for along fcst_types

        if flo_mean == clim_vals["lo_flo_clim"]:
            lo_verif = pd.concat(data)

        else : 
            hi_verif = pd.concat(data)
        
    return lo_verif, hi_verif

# calculate probabilistic verification metrics:
def prob_metrics(bc_df, clim_vals,
    fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"]):
    
    prob_verif = []
    # Calculate CRPS for high and low flow conditions:
    for flo_con in ["low", "high"]:

        # define the subset of dataset to work with:
        if flo_con == "low":
            df  = bc_df[bc_df["Obs"] <= clim_vals["q60_flo"]]
        else :
            df  = bc_df[bc_df["Obs"] > clim_vals["q60_flo"]]

        crps_vals = []
        # loop through the raw and bias corrected forecasts:
        for i in fcst_types:

            ## CRPS Hydrostats:
            # frcsts  = df[["Q_dmb"]].reset_index().pivot(
            #     index = "date", columns = "ens_mem", values = "Q_dmb").values
            # obs     = df.xs(key = 52)["Obs"].values

            # crps = ens_crps(obs, frcsts)

            # CRPS xskillscore 
            ds           = df.xs(key = 1)[["Obs"]].to_xarray()
            ds['frcsts'] = df.reorder_levels(["date", "ens_mem"]) \
                                .sort_index().to_xarray()[i]
            ds      = ds.rename_dims({"ens_mem":"member"})
            crps    = ds.xs.crps_ensemble('Obs', 'frcsts').values

            crps_vals.append(crps)
            
            # end for along fcst_types

        # create a df with the crps values and the forecast type
        data = pd.DataFrame(
            {"fcst_type":fcst_types, 
            "crps":np.array(crps_vals)}
        )
        # add a column about the flow climatology information:
        data["flow_clim"] = flo_con 
        prob_verif.append(data)

    # end for along flow climatology
    prob_verif = pd.concat(prob_verif)
    prob_verif = prob_verif.set_index(["flow_clim", "fcst_type"], 
                append= True).droplevel(0).sort_index()

    return prob_verif

# Integrate overall bias correction process in the function :
# input df already contains observations as well
def post_process(fcst_data, win_len, clim_vals,
        fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"]):

    # Bias correct the forecasts using DMB and LDMB
    bc_df = bc_fcsts(df = fcst_data, win_len = win_len )

    # Separate dataframes for deterministic forecasts:
    # df = t1.reset_index()
    df_det = det_frcsts(bc_df.reset_index(), fcst_types)

    # calculate the metrics:
    lo_verif, hi_verif = metric_calc(df_det, clim_vals, fcst_types)

    # calculate probabilitic verification (CRPS):
    prob_verif = prob_metrics(bc_df, clim_vals, fcst_types)

    return lo_verif, hi_verif, bc_df, prob_verif, df_det

# forecast calibration:
def fcst_calibrator (fcst_data, clim_vals, 
        fcst_types = ["Q_raw", "Q_dmb", "Q_ldmb"]):

    windows     = [2, 3, 5, 7, 10, 15, 20, 30]
    lo_verif    = []
    hi_verif    = []
    prob_verif  = []
    for win_len in windows:
        lo_df, hi_df, bc_df, prob_df =  post_process(fcst_data, win_len, 
                            clim_vals, fcst_types)
        lo_df["win_length"]     = win_len
        hi_df["win_length"]     = win_len
        prob_df["win_length"]   = win_len
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)
        prob_verif.append(prob_df)

    # create a single large dataframe for low and high flow seasons:
    lo_verif    = pd.concat(lo_verif)
    hi_verif    = pd.concat(hi_verif)
    prob_verif  = pd.concat(prob_verif)

    lo_verif = lo_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    hi_verif = hi_verif.set_index(["win_length", "fcst_type"], append= True
                    ).reorder_levels(["win_length", "fcst_type", "det_frcst"]).sort_index()
    prob_verif = prob_verif.set_index(["win_length"], append= True
                    ).reorder_levels(["win_length", "flow_clim", "fcst_type"]).sort_index()
    
    return lo_verif, hi_verif, prob_verif

## DMB VARIATIONS FUNCTIONS:
def dmb_calc_variations(df, window, variation = "dmb"):
    
    # implementation based on Dominique:
    if variation == "ldmb":
        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the dmb parameter:
        wt_DMB = np.vstack(Q_wins) * wts * np.reciprocal(np.vstack(Obs_wins))
        # add padding and sum the array
        df[variation] = np.pad(np.sum(wt_DMB, axis = 1), 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)
        return df

    # Vaariation of the original implementation, to make it align with original
    # un-weighted dmb implementation
    # take ratio of the sum of the weighted forecasts and observations
    if variation == "ldmb-var":

        # define the weights applied:
        wts = ( window + 1 - np.arange(1,window+1) ) / sum(np.arange(window+1))

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.sum( np.vstack(Q_wins) * wts, axis = 1) / \
                np.sum(np.vstack(Obs_wins) * wts, axis = 1)
        # add padding and sum the array
        df[variation] = np.pad(wt_DMB, 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)
        return df
        
    # take the sum of the ratios 
    if variation == "dmb-var":

        # define lists that will house the rolling values of raw forecasts and observations:
        Q_wins = []
        Obs_wins = []
        # add rolling values of observations and raw forecasts to the list 
        df['Q_raw'].rolling(window).apply(lambda x:Q_wins.append(x.values) or 0)
        df['Obs'].rolling(window).apply(lambda x:Obs_wins.append(x.values) or 0)
        # convert both lists to (N_days - win_len + 1) x win_len 2d numpy arrays and then 
        # calculate the DMB parameter:
        wt_DMB = np.sum ( np.vstack(Q_wins)/ \
                np.vstack(Obs_wins) , axis = 1)
        # add padding and sum the array
        df[variation] = np.pad(wt_DMB, 
                    pad_width = (window-1,0), 
                        mode = "constant", 
                            constant_values = np.nan)

        return df

    # dmb original implementation based on McCollor & Stull, Bourdin 2013:
    # ratio of the sums:
    else:
        df[variation] =  df.Q_raw.rolling(window).sum().values / \
            df.Obs.rolling(window).sum().values
        return df

def bc_fcsts_variations(df, win_len):     
    
    dmb_vars = ["dmb", "ldmb", "dmb-var", "ldmb-var"]
    
    for variation in dmb_vars:
        
        # Calculate dmb ratio:
        df = df.groupby(by = "ens_mem").apply(
            lambda x:dmb_calc_variations(x, window = win_len, variation = variation)
        )    

        # APPLY BIAS CORRECTION FACTOR:
        df = df.groupby(by = "ens_mem", dropna = False).     \
            apply(lambda df:df.assign(
                new_val = df["Q_raw"].values / df[variation].shift(periods=1).values )
            ).rename(
                columns = {'new_val':"Q_"+variation}
            ).sort_index()
    
    return df

def dmb_vars_test(fcst_types, days, win_len, site, fcst_data, obs_dir): 
    lo_verif    = []
    hi_verif    = []
    prob_verif  = []
    # loop through the forecast horizons
    for day in days:

        # add observations:
        [fcst_data_day, clim_vals] = add_obs(
        place = site, fcst_df = fcst_data, obs_dir = obs_dir, day = day)

        bc_df   = bc_fcsts_variations(fcst_data_day, win_len)   
        df_det  = det_frcsts(bc_df.reset_index(),
                fcst_types)

        # deterministic metric:
        lo_df, hi_df = metric_calc(df_det, clim_vals, fcst_types)

        # calculate probabilitic verification (CRPS):
        prob_df = prob_metrics(bc_df, clim_vals, fcst_types)

        lo_df["day"]   = day
        hi_df["day"]   = day
        prob_df["day"] = day

        # one large df with day information
        lo_verif.append(lo_df) 
        hi_verif.append(hi_df)
        prob_verif.append(prob_df)

    # create a single large dataframe for low and high flow seasons:
    lo_verif    = pd.concat(lo_verif)
    hi_verif    = pd.concat(hi_verif)
    prob_verif  = pd.concat(prob_verif)

    lo_verif["flow_clim"]   = "low"
    hi_verif["flow_clim"]   = "high"

    det_verif   = pd.concat([lo_verif, hi_verif])

    det_verif    = det_verif.set_index(["day", "flow_clim", "fcst_type"], append= True
                    ).reorder_levels(["day", "flow_clim", "fcst_type", "det_frcst"]).sort_index()
    prob_verif  = prob_verif.set_index(["day"], append= True
                    ).reorder_levels(["day", "flow_clim", "fcst_type"]).sort_index()

    return det_verif, prob_verif

# RUNOFF DATASET CREATION
def runoff_data_creator(site, date_range):

    wt_df = pd.read_pickle("./pickle_dfs/" + site + "_wt.pkl")

    rt_dir      = r"../1_Data/runoff_forecasts/"
    
    init_date_list = pd.date_range(start = date_range[0], 
                        end = date_range[1]).strftime("%Y%m%d").values
    runoff_data = []
    for init_date in init_date_list:
        # loop through the ensemble members:
        for ens_mem in np.arange(1,53): 
            # forecast filter points to for high/low res forecasts:
            filtr_pts = wt_df.xs('high') if ens_mem == 52 \
                else wt_df.xs('low')

            # load the forecast files:
            fname       = f'runoff_{ens_mem:d}.nc'    
            file_pth    = os.path.join(rt_dir, init_date, fname)    
            data        = xr.open_dataset(file_pth)

            runoff_vals = []

            # resample the forecasts to daily 
            test = data.RO.resample(time = '1D').mean()
            
            # loop through the forecast grids that intersect with the 
            # catchment:
            for i in range(len(filtr_pts)):

                # substitution to make code readable:
                easy_var    = test.sel(lon = filtr_pts.lon[i],
                            lat = filtr_pts.lat[i],
                            method= 'nearest')

                easy_var[:] = easy_var * filtr_pts.weight[i]/100 \
                    * filtr_pts.grid_area[i]

                runoff_vals.append(easy_var)

            # sum the runoff values to produce total runoff time series 
            # for the catchment:
            catch_RO = np.sum(runoff_vals, axis = 0)
            df = pd.DataFrame(
                {
                    'runoff': catch_RO
                },
                index = test.time.values
            )
            df.index.name = 'date'

            # set the ensemble value based on the range index
            df['ens_mem'] = ens_mem

            # add in information on initial date:
            df["init_date"] = init_date

            # # specify the day of the forecast
            df["day_no"] = 1 + (df.index.get_level_values('date') -  
                            pd.to_datetime(init_date, 
                                format = '%Y%m%d')).days 

            runoff_data.append(df)

        # end for ensemble list
    # end for forecast run dates

    runoff_data = pd.concat(runoff_data)
    runoff_data.set_index(['ens_mem', 'day_no'] , append = True, inplace = True )
    runoff_data = runoff_data.reorder_levels(["ens_mem", "day_no", "date"]).sort_index()

    runoff_data.to_pickle("./pickle_dfs/" + site + "_runoff.pkl")

    return runoff_data