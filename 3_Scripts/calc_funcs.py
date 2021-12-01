
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
