# %% Set up verification data:
# set verification dates to be 1 year after the calibration date:
verif_date_list = pd.to_datetime(init_date_list).shift(365, freq = "d").strftime("%Y%m%d").values
verif_data      = df_creator(rt_dir, verif_date_list, riv_id, ens_members)
verif_data = verif_data.xs(key = day, level = "day_no")[["Qout","init_date"]].sort_index()
verif_data = pd.merge(verif_data, obs, left_index=True, right_index=True)

# %%
test        = t1[["DMBn"]]
test.index  = test.index.set_levels(test.index.levels[1].shift(365, freq = "d"), 
                level = 1) 

# %% reorder columns
# cols = list(test)
# cols[0], cols[1] = cols[1] , cols[0]
# t1 = t1.loc[:,cols]

# %% Reset all the indices
# reset index:
df = (test.reset_index().sort_index())

# %%
# df.loc[lambda x : ( x["ens_mem"] == 1 ) | ( x["ens_mem"] == 2 ), :]

# %%
def dmb_test(df):
    df.Qout.rolling(2).sum().values / df.Obs.rolling(2).sum().values

test.rolling(2).apply(lambda x: x.assign(DMBn = dmb_test))

test = t1.loc[ pd.IndexSlice[ [1,52],:], :]

# pick data relating to specific ens_mem with the latest date values:
test = df[df["ens_mem"] == 52].sort_values(
        'date', ascending=False).groupby('month').head(3)

# %% MONTHLY CALCULATION OF NSE:
# calculate NSE for the raw ensemble spread:   
NSE = df.groupby(by = ["month", "Obs_mean"],  dropna = False). \
        apply(lambda x:nse_form(x, fcst_type = "Qout")).reset_index()
NSE.rename(columns = {0:'raw'}, inplace = True)
NSE.drop(["Obs_mean"], axis = 1, inplace = True)

# forecast output variables to calculate the bias correction metric:
fcst_type = ["Q_dmb", "Q_ldmb", "Q_med", "Q_mean"]

# Loop through the forecast output variables and calculate the NSE
# coefficient for each of the cases. 
for i in fcst_type:
    if i == "Q_med":
        for j in ["Qout", "Q_dmb", "Q_ldmb"]:
            NSE["med_"+j] = df_med.groupby(by = ["month", "Obs_mean"],  
                        dropna = False). \
                apply(lambda x:nse_form(x, fcst_type = j)). \
                reset_index()[0]

    elif i == "Q_mean":
        for j in ["Qout", "Q_dmb", "Q_ldmb"]:
            NSE["mean_" + j] = df_mean.groupby(by = ["month", "Obs_mean"],  dropna = False). \
                apply(lambda x:nse_form(x, fcst_type = j)). \
            reset_index()[0]

    else:
        NSE[i] = df.groupby(by = ["month", "Obs_mean"],  dropna = False). \
                apply(lambda x:nse_form(x, fcst_type = i)). \
            reset_index()[0]

# %% rearranging the MHPP data:
test = pd.read_csv( os.path.join(obs_dir, "MHPS_DISCHARGE-2077"+".csv"),
            header = 0)
test.head()            
test = pd.melt(test, id_vars = 'Days', var_name = "month", value_name = "discharge" )
test.to_csv(os.path.join(obs_dir, "MHPS_DISCHARGE_long-2077"+".csv"))


# %%
                # add a dummy trace for legend entries:
                # fig.append_trace(
                #     go.Scatter(
                #         x = [1], y = [1],
                #         marker = {
                #             'size'      : 10,
                #             'opacity'   : 0
                #         },
                #         line = dict(
                #             color = 'black', width=2,
                #             dash  = dash_opt
                #             ),                        
                #         name =  det_frcst, 
                #         legendgroup = det_frcst, 
                #         hoverinfo = 'skip', 
                #         showlegend = legend_decide                           
                #     ),row = row, col = col
                # )

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
