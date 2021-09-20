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

# %%
test = pd.merge(verif_data, test, left_index=True, right_index=True)
test["Q_bc"] = test.Qout / test.DMBn

# %% reorder columns
# cols = list(test)
# cols[0], cols[1] = cols[1] , cols[0]
# t1 = t1.loc[:,cols]

# %% Reset all the indices
# reset index:
df = (test.reset_index().sort_index())
df

# %% prepare plot

# %%
# df.loc[lambda x : ( x["ens_mem"] == 1 ) | ( x["ens_mem"] == 2 ), :]

# %%


def dmb_test(df):
    df.Qout.rolling(2).sum().values / df.Obs.rolling(2).sum().values

test.rolling(2).apply(lambda x: x.assign(DMBn = dmb_test))

test = t1.loc[ pd.IndexSlice[ [1,52],:], :]