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
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
fig.suptitle("day 2 forecasts initialised for different dates for 2014 January", 
            y = 0.96 )
fig.text(0.02, 0.5, "Flow ($m^3/s$)", va = "center", rotation = "vertical", 
            fontsize = "large")
fig.subplots_adjust(left = 0.12, hspace = 0.3)

sn.set(style = "darkgrid")
# plot the high-resolution forecast:
p1 = sn.scatterplot(x = "init_date", y = "Qout", data = df[df["ens_mem"] == 52], 
                color = "black", ax = ax[0], label = "high-res", legend = False)
sn.scatterplot(x = "init_date", y = "Q_bc", data = df[df["ens_mem"] == 52], 
                color = "black", ax = ax[1])
# plot the observations:
ax[0].plot(df.groupby("init_date")['Obs'].mean(), "ro", label = "observations")
ax[1].plot(df.groupby("init_date")['Obs'].mean(), "ro")

# plot raw forecasts
sn.violinplot(x = "init_date", y = "Qout", data = df, ax = ax[0], 
                color = "skyblue", width = 0.5 , linewidth = 2)
# plot bias corrected forecasts:
sn.boxplot(x = "init_date", y = "Q_bc", data = df, ax = ax[1], 
                color = "skyblue", width = 0.5)

# aesthetic changes:
ax[0].set_xlabel("")
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[1].set_xlabel("initial date")
ax[0].set_title("Raw forecasts")
ax[1].set_title("bias corrected forecasts")

# add a legend:
fig.legend(loc = "center right",
            title = "Legend")

plt.show()

# %%
# df.loc[lambda x : ( x["ens_mem"] == 1 ) | ( x["ens_mem"] == 2 ), :]

# %%


def dmb_test(df):
    df.Qout.rolling(2).sum().values / df.Obs.rolling(2).sum().values

test.rolling(2).apply(lambda x: x.assign(DMBn = dmb_test))

test = t1.loc[ pd.IndexSlice[ [1,52],:], :]