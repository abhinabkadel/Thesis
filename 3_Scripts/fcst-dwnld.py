#%%
import datetime as dt
import os 
import wget

#%% user input for download dates
# set the date range that the data is available
# available from Mar 28 2020 -> July 30 2020
begin       = "20210323"
end         = "20210331"

# %%
dwnld_dates = []
begin   = dt.datetime.strptime(begin, "%Y%m%d")
end     = dt.datetime.strptime(end, "%Y%m%d")
delta   = end - begin
for i in range(delta.days + 1):
    day = begin + dt.timedelta(days = i)
    dwnld_dates.append( day.strftime("%Y%m%d") )

# %% Set up the forecast directory:
root        = "http://110.34.30.197:8080/thredds/fileServer/ECMWF/"
# 52 files so range from 1 to 53
fname       = ["Qout_south_asia_geoglowsn_" + fno + ".nc" \
            for fno in list(map(str, range(1,53))) ] 

# %% Loop through the days
for day in dwnld_dates:
    dwnld_lnks = [root + day + ".00/" + no \
                for no in fname]
        # check if download folder exists, if not create it:
    if os.path.isdir(day) == False:
        os.mkdir(day)
    count = 0

    for lnk in dwnld_lnks:
        print("\ndownloading files for " + day + " ...")
        wget.download(lnk, out = "./" + day + "/")
        count = count + 1

    if count == 52:
        print('\nAll 52 files downloaded \n')
    else: 
        print(count + 'files downloaded. Some are missing. Check the download folder')