#!/Users/akadel/opt/anaconda3/envs/thesis/bin/python
# %%
# run the file by using command: python <script_path> > <text_file_path>
import os
import pandas as pd
import re
import geopandas as gpd
import xarray as xr

# %% FUNCTION 1: EXTRACT DATA POINTS THAT FALL WITHIN NEPAL
def Nepal_extractor(Q_file, npl):
    # open Q_file such that the file is automatically closed after computations 
    # are completed:
    with xr.load_dataset(Q_file) as Q_data:
        # store lat, lon, rivid as pandas dataframe
        Qdat_pts = pd.DataFrame (data = list(
                    zip(Q_data.lon.values, Q_data.lat.values, Q_data.rivid.values) ),
                        columns=['lon', 'lat', 'rivid'] )
        # convert pandas dataframe to shapely
        Qdat_pts_shply  = gpd.points_from_xy(Qdat_pts.lon, Qdat_pts.lat )
        # extract the lat, lon points and corresponding rivid values that fall
        # within Nepal
        Qnpl_pts_shply = Qdat_pts[Qdat_pts_shply.within(npl.geometry[0])]
        # dicharge datasets for Nepal only
        Q_npl = Q_data.sel(rivid = Qnpl_pts_shply.rivid.values)
        # # close the dataset file to delete it later
        # Q_data.close()

    return Q_npl

# %%
src_rt      = r"wfrt:/reforecasts-2/"
src_dir     = pd.date_range(start='20160101', end='20161231').strftime("%Y%m%d.%H").values
dst_rt      = "/Users/akadel/Documents/Kadel/Thesis/Fcst_data"
# rclone_cmd  = "rclone copy --verbose --update --progress --dry-run "
mappath 	= "/Users/akadel/Documents/Kadel/Thesis/Nepal-boundary/data/Outline.shp"
rclone_cmd  = "rclone copy --verbose --update --progress "

# %% load the shapefile for map of Nepal
npl  = gpd.read_file(mappath)

# %%
for folder in src_dir:
    print(folder)
    print("\n")
    for i in [*range(1,52), 52]:
    # for i in [1]:
        print(i)
        fname   = f"Qout_south_asia_mainland_{i:d}.nc"
        src_pth = src_rt + folder + "/" + fname
        dst_pth = os.path.join(dst_rt, folder[:-3])
        os.system(rclone_cmd + src_pth + " " + dst_pth)

        # variable for the downloaded filepath
        Q_file   = os.path.join(dst_pth, fname)
        print(f"{os.path.join(folder[:-3], fname)} downloaded")
        
        # path for file containing Nepal data only 
        sav_pth = os.path.join( dst_pth, 
               re.split(r"[\\]", Q_file)[-1]) \
                    .replace("south_asia_mainland","npl")

        # check if the save file already exists
        # if os.path.isfile(sav_pth) == False:

        # Extract data points that fall within Nepal
        Q_npl = Nepal_extractor(Q_file, npl)
        Q_npl.to_netcdf(path = sav_pth)
        print(f"{sav_pth} saved \n")

        os.remove(Q_file)

# %% plot to test:
