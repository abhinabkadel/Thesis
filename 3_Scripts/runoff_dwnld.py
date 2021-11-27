# %%
# import os
import pandas as pd
# import re
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
# %%
src_rt      = r"/Users/akadel/Documents/Kadel/Thesis/1_Data/runoff_forecasts"
rclone_cmd  = "rclone copy --verbose --update --progress "
mappath 	= "/Users/akadel/Documents/Kadel/Thesis/1_Data/GIS_data/Nepal-GIS_files/Nepal-boundary/data/Outline.shp"

# %%
folder  = "/Runoff.20140101.00.exp1.Fgrid.netcdf"
i = 5
# %% 
fname = f"{i:d}.runoff.nc"
src_pth = src_rt + folder + "/" + fname
npl         = gpd.read_file(mappath)
# open Q_file such that the file is automatically closed after computations 
# are completed:
# with xr.load_dataset(fname) as runoff_data:
# %%
runoff_data = xr.load_dataset(src_pth)

# %%
# extract runoff data only:
data = runoff_data.RO
# get coordinate points as numpy array
grdpts = np.array(np.meshgrid(data.lon.values, data.lat.values)).T.reshape(-1,2)
grdpts_pd = pd.DataFrame (data = grdpts, index=None,
                columns=['longitude', 'latitude'])
# filter
nepal_bounds = npl.bounds

# narrow down the points to those around Nepal only:
data.sel(
        lon=slice(nepal_bounds.minx.values[0], nepal_bounds.maxx.values[0]),
        lat=slice(nepal_bounds.maxy.values[0], nepal_bounds.miny.values[0])
) 



# # %%

# fig, ax1 = plt.subplots()
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.title('HKH streamflow prediction tool coverage') 

# # jump by 20 values while plotting to speedup 
# ax1.plot( grdpts_pd.longitude.values, grdpts_pd.latitude.values, 
#         '.', alpha = 0.3, markersize = 5)

# # add a basemap
# npl_crs = npl.crs.to_string()
# ctx.add_basemap(ax1, 
#         crs = npl_crs, zoom = 10,
#         source = ctx.providers.Stamen.TonerHybrid
#         )

# plt.savefig(fname = "my_plot.svg",
#             bbox_inches = "tight", dpi = 300)

# # %%

# # %%