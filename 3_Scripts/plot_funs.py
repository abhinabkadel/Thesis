# %% 
import contextily as ctx
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd

#%% Import dataset 
# Qout_file   = r"E:\Toolkit\Forecast-downloads\20-04April\20200401\Qout_south_asia_geoglowsn_51.nc"'
Qout_file   = r"D:\\Masters\\Thesis\\Test_downloads\\20140101\\Qout_south_asia_mainland_1.nc"
nc_dat      = xr.open_dataset(Qout_file)


# %% Original coverage of the operational forecasts 
fig, ax1 = plt.subplots()
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('HKH streamflow prediction tool coverage') 

# jump by 20 values while plotting to speedup 
ax1.plot( nc_dat.lon.values[:], nc_dat.lat.values[:], 
            'x', alpha = 0.5, markersize = 3)

ctx.add_basemap(ax1,     
    crs = "EPSG:4326",
    source = ctx.providers.Stamen.TonerLite
    )

# %% add locations of the hydro dams:
# # Naugad:
# ax1.plot(80.591, 29.720, 's', markersize = 6, label = "Naugad")
# # Kali Gandaki
# ax1.plot(83.585, 27.925, 's', markersize = 6, label = "Kali Gandaki A")
# ax1.plot(84.496, 27.884, 's', markersize = 6, label = "Marsyangdi")
# ax1.plot(84.869, 27.987, 's', markersize = 6, label = "Ankhu")
# ax1.plot(87.880, 26.826, 's', markersize = 6, label = "Mai")
# ax1.plot(87.818, 26.723, 's', markersize = 6, label = "Mai-Cascade")

# # Naugad:
# ax1.plot(80.591, 29.720, 'sr', markersize = 6, label = "run-of-river weir")
# # Kali Gandaki
# ax1.plot(83.585, 27.925, 'sr', markersize = 6, label = "")
# ax1.plot(84.496, 27.884, 'sr', markersize = 6, label = "")
# ax1.plot(84.869, 27.987, 'sr', markersize = 6, label = "")
# ax1.plot(87.880, 26.826, 'sr', markersize = 6, label = "")
# ax1.plot(87.818, 26.723, 'sr', markersize = 6, label = "")


# %% All done show the map:
plt.legend()
plt.show()

# fname = os.path.join("my_plot.svg")
# plt.savefig(fname = fname, 
#             bbox_inches = "tight",
#             dpi = 300)