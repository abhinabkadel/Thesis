#%% Import packages
import contextily as ctx
import geopandas as gpd
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shapely.speedups
import xarray as xr


# %% HANDY FUNCTION 1: Plot
def plotter(data, npl, npl_data, typ):
	# data is the original HKH xarray Dataset
	# npl is the gpd data corresponding to map of Nepal
	# npl_data is the subset of HKH data corresponding to Nepal. Can be 
	# either:
		# pandas dataframe (lon, lat, rivid)
		# xarray Dataset 
	# +++++++++ HKH datapoints coverage +++++++++++
	fig, ax1 = plt.subplots()
	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.title('HKH streamflow prediction tool coverage') 

	# jump by 20 values while plotting to speedup 
	ax1.plot( data.lon.values[::20], data.lat.values[::20], '.', alpha = 0.5, markersize = 5)

	# add a basemap
	npl_crs = npl.crs.to_string()
	ctx.add_basemap(ax1, 
			crs = npl_crs,
			source = ctx.providers.Stamen.TonerLite
			)
	plt.show()

	# ++++++++++ Nepal datapoints coverage +++++++++++
	fig, ax1 = plt.subplots()
	plt.xlabel('longitude')
	plt.ylabel('latitude')
	plt.title('Streamflow prediction tool-Nepal coverage')
	# plot map of Nepal and then the filtered datapoints:
	npl_arr = np.array(npl.geometry[0].exterior)
	if typ == 'pandas':
		ax1.plot( npl_data.longitude.values, npl_data.latitude, '.', alpha = 0.5, markersize = 5)
	elif typ == 'xarray':
		ax1.plot( npl_data.lon.values, npl_data.lat, '.', alpha = 0.5, markersize = 5)
	ax1.plot( npl_arr[:,0], npl_arr[:,1])
	plt.show()
	# fname = os.path.join("my_plot.svg")
	# plt.savefig(fname = fname, 
	#             bbox_inches = "tight",
	#             dpi = 300)

# %% HANDY FUNCTION 2: River ID extractor
def rivid_extractor(dat_pts, npl):
	# convert the lat, lon points to shapely points
	dat_pts_shply = gpd.points_from_xy(dat_pts.longitude, dat_pts.latitude)
	
	# Filter only the data points that lie within Nepal
	shapely.speedups.enable()
	data_filtrd = dat_pts[ dat_pts_shply.within( npl.geometry[0] ) ]

	return data_filtrd

def main(Qout_file, mappath, plt_maps, count):
	#%% Read the data and the map files:
	data = xr.open_dataset(Qout_file)
	npl  = gpd.read_file(mappath)
	# convert lat, lon to shapely points
	dat_pts = pd.DataFrame (data = list(zip(data.lon.values, data.lat.values, data.rivid.values) ),
						columns=['longitude', 'latitude', 'rivid'] )

	# check if npl_rivid.npy file exists in the scripts folder
	if ( os.path.isfile('npl_rivid.npy') == False ) :
		print("No river ID information file found")
		data_filtrd = rivid_extractor(dat_pts, mappath)
		# save the array with river ids based on Nepal
		np.save('npl_rivid.npy', data_filtrd.rivid.values)
		print("river ID information saved.")
		if plt_maps == True:
			plotter(data, npl, npl_data = data_filtrd, typ = 'pandas')

	elif ( os.path.isfile('npl_rivid.npy') == True ):
		# print(f"Subsetting the data file {count:d}...\n")
		# load the file with desired river id data
		npl_rivid 	= np.load('npl_rivid.npy')

		# save discharge data corresponding to Nepal only
		data_npl 	= data.sel(rivid = npl_rivid)
		
		if plt_maps == True:
			plotter(data, npl, npl_data = data_npl, typ = 'xarray')

		return data_npl