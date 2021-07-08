#%%
import geopandas as gpd

#%%
filepath =r"D:\Masters\Thesis\GIS_data\Nepal-GIS_files\Hydrology_stations-Nepal\hydrology-stations.shp" 
shp_data = gpd.read_file(filepath)
