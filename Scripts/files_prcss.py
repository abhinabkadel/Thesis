# %%
import os
import Nepal_extractor 
import re

# %%
def savename(filepath):
    fldr_pth = (os.path.join("..", "Fcst_data", 
                            re.split(r"[\\]", filepath)[-2]))
    if os.path.isdir(fldr_pth) == False:
        os.mkdir(fldr_pth)
    sav_pth  = os.path.join( fldr_pth, 
               re.split(r"[\\]", filepath)[-1]) \
                    .replace("south_asia","npl")
    return sav_pth

# %%
dirName     = r"E:\Toolkit\Forecast-downloads\21-03March"
mappath 	= r"..\GIS_data\Nepal-GIS_files\Nepal-boundary\data\Outline.shp"
plt_maps 	= False
filelist = []
for root, dirs, files in os.walk(dirName):
    print("\n" + root)
    fldr_fils = []
    count = 0
    for i in files:
        cntnt = os.path.join(root,i)
        if cntnt.endswith('.nc') == True:
            data_npl = Nepal_extractor.main(cntnt, mappath, plt_maps, count)
            
            sav_pth = savename(cntnt)
            data_npl.to_netcdf(path = sav_pth)

            fldr_fils.append(cntnt)
            count = count + 1

    # check if list is empty    
    if not fldr_fils:
        print('no netcdf files in the folder. \n')