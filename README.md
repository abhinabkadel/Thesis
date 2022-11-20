Scripts and datasets for my thesis work on "Applications of streamflow forecasts for run-of-river hydroelectric systems in Nepal"

## Background:

Most developing countries lack the resources required to develop and maintain their own nationwide fine-resolution streamflow forecasting systems, which has led to the emergence of global streamflow forecasting methods with coarser resolution. The GEOGloWS-ECMWF Streamflow Service (GESS) is a global forecasting model that uses runoff output from the European Center for Medium-range Weather Forecasts (ECMWF) and uses the vector-based routing tool called RAPID to produce 52-member ensemble forecasts 15 days into the future at smaller catchments worldwide. This can be a valuable tool for developing countries to prepare for disasters such as devastating floods or for efficient hydropower operation. 

### Project Aim
The project explores the quality of the GESS forecasts for hydropower operation in Nepal. Nepal does not have a nationwide streamflow forecasting system and its complex terrain is a forecasting challenge for hydrometeorological forecasts. 

The focus is on two unique operating constraints faced by hydropower operation. Bidding in the day-ahead market and to prepare for devastating floods. 

The work involves following tasks:
    - Data filtering and cleaning
    - Verification against observations
    - Statistical Bias correction of raw forecasts
    - Generation of flood/energy generation forecasts

## Where to start?:

./6_Presentations contains my final defense presentation as a pdf, which would be a first place of reference to learn about my work and limitations. 

./7_Writings contains my complete thesis, which has the details. 

./3_Scripts has the python scripts used in the analysis. Each python script can be run on its own using an IDE of your choice. 
Most of the scripts include jupyter notebook like cells, which can be run independently. 

./1_Data does not contain all the data due to their size. Contact me if you wish to play around. 

./5_Images contains results from my GIS analysis. I have not shared the entire geodatabase due to its size. 

## The main scripts:
All the scripts are in the ./3_Scripts/ folder
1. 2_raw_analysis.py verifies the raw GESS forecasts and the relevant plots used to visualise the results.
2. 3_upstream_analysis.py is used to analyse the forecast performance upstream of the Marsyangdi catchment to identify plausible weaknesses in the GESS forecast chain.
3. 4_bias_crrction_variations.py implements different variations of the Degree of Mass Balance (DMB), a statistical bias correction method that has been successfully used by the WFRT. 
4. 5_bias_crrction_calibration.py used to identify the desired window length for the DMB bias corrector. 
5. 6_verification.py uses the optimum bias correction configuration from files 3. and 4., and tests it on a year long of archived forecast data. 
6. 7_end_user_energy.py produces energy generation and revenue forecasts from the streamflow forecasts. Also includes comparison of the forecast performance vs persistance and hindsight (perfect information).
7. 8_prb_frcsts.py contains my attempt to produce probabilistic forecasts based on Bourdin, 2014 and Kernel density estimation. 
8. 9_flood_frcst.py calculates the Q5 flood threshold value, and then produces binary flood forecasts. Calculates the flood forecasting system accuracy and reliability using hit rates and false alarm ratios.
9. calc_funcs.py contains the most commonly used functions for creating dataframes, bias correction or performing verification.
10. drive_dwnld(-mac).py downloads the forecast archive from a google cloud server, and extracts the points that fall within Nepal.
11. fcst-dwnld.py downloads files hosted in a local cloud server that was available through ICIMOD.
12. files_prcss.py renames files for further processing by the other scripts. 
13. Frcst-downloader.sh was used to download individual files that might have been corrupted due to incomplete downloads from other scripts
14. Nepal_extractor.py extracts the forecast points that are in Nepal
15. npl_rivid.npy contains all the riverIDs that fall within Nepal, which was saved during first run of Nepal_extractor.py. This speeds up process when used in later iterations. 
16. plt_funcs.py contains most of the plotly plots used in this thesis project such as: historgram, scatter, box and whisker, line, time-series. 
17. runoff_dwnld.py downloads the runoff forecasts, which is used to justify  the streamflow forecast performance. 


## Softwares Used:
Python                    3.9.9       
ArcGIS Pro                2.9.5
### Python packages used:
contextily                1.2.0       
geopandas                 0.10.2      
h5netcdf                  0.13.0      
hydroerr                  1.24        
hydrostats                0.78        
jupyter_client            7.1.1       
libnetcdf                 4.8.1       
netcdf4                   1.5.8       
pandas                    1.3.5       
plotly                    5.5.0       
scikit-learn              1.0.2       
scipy                     1.7.3       
shapely                   1.8.0       
xarray                    0.20.2      
xskillscore               0.0.24      

## Directory Structure:
abhinabkadel/Thesis
|
|   .gitignore
|   environment.yml
|   README.md
|   Thesis.code-workspace
|
├───.vscode
|
├───1_Data
│   ├───DHM_data
│   ├───Fcst_data
│   ├───GIS_data
│   ├───Hydro_Projects_data
│   ├───IMERG_data
│   ├───obs_data
│   │   └───Required data
│   └───runoff_forecasts
|
├───2_Literature
│   ├───Articles
│   │   ├───30-day-forecasts
│   │   ├───DHM
│   │   ├───Forecast-Value
│   │   ├───From-Werner
│   │   ├───Hydrometeorology-Nepal
│   │   ├───Hydropower-Nepal
│   │   ├───Nepal
│   │   ├───Probabilistic Forecasts
│   │   ├───Streamflow modelling
│   │   └───WFRT-authored
|   |
│   ├───Books
│   │   ├───2019-Handbook_Hydro_Ens-Chapters
│   │   ├───Economic Value of Climate and Weather Forecasts (Book)
│   │   ├───Fleming-Where the River flows
│   │   ├───Stull-2017-Practical_Meteorology
│   │   ├───Warner-2011-NWP
│   │   └───Wilks-Statistical_Methods
|   |
│   ├───Python
|   |
│   └───Reports
|
├───3_Scripts
│   │   2_raw_analysis.py
│   │   3_upstream_analysis.py
│   │   4_bias_crrction_variations.py
│   │   5_bias_crrction_calibration.py
│   │   6_verification.py
│   │   7_end_user_energy.py
│   │   8_prb_frcsts.py
│   │   9_flood_frcst.py
│   │   calc_funcs.py
│   │   drive_dwnld-mac.py
│   │   drive_dwnld.py
│   │   fcst-dwnld.py
│   │   files_prcss.py
│   │   Frcst-downloader.sh
│   │   Nepal_extractor.py
│   │   npl_rivid.npy
│   │   plt_funcs.py
│   │   runoff_dwnld.py
│   │
│   ├───iframe_figures
│   │
│   ├───pickle_dfs
│   │       55535_runoff.pkl
│   │       Balephi.pkl
│   │       Balephi_runoff.pkl
│   │       Balephi_wt.pkl
│   │       Marsyangdi.pkl
│   │       Marsyangdi_runoff.pkl
│   │       Marsyangdi_wt.pkl
│   │       Naugadh.pkl
│   │       Naugadh_runoff.pkl
│   │       Naugadh_wt.pkl
│   │       Trishuli.pkl
│   │       Trishuli_runoff.pkl
│   │       Trishuli_wt.pkl
│   │       Tumlingtar.pkl
│   │       Tumlingtar_runoff.pkl
│   │       Tumlingtar_wt.pkl
│   │       wts_55534.csv
│   │       wts_55534.xlsx
│   │       wts_55535.csv
│   │       wts_55535.xlsx
│   │       wts_55536.csv
│   │       wts_55536.xlsx
│   │       wts_55543.csv
│   │       wts_55543.xlsx
│   │
│   ├───Sites_info
│   │       sites_tbl.csv
│   │       sites_tbl.pkl
│   │       sites_tbl.xlsx
│   │
│   └───__pycache__
│           calc_funcs.cpython-39.pyc
│           Nepal_extractor.cpython-39.pyc
│           plt_funcs.cpython-39.pyc
│   
├───5_Images
|
├───6_Presentations
│   └───Images
|
└───7_Writings

## Author:
Abhinab Kadel
<abhikad@mail.ubc.ca> 

*Last updated: 12-11-2022*