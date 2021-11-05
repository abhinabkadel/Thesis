#!/bin/bash
for i in {34..52}
do 
  wsl wget "http://110.34.30.197:8080/thredds/fileServer/ECMWF/20201228.00/Qout_south_asia_geoglowsn_$i.nc"
done