# Wildfire Prediction
Canada Wildfire Prediction Using Deep Learning.  
The objective is to predict the future fire occurrences for the next year given various features from the past year.

# Data
- [Canada Boundary Shapefile](https://open.canada.ca/data/en/dataset/a883eb14-0c0e-45c4-b8c4-b54c4a819edb)  

## Features  
- [MODIS/Terra Vegetation Indices Monthly L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod13a3v061/)
  - Normalized Difference Vegetation Index (NDVI)
  - Enhanced Vegetation Index (EVI)
- [MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mod15a2hv061/)
  - Leaf Area Index (LAI)
- [MODIS/Terra Land Surface Temperature/Emissivity 8-Day L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod11a2v061/)
  - Land Surface Temperature (LST)
- [MODIS/Terra Thermal Anomalies/Fire 8-Day L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod14a2v061/)
  - Fire Mask
- [MODIS/Terra+Aqua Burned Area Monthly L3 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mcd64a1v061/)
  - Burn Area
- [MODIS/Terra Net Evapotranspiration Gap-Filled 8-Day L4 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mod16a2gfv061/)
  - Total Evapotranspiration
- [Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2129)
  - Day Length (dayl)
  - Precipitation (prcp)
  - Maximum air temperature (tmax)
  - Minimum air temperature (tmin)
  - Water vapor pressure (vp)
- [NASADEM Merged DEM Global 1 arc second](https://lpdaac.usgs.gov/products/nasadem_hgtv001/)
  - Elevation
- [ASTER Global Water Bodies Database](https://lpdaac.usgs.gov/products/astwbdv001/)
  - Water Bodies

### Aggregation  
- All features that are not already monthly based are averaged to have a monthly temporal granularity.  

## Target  
- [NBAC Canada Fire Polygons](https://cwfis.cfs.nrcan.gc.ca/datamart)