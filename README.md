# Wildfire Prediction
Canada Wildfire Prediction Using Deep Learning.  
The objective is to predict the future fire occurrences for the next years given various features from the past years.

# Data  
## Boundaries
- [Canada Boundary Shapefile](https://open.canada.ca/data/en/dataset/a883eb14-0c0e-45c4-b8c4-b54c4a819edb)  

## Features  
The model receives as input the following features (we stack the features for the 3 years prior to the target's first year).  
### Dynamic
#### Vegetation
- [MODIS/Terra Vegetation Indices 16-Day L3 Global 250 m SIN Grid](https://lpdaac.usgs.gov/products/mod13q1v061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Normalized Difference Vegetation Index (NDVI)
    - Enhanced Vegetation Index (EVI)
- [MODIS/Terra Vegetation Continuous Fields Yearly L3 Global 250 m SIN Grid](https://lpdaac.usgs.gov/products/mod44bv061/)
  - Temporal Extent : 2000-03-05 to Present  
  - Features : 
    - Percent Tree Cover  
    - Percent Non-Tree Cover
    - Percent Non Vegetated
- [MODIS/Terra Land Water Mask Derived from MODIS and SRTM L3 Yearly Global 250 m SIN Grid](https://lpdaac.usgs.gov/products/mod44wv061/)
  - Temporal Extent : 2000-01-01 to Present
  - Features : 
    - Water Mask
- [MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mod15a2hv061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Leaf Area Index (LAI)
- [MODIS/Terra Land Surface Temperature/Emissivity 8-Day L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod11a2v061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Day Land Surface Temperature
    - Night Land Surface Temperature 
- [MODIS/Terra Thermal Anomalies/Fire 8-Day L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod14a2v061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Fire Mask
- [MODIS/Terra+Aqua Burned Area Monthly L3 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mcd64a1v061/)
  - Temporal Extent : 2000-11-01 to Present
  - Features :
    - Burn Area
#### Weather
- [MODIS/Terra Net Evapotranspiration Gap-Filled Yearly L4 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mod16a3gfv061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Total Evapotranspiration
- [Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2129)
  - Temporal Extent : 1950-01-01 to Present  
  - Features : 
    - Day Length (dayl)
    - Precipitation (prcp)
    - Maximum air temperature (tmax)
    - Minimum air temperature (tmin)
    - Water vapor pressure (vp)  

### Static  
#### Elevation 
- [NASADEM Merged DEM Global 1 arc second](https://lpdaac.usgs.gov/products/nasadem_hgtv001/)
  - Elevation

### Aggregation  
- All features that are not already yearly based are averaged to have a yearly temporal granularity.  

## Target  
- [NBAC Canada Fire Polygons](https://cwfis.cfs.nrcan.gc.ca/datamart)