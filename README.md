# Wildfire Prediction
Canada Wildfire Prediction Using Deep Learning.  
The objective is to predict the future fire occurrences for the next years given various features from the past years.

# Data  
## Boundaries
- [Canada Boundary Shapefile](https://open.canada.ca/data/en/dataset/a883eb14-0c0e-45c4-b8c4-b54c4a819edb)  

## Features  
The model receives as input the following features (we stack the features for the 5 years prior).  
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
- [MODIS/Terra Leaf Area Index/FPAR 8-Day L4 Global 500 m SIN Grid](https://lpdaac.usgs.gov/products/mod15a2hv061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Leaf Area Index (LAI) 

#### Weather
- [ERA5 Reanalysis 100m U component of wind](https://codes.ecmwf.int/grib/param-db/228246)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - 100m U component of wind
- [ERA5 Reanalysis 100m V component of wind](https://codes.ecmwf.int/grib/param-db/132)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - 100m V component of wind
- [ERA5 Reanalysis 2m Temperature](https://codes.ecmwf.int/grib/param-db/167)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - 2m Temperature
- [ERA5 Reanalysis Potential evaporation](https://codes.ecmwf.int/grib/param-db/228251)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - Potential evaporation
- [ERA5 Reanalysis Surface net solar radiation](https://codes.ecmwf.int/grib/param-db/176)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - Surface net solar radiation
- [ERA5 Reanalysis Surface runoff](https://codes.ecmwf.int/grib/param-db/8)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - Surface runoff
- [ERA5 Reanalysis Total precipitation](https://codes.ecmwf.int/grib/param-db/228)
  - Temporal Extent : 1940-01-01 to Present
  - Features : 
    - Total precipitation

#### Thematic
- [MODIS/Terra Land Water Mask Derived from MODIS and SRTM L3 Yearly Global 250 m SIN Grid](https://lpdaac.usgs.gov/products/mod44wv061/)
  - Temporal Extent : 2000-01-01 to Present
  - Features : 
    - Water Mask
- [MODIS/Terra Thermal Anomalies/Fire 8-Day L3 Global 1 km SIN Grid](https://lpdaac.usgs.gov/products/mod14a2v061/)
  - Temporal Extent : 2000-02-18 to Present
  - Features : 
    - Fire Mask  

### Static  
#### Elevation 
- [NASADEM Merged DEM Global 1 arc second](https://lpdaac.usgs.gov/products/nasadem_hgtv001/)
  - Elevation

### Aggregation  
- All features that are not already yearly based are averaged to have a yearly temporal granularity.  

## Target  
- [NBAC Canada Fire Polygons](https://cwfis.cfs.nrcan.gc.ca/datamart)
- Currently the model is trained to predict the future wildfire occurrences for the next 5 years.  

# Contributing

1. Create conda environment using `environment.yaml`
2. Install pre-commit hooks  
```bash
pre-commit install --allow-missing-config
```  
