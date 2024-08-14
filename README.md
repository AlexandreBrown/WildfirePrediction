# Wildfire Prediction
Canada Wildfire Prediction Using Deep Learning.  
The objective is to predict the future fire occurrences for the next years given various inputs from the past years and current years (current years only for data that is not impacted by fires).

# Data  
## Boundaries
- [Canada Boundary Shapefile](https://open.canada.ca/data/en/dataset/a883eb14-0c0e-45c4-b8c4-b54c4a819edb)  

## Data  
The model receives as input the following data (we stack the data from multiple years).  
For data that is not affected by fires (eg: weather data), we use the data for the current year and for data that is affected by fires (eg: vegetation data), we use the data from a previous year(s).  
For instance, if we want to predict the wildfires for the year 2005, then we will use the weather data from 2005 and the vegetation data from 2004 (or 2004 and 2003 if the history window is longer, this is configurable when training).   
### Dynamic  
Dynamic data is data that changes with time. This data is updated on a daily/weekly/bi-weekly/monthly/yearly basis (depending on which data).
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
Static data is the data that does not change over time or for which we only have 1 time slice.
#### Elevation 
- [NASA Shuttle Radar Topography Mission Global 3 arc second](https://lpdaac.usgs.gov/products/srtmgl3v003/)
  - Elevation

### Data Aggregation  
- All the data that is not already yearly based is averaged to have a yearly temporal granularity.  

## Target  
- [NBAC Canada Fire Polygons](https://cwfis.cfs.nrcan.gc.ca/datamart)
- Currently the model is trained to predict the future wildfire occurrences for the next X years.  

# End-To-End Pipeline
## Prerequisites
### Conda/Micromamba
- Download and install conda or micromamba (recommended) : https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- Create a new environment using the `environment.yaml` file from this repository (see [this guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html#conda-yaml-spec-files)).
- Activate your new environment (see [this guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html#quickstarts)).

### NASA Earthdata Account
- Create an account : https://urs.earthdata.nasa.gov/users/new
- This is required to download the NASA Earthdata data.

### CDS Account
- Create an ECMWF Account : https://accounts.ecmwf.int/auth/realms/ecmwf/login-actions/registration?client_id=cms-www&tab_id=vmMaA16DI6A
- Login to CDS and setup your API access following this guide : https://cds-beta.climate.copernicus.eu/how-to-api 
- This is required to download the ERA5 data.
- Accept the terms of use for the ERA5 data (scroll down to terms of use section and click accept) : https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download

## Pipeline Steps
### Download Data
Here is an overview of how the data is download for each data source and the various steps involved (not including resume logic and low level details) :  
<img src="doc/images/download_data_overview.png" height="600px"/>
  
The first step of the pipeline is to download the data that will be used for training the model.  
To do so, one only needs to execute one script, **make sure you are at the root of this repository in your terminal**.  
1. Setup environment variables in a terminal :
   ```bash
   export NASA_EARTH_DATA_USER=<YOUR USERNAME>
   ```
   eg: `export NASA_EARTH_DATA_USER=alexandrebrown`  

   ```bash
   export NASA_EARTH_DATA_PASSWORD="<YOUR_PASSWORD>"
   ```
   eg: `export NASA_EARTH_DATA_PASSWORD="mypassword123"`  

   ```bash
   export PYTHONPATH=$(pwd)/src
   ```
2. Edit the configuration file under `config/download_data.yaml` (or leave as-is) so that you it matches what you want to download (you can also leave everything as-is and only change the year/month range).   

   One thing to note is that the `logs.nasa_earth_data_logs_folder_path` path must be `null` if you do not wish to resume a previous execution of the download script for the NASA Earthdata **or if it is your first time running the script**. If you executed the download_data script but an issue occurred (eg: NASA servers went down), check under the `logs/` folder for the path to the auto-generated log file. You can pass in the path to the log folder to resume from it (eg: `logs/download_data/20240711-081839` will resume using the log file inside `logs/download_data/20240711-081839`).  

   To learn more about NASA Earthdata API and what products and layers mean, see the notebook under `experiments/explore_nasa_earth_data.ipynb`.  

   Each product can have one or more layers, the download_data config allows us to specify which products and which layers from each product to download. The layer name must match the exact layer name (see the exporation notebook for more details on how to get this value).  
3. Run the `download_data` script :  
   ```bash
   python -m download_data
   ```
   Note : If an error occurred during the execution (eg: third party server went down/download failed), you can avoid re-submitting processing requests that were already sent before the crash. To do so, look under `logs/download_data` and copy the path to the folder that was generated and put this path under `logs.nasa_earth_data_logs_folder_path`.  
   For instance, if I have the following : `logs/download_data/20240711-081839` from my previous execution, then I would update the `config/download_data.yaml` to have :  
   ```yaml
    logs:
      nasa_earth_data_logs_folder_path: "logs/download_data/20240711-081839/"
   ```

This will download the data based on your `config/download_data.yaml` configuration.  
The current data sources that are supported are :  
- era5
- nasa_earth_data

### Generate Dataset
Once the raw data has been downloaded, the data needs to be processed because some data might be daily, some might be bi-weekly or monthly and some data sources might return tiles while some might return one file for the entire Canada.  
The goal of this step is to take all the data that was downloaded and produce as output tiles of the desired resolution and aggregated yearly.   
Here is an overview of the various steps (at a high level) :   

<img src="doc/images/generate_dataset_overview.png" height="600px"/>  

Note that this step outputs big tiles and these tiles are usually larger than the tiles that the model will take as input. This was done to ensure that we split our train/val data in a way that avoids leakage. Smaller tiles (eg: 128x128) will be created during training based on the big tiles.  
Each big tile is of dimension C x H x W where = C the number of different sources of data (eg: NDVI, EVI, LAI, ...), H = 512 and W = 512.  
So each big tile represents the data inputs stacked for 1 year for the area delimited by the 512x512 pixels area. 

1. Edit the configuration (or leave as-is) under `config/generate_dataset.yaml`.   
    - One can change the pixel size resolution (eg: 250 meters), tile size in pixels (eg: 512x512).
    - The sources names must match the folder name created during the download step. 
2. Execute the dataset generation script :  
    ```bash
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python -m generate_dataset
    ```

# Contributing  
1. Follow the [prerequisites](#prerequisites) steps.
1. [Download & Install Trufflehog Binary](https://github.com/trufflesecurity/trufflehog/releases/tag/v3.81.8)
1. Install pre-commit hooks  
```bash
pre-commit install --allow-missing-config
```  
