import getpass
import hydra
import os
import cdsapi
import shutil
import asyncio
import itertools
from pathlib import Path
from omegaconf import DictConfig
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from data_sources.nasa_earth_data_api import NasaEarthDataApi
from grid.square_meters_grid import SquareMetersGrid


def download_era5_data(cfg: DictConfig):
    data_output_base_path = Path(cfg.outputs.data_output_base_path)
    data_output_base_path.mkdir(parents=True, exist_ok=True)
    
    era5_client = cdsapi.Client()
    for year in range(cfg.periods.year_start_inclusive, cfg.periods.year_end_inclusive + 1):
        print(f"Year: {year}")
        for month in range(cfg.periods.month_start_inclusive, cfg.periods.month_end_inclusive + 1):
            print(f"Month: {month}")
            for variable in cfg.sources.era5.variables:
                print(f"Variable {variable}...")
                
                era5_request_json = {
                    "product_type": cfg.sources.era5.product_type,
                    "format": "netcdf",
                    "variable": variable,
                    "area": [63, -140, 40, -50],
                    "year": f"{year}",
                    "month": f"{month:02}",
                }
                
                output_path = data_output_base_path / Path(f"{year}") / Path(f"{month}") / Path(f"era5_{cfg.sources.era5.product_type}_{variable}") / Path("data.nc")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                era5_client.retrieve(
                    cfg.sources.era5.dataset,
                    era5_request_json, 
                    output_path.name
                )
                
                shutil.move(output_path.name, output_path)


def download_nasa_earth_data(cfg: DictConfig):
    nasa_earth_data_api = NasaEarthDataApi()
    
    nasa_earthdata_user = os.environ.get("NASA_EARTH_DATA_USER", None)
    if nasa_earthdata_user is None:
        nasa_earthdata_user = getpass.getpass(prompt = 'Enter NASA Earthdata Login Username: ')
          
    nasa_earthdata_password = os.environ.get("NASA_EARTH_DATA_PASSWORD", None)
    if nasa_earthdata_password is None:
        nasa_earthdata_password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ') 

    nasa_earth_data_api.login(username=nasa_earthdata_user, password=nasa_earthdata_password)
    
    nasa_earth_data_api.load_products()
    
    products_layers = cfg.sources.nasa_earth_data.products_layers
    
    select_nasa_earth_data_products_layers(nasa_earth_data_api, products_layers)
    
    print("Loading Canada boundary...")
    canada = CanadaBoundary(CanadaBoundaryDataSource(Path(cfg.boundaries.output_path))) 
    canada.load(provinces=cfg.boundaries.provinces)
    
    grid = SquareMetersGrid(
        pixel_size_in_meters=cfg.grid.pixel_size_in_meters,
        tile_size_in_pixels=cfg.grid.tile_size_in_pixels,    
    )
    print("Tiling boundary...")
    tiles_boundaries = grid.get_tiles_boundaries(canada.boundary)
        
    nasa_earth_data_api.submit_tasks(
        tiles=tiles_boundaries,
        year_start_inclusive=cfg.periods.year_start_inclusive,
        year_end_inclusive=cfg.periods.year_end_inclusive,
        month_start_inclusive=cfg.periods.month_start_inclusive,
        month_end_inclusive=cfg.periods.month_end_inclusive,
        pixel_size_in_meters=cfg.grid.pixel_size_in_meters,
        tile_size_in_pixels=cfg.grid.tile_size_in_pixels,
        logs_folder_path=cfg.logs.nasa_earth_data_logs_folder_path
    )
    
    nasa_earth_data_api.wait_until_tasks_complete()
    
    products_names = [list(k.keys())[0] for k in cfg.sources.nasa_earth_data.products_layers]
    products_layers = list(itertools.chain.from_iterable([list(k.values())[0] for k in cfg.sources.nasa_earth_data.products_layers]))
    
    asyncio.run(nasa_earth_data_api.download_data(
        data_output_base_path=cfg.outputs.data_output_base_path,
        year_start_inclusive=cfg.periods.year_start_inclusive,
        year_end_inclusive=cfg.periods.year_end_inclusive,
        month_start_inclusive=cfg.periods.month_start_inclusive,
        month_end_inclusive=cfg.periods.month_end_inclusive,
        products_names=products_names,
        products_layers=products_layers
        )
    )


def select_nasa_earth_data_products_layers(nasa_earth_data_api: NasaEarthDataApi, products_layers: DictConfig) -> None:
    for product in products_layers:
        product_name_and_version = list(product.keys())[0]
        print(f"Selecting product {product_name_and_version}...")
        nasa_earth_data_api.select_product(product_name_and_version)
        
        product_layers = list(product.values())[0]
        for layer in product_layers:
            print(f"Selecting layer {layer} for product {product_name_and_version}...")
            nasa_earth_data_api.select_layer(product_name_and_version, layer)


@hydra.main(version_base=None, config_path="config", config_name="download_data")
def main(cfg : DictConfig):
    if "era5" in cfg.sources:
        print("Downloading ERA5 data...")
        download_era5_data(cfg)
        print("ERA5 data downloaded!")
    
    if "nasa_earth_data" in cfg.sources:
        print("Downloading NASA Earth Data...")
        download_nasa_earth_data(cfg)
        print("NASA Earth Data downloaded!")

if __name__ == "__main__":
    main()
