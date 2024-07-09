import getpass
import hydra
from pathlib import Path
from omegaconf import DictConfig
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from data_sources.nasa_earth_data_api import NasaEarthDataApi
from grid.square_meters_grid import SquareMetersGrid


def select_products_layers(nasa_earth_data_api: NasaEarthDataApi, products_layers: DictConfig) -> None:
    for product in products_layers:
        product_name_and_version = list(product.keys())[0]
        print(f"Selecting product {product_name_and_version}...")
        nasa_earth_data_api.select_product(product_name_and_version)
        
        product_layers = list(product.values())[0]
        for layer in product_layers:
            print(f"Selecting layer {layer} for product {product_name_and_version}...")
            nasa_earth_data_api.select_layer(product_name_and_version, layer)


@hydra.main(version_base=None, config_path="config", config_name="data_download")
def main(cfg : DictConfig) -> None:
    nasa_earth_data_api = NasaEarthDataApi()
    nasa_earthdata_user = getpass.getpass(prompt = 'Enter NASA Earthdata Login Username: ')      
    nasa_earthdata_password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ') 

    nasa_earth_data_api.login(username=nasa_earthdata_user, password=nasa_earthdata_password)
    
    nasa_earth_data_api.load_products()
    
    products_layers = cfg.sources.nasa_earth_data.products_layers
    
    select_products_layers(nasa_earth_data_api, products_layers)
    
    canada = CanadaBoundary(CanadaBoundaryDataSource(Path(cfg.boundaries.output_path))) 
    canada.load(exclude_area_above_60_degree=cfg.boundaries.exclude_area_above_60_degree)
    
    grid = SquareMetersGrid(
        tile_resolution_in_meters=cfg.grid.tile_resolution_in_meters,
        tile_length_in_pixels=cfg.grid.tile_length_in_pixels,    
    )
    tiles = grid.get_tiles(canada.boundary)
    
    nasa_earth_data_api.submit_tasks(
        tiles=tiles,
        year_start_inclusive=cfg.periods.year_start_inclusive,
        year_end_inclusive=cfg.periods.year_end_inclusive,
        month_start_inclusive=cfg.periods.month_start_inclusive,
        month_end_inclusive=cfg.periods.month_end_inclusive,
        tile_resolution_in_meters=cfg.grid.tile_resolution_in_meters,
        tile_length_in_pixels=cfg.grid.tile_length_in_pixels
    )

if __name__ == "__main__":
    main()
