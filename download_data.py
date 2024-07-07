import getpass
import hydra
from omegaconf import DictConfig
from data_sources.nasa_earth_data_api import NasaEarthDataApi

@hydra.main(version_base=None, config_path="config", config_name="data_download")
def main(cfg : DictConfig) -> None:
    nasa_earth_data_api = NasaEarthDataApi()
    nasa_earthdata_user = getpass.getpass(prompt = 'Enter NASA Earthdata Login Username: ')      
    nasa_earthdata_password = getpass.getpass(prompt = 'Enter NASA Earthdata Login Password: ') 

    nasa_earth_data_api.login(username=nasa_earthdata_user, password=nasa_earthdata_password)
    del nasa_earthdata_user, nasa_earthdata_password
    
    nasa_earth_data_api.load_products()
    
    products_layers = cfg.sources.nasa_earth_data.products_layers
    
    for product in products_layers:
        product_name_and_version = list(product.keys())[0]
        print(f"Selecting product {product_name_and_version}...")
        nasa_earth_data_api.select_product(product_name_and_version)
        
        product_layers = list(product.values())[0]
        for layer in product_layers:
            print(f"Selecting layer {layer} for product {product_name_and_version}...")
            nasa_earth_data_api.select_layer(product_name_and_version, layer)

if __name__ == "__main__":
    main()
