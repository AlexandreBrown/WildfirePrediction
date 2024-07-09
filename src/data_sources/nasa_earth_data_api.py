import requests
import calendar
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from IPython.display import display, HTML


class NasaEarthDataApi:
    def __init__(self):
        self.base_url = 'https://appeears.earthdatacloud.nasa.gov/api/'
        self.selected_products = []
        self.projections = {}
        
    def login(self, username: str, password: str):
  
        login_response = requests.post(f"{self.base_url}login", auth=(username, password))

        if not (login_response.status_code >= 200 and login_response.status_code < 300):
            raise Exception(f"Login failed with status code {login_response.status_code} and message {login_response.reason}")

        token_response = login_response.json()
        self.token = token_response['token']                      
        self.auth_header = {'Authorization': f"Bearer {self.token}"}
        
        print("Login successful!")
    
    def load_products(self) -> dict:
        product_response = requests.get(f"{self.base_url}product").json()                         
        print(f"AppEEARS currently supports {len(product_response)} products.") 
        self.products = {p['ProductAndVersion']: p for p in product_response}
    
    def display_products(self, products: dict):
        flattened_products = []
        for k, v in products.items():
            flattened_product = {'ProductAndVersion': k}
            flattened_product.update(v)
            flattened_products.append(flattened_product)
        
        df = pd.DataFrame(flattened_products)
        display(HTML(df.to_html()))
    
    def display_products_with_description(self, keyword: str):
        products_of_interest = {k: v for k, v in self.products.items() if keyword.lower() in v['Description'].lower()}
        self.display_products(products_of_interest)
    
    def select_product(self, product_and_version: str):
        if product_and_version in [p['ProductAndVersion'] for p in self.selected_products]:
            print(f"Product {product_and_version} already selected, action ignored!")
            return
        
        self.selected_products.append(self.products[product_and_version])
    
    def display_selected_products_layers(self):
        for selected_product in self.selected_products:
            selected_product_layers = requests.get(f"{self.base_url}product/{selected_product['ProductAndVersion']}").json()
            df = pd.DataFrame({
                selected_product['ProductAndVersion']: list(selected_product_layers.keys()),
                'Description': [selected_product_layers[k]['Description'] for k in selected_product_layers.keys()]
                })
            display(HTML(df.to_html())) 
    
    def select_layer(self, product_and_version: str, layer: str):
        for selected_product in self.selected_products:
            if selected_product['ProductAndVersion'] == product_and_version:
                selected_product['layer'] = layer
                return
        raise Exception(f"Product {product_and_version} not found in selected products!")
    
    def get_products_layers(self) -> list:
        products_layers = []
        for selected_product in self.selected_products:
            products_layers.append({
                'product': selected_product['ProductAndVersion'],
                'layer': selected_product['layer'],
            })
        return products_layers
    
    def display_projections(self): 
        self.load_projections()
        df = pd.DataFrame(self.projections)
        display(HTML(df.to_html()))
    
    def load_projections(self) -> dict:
        self.projections = requests.get(f"{self.base_url}spatial/proj").json()

    def submit_tasks(
        self,
        tiles: gpd.GeoDataFrame,
        year_start_inclusive: int,
        year_end_inclusive: int,
        month_start_inclusive: int,
        month_end_inclusive: int,
        tile_resolution_in_meters: int,
        tile_length_in_pixels: int
    ) -> list:
        self.load_projections()
        
        tiles_json = json.loads(tiles.to_json())
        
        tasks_responses = []
  
        for product_layer in self.get_products_layers():
            product, layer = product_layer['product'], product_layer['layer']
            for year in range(year_start_inclusive, year_end_inclusive + 1):
                for month in range(month_start_inclusive, month_end_inclusive + 1):
                    task_response = self.submit_task(
                        year=year,
                        month=month,
                        tile_resolution_in_meters=tile_resolution_in_meters,
                        tile_length_in_pixels=tile_length_in_pixels,
                        product=product,
                        layer=layer,
                        tiles_json=tiles_json
                    )
                    tasks_responses.append(task_response)
                
        return tasks_responses
    
    def submit_task(
        self,
        year: int,
        month: int,
        tile_resolution_in_meters: int,
        tile_length_in_pixels: int,
        product: str,
        layer: str,
        tiles_json: dict
    ):
        task_type = "area" # 'area', 'point'
        task_name = f"canada_{tile_resolution_in_meters}m_{tile_length_in_pixels}px_{product}_{layer}_{year}_{month:02}"  
        start_date_mm_dd_yyyy = f"{month:02}-01-{year}"
        end_day = calendar.monthrange(year=year, month=month)[1]
        end_date_mm_dd_yyyy = f"{month:02}-{end_day:02}-{year}"
        recurring = False
        output_format = 'netcdf4' # 'netcdf4', 'geotiff'
        output_projection = self.projections['geographic']['Name']
        task = {
            "task_type": task_type,
            "task_name": task_name,
            "params": {
                "dates": [
                    {
                        "startDate": start_date_mm_dd_yyyy,
                        "endDate": end_date_mm_dd_yyyy,
                        "recurring": recurring
                    }
                ],
                "layers": [{'product': product, 'layer': layer}],
                'output': {
                    'format': {
                        'type': output_format
                    },
                    'projection': output_projection
                },
                "geo": tiles_json,
            }
        }

        task_response = requests.post(f"{self.base_url}task", json=task, headers=self.auth_header)
        print(f"Status Code: {task_response.status_code} {task_response.reason}")
        task_response_json = task_response.json()
        print(f"Task Response JSON: {task_response_json}")