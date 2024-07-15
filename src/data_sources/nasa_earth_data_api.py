import json
import hashlib
import requests
import logging
import calendar
import pandas as pd
import geopandas as gpd
import time
import aiohttp
import asyncio
from pathlib import Path
from IPython.display import display, HTML
from typing import Optional
from datetime import datetime
from omegaconf import DictConfig


logging.basicConfig(level=logging.INFO)

class NasaEarthDataApi:
    def __init__(self):
        self.base_url = 'https://appeears.earthdatacloud.nasa.gov/api/'
        self.selected_products = []
        self.projections = {}
        self.earth_data_tasks_info_file_name = "earth_data_tasks_info.json"
        
    def login(self, username: str, password: str):
  
        login_response = requests.post(f"{self.base_url}login", auth=(username, password))

        if not (login_response.status_code >= 200 and login_response.status_code < 300):
            raise Exception(f"Login failed with status code {login_response.status_code} and message {login_response.reason}")

        token_response = login_response.json()
        self.token = token_response['token']                      
        self.auth_header = {'Authorization': f"Bearer {self.token}"}
        
        logging.info("Login successful!")
    
    def load_products(self) -> dict:
        product_response = requests.get(f"{self.base_url}product").json()                         
        logging.info(f"AppEEARS currently supports {len(product_response)} products.") 
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
            logging.info(f"Product {product_and_version} already selected, action ignored!")
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
                if 'layer' not in selected_product.keys():
                    selected_product['layer'] = [layer]
                else:
                    selected_product['layer'] = selected_product['layer'] + [layer]
                return
        raise Exception(f"Product {product_and_version} not found in selected products!")
    
    def get_products_layers(self) -> list:
        products_layers = []
        for selected_product in self.selected_products:
            for selected_layer in selected_product['layer']:
                products_layers.append({
                    'product': selected_product['ProductAndVersion'],
                    'layer': selected_layer,
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
        pixel_size_in_meters: int,
        tile_size_in_pixels: int,
        logs_folder_path: Optional[str] = None
    ):  
        tiles = tiles.to_crs(epsg=4326)
        tiles_json = json.loads(tiles.to_json())
        
        if logs_folder_path is not None:
            logging.info(f"Resuming tasks from log folder {str(logs_folder_path)}...")
            self.logs_folder_path = Path(logs_folder_path)
            with open(self.logs_folder_path / self.earth_data_tasks_info_file_name, 'r') as f:
                self.tasks_info = json.load(f)
            
            tasks_hash = {t['task_hash']: True for t in self.tasks_info}
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.logs_folder_path = Path(f"logs/download_data/{timestamp}")
            self.logs_folder_path.mkdir(parents=True, exist_ok=True)
            tasks_hash = {}
            self.tasks_info = []
        
        logging.info("Submitting tasks...")
        for year in range(year_start_inclusive, year_end_inclusive + 1):
            logging.info(f"Year: {year}")
            for month in range(month_start_inclusive, month_end_inclusive + 1):
                logging.info(f"Month: {month}")
                for product_layer in self.get_products_layers():
                    product, layer = product_layer['product'], product_layer['layer']
                    logging.info(f"Product: {product} | Layer: {layer}")
                    
                    self.wait_until_tasks_limit_not_reached()

                    task = self.create_task(year, month, pixel_size_in_meters, tile_size_in_pixels, product, layer, tiles_json)
                    
                    if self.products[product]['TemporalGranularity'] == 'Static':
                        task['params']['dates'] = [{
                            'startDate': self.format_date(self.products[product]['TemporalExtentStart']),
                            'endDate': self.format_date(self.products[product]['TemporalExtentEnd'])
                        }]
                    
                    task_hash = self.hash_task(task)
                    logging.info(f"Task Hash: {task_hash}")
                    
                    if tasks_hash.get(task_hash) is not None:
                        logging.info(f"Task {task_hash} already submitted, action ignored!")
                        continue
                    
                    task_id = self.submit_task(task)
                    
                    tasks_hash[task_hash] = True
                    task_info = {
                        'task_id': task_id,
                        'task_hash': task_hash,
                        'product': product,
                        'layer': layer,
                        'year': year,
                        'month': month,
                    }
                    self.tasks_info.append(task_info)
                    
                    self.save_tasks_info()
    
    def format_date(self, date_str):
        parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
        formatted_date = parsed_date.strftime('%m-%d-%Y')
        return formatted_date
    
    def wait_until_tasks_limit_not_reached(self):
        tasks_limit_reached = True
        while tasks_limit_reached:
            logging.info("Waiting for tasks to complete...")
            tasks_response = requests.get(f"{self.base_url}task", headers=self.auth_header).json()
            nb_tasks_not_done = len([t for t in tasks_response if t['status'] != 'done'])
            tasks_limit_reached = nb_tasks_not_done >= 1000
            if tasks_limit_reached:
                time.sleep(60)
    
    def create_task(
        self,
        year: int,
        month: int,
        pixel_size_in_meters: int,
        tile_size_in_pixels: int,
        product: str,
        layer: str,
        tiles_json: dict
    ) -> dict:
        task_type = "area" # 'area', 'point'
        base_task_name = f"canada_{pixel_size_in_meters}m_{tile_size_in_pixels}px_{product}_{layer}".replace(".", "_").replace(" ", "_").replace("__","_")
        if self.products[product]['TemporalGranularity'] == 'Static':
            task_name = base_task_name
        else:
            task_name = f"{base_task_name}_{year}_{month:02}" 
        
        start_date_mm_dd_yyyy = f"{month:02}-01-{year}"
        end_day = calendar.monthrange(year=year, month=month)[1]
        end_date_mm_dd_yyyy = f"{month:02}-{end_day:02}-{year}"
        recurring = False
        output_format = 'netcdf4' # 'netcdf4', 'geotiff'
        output_projection = 'geographic'
        return {
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
    
    def hash_task(self, task: dict) -> str:
        return hashlib.md5(json.dumps(task).encode()).hexdigest()
    
    def submit_task(
        self,
        task: dict
    ) -> str:
        task_response = requests.post(f"{self.base_url}task", json=task, headers=self.auth_header)
        logging.info(f"Status Code: {task_response.status_code} {task_response.reason} {task_response.text}")
        task_id = task_response.json()['task_id']
        return task_id
        
    def save_tasks_info(self):
        with open(self.logs_folder_path / self.earth_data_tasks_info_file_name, 'w') as f:
            json.dump(self.tasks_info, f)
    
    def delete_tasks(self, tasks_ids: list):
        for task_id in tasks_ids:
            task_response = requests.delete(f"{self.base_url}task/{task_id}", headers=self.auth_header)
            logging.info(f"Status Code: {task_response.status_code} {task_response.reason} {task_response.text}")

    def wait_until_tasks_complete(self):
        tasks_ids = set([t['task_id'] for t in self.tasks_info])
        
        tasks_done = False
        while not tasks_done:
            logging.info("Waiting for tasks to complete...")
            tasks_response = requests.get(f"{self.base_url}task", headers=self.auth_header).json()
            tasks_response = [t for t in tasks_response if t['task_id'] in tasks_ids]
            nb_tasks_done = len([t for t in tasks_response if t['status'] == 'done'])
            tasks_done = nb_tasks_done == len(tasks_ids)
            if not tasks_done:
                time.sleep(600)

    async def download_data(
        self, 
        data_output_base_path: str,
        year_start_inclusive: int,
        year_end_inclusive: int,
        month_start_inclusive: int,
        month_end_inclusive: int,
        products_names: list,
        products_layers: list
    ):
        logging.info("Downloading data...")
        data_output_base_path = Path(data_output_base_path)
        
        tasks_info_of_interest = []
        for task_info in self.tasks_info:
            if task_info['year'] >= year_start_inclusive and \
                task_info['year'] <= year_end_inclusive and \
                    task_info['month'] >= month_start_inclusive and \
                        task_info['month'] <= month_end_inclusive and \
                            task_info['product'] in products_names and \
                                task_info['layer'] in products_layers:
                    tasks_info_of_interest.append(task_info)
 
        self.tasks_info = tasks_info_of_interest
 
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            for task_info in self.tasks_info:
                await self.download_task_files(task_info, data_output_base_path, session)
                logging.info(f"Task {task_info['task_id']} {task_info['product']} {task_info['layer']} {task_info['year']} {task_info['month']} downloaded!")

    async def download_task_files(self, task_info: dict, data_output_base_path: Path, session: aiohttp.ClientSession):
        task_id = task_info['task_id']
        product = task_info['product'].replace(".", "_").replace(" ", "_").replace("__", "_")
        layer = task_info['layer'].replace(".", "_").replace(" ", "_").replace("__", "_")
        year = task_info['year']
        month = task_info['month']
        
        if self.products[task_info['product']]['TemporalGranularity'] == 'Static':
            output_path = data_output_base_path / "static_data" / f"{product}_{layer}"
        else:
            output_path = data_output_base_path / f"{year}" / f"{month}" / f"{product}_{layer}"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        bundle = await session.get(f'{self.base_url}bundle/{task_id}', headers=self.auth_header)
        bundle = await bundle.json()
        
        files = {f['file_id']: f['file_name'] for f in bundle['files']}

        asyncio_semaphore = asyncio.Semaphore(64)
        tasks = [asyncio.create_task(self.download_file(asyncio_semaphore, session, task_id, file_id, file_name, output_path)) for file_id, file_name in files.items()]
        await asyncio.gather(*tasks)
    
    async def download_file(self, asyncio_semaphore, session, task_id: str, file_id, file_name: str, output_path: Path):
        async with asyncio_semaphore:
            async with session.get(f"{self.base_url}bundle/{task_id}/{file_id}", headers=self.auth_header) as dl:
                dl.raise_for_status()
                if file_name.endswith('.tif'):
                    filename = file_name.split('/')[1]
                else:
                    filename = file_name

                filepath = output_path / Path(filename)
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = await dl.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)