import numpy as np
import geopandas as gpd
from shapely.geometry import box


class SquareMetersGrid:
    def __init__(self, tile_resolution_in_meters: int, tile_length_in_pixels: int):
        self.tile_resolution_in_meters = tile_resolution_in_meters
        self.tile_length_in_pixels = tile_length_in_pixels
    
    def get_tiles(self, geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        minx, miny, maxx, maxy = geometry.total_bounds

        self.tile_size_in_meters = self.tile_resolution_in_meters *  self.tile_length_in_pixels

        x_coords = np.arange(minx, maxx, self.tile_size_in_meters)
        y_coords = np.arange(miny, maxy, self.tile_size_in_meters)
        grid_cells = [box(x, y, x + self.tile_size_in_meters, y + self.tile_size_in_meters) for x in x_coords for y in y_coords]

        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=geometry.crs)

        tiles = grid[grid.intersects(geometry.union_all())]
        
        return tiles

    def get_tile_size_in_meters(self):
        return self.tile_size_in_meters
    
    def get_tile_size_in_km(self):
        return self.tile_size_in_meters / 1000
