# Will use in future
# def is_tile_in_province():
    
# geotransform = band_ds.GetGeoTransform()

# minx = geotransform[0]
# miny = geotransform[3] + geotransform[5] * band_ds.RasterYSize
# maxx = geotransform[0] + geotransform[1] * band_ds.RasterXSize
# maxy = geotransform[3]

# tile_bbox = box(minx, miny, maxx, maxy)
# print(f"Tile Bounding Box: {tile_bbox.bounds}")

# for pruid in provinces_grid_cells.keys():
#     province_boundary = canada_boundary[canada_boundary['PRUID'] == pruid]
#     tile_gdf = gpd.GeoDataFrame(index=[0], crs=pyproj.CRS.from_wkt(band_ds.GetProjection()), geometry=[tile_bbox])
#     tile_gdf.to_crs(province_boundary.crs, inplace=True)

#     intersecting_provinces = gpd.overlay(tile_gdf, province_boundary, how='intersection')

#     if not intersecting_provinces.empty:
#         print(f"Tile intersects with {pruid}:")
#         print(intersecting_provinces)