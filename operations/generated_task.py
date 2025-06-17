import pandas as pd
import numpy as np
import geopandas as gpd

# Geospatial operations
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import unary_union, nearest_points
from shapely.affinity import translate, rotate, scale
from shapely import wkt, wkb

from operations.get_primary_roads_as_gdf import run as get_primary_roads
from operations.get_pipelines_as_gdf import run as get_pipelines
from operations.spatial_func import run_closest

def final_gdf():
    # Step 1: Load GeoDataFrames
    primary_roads_gdf = get_primary_roads()
    pipeline_gdf = get_pipelines()

    # Step 2: Find the closest geometries within a distance of 200m using the utility function
    nearest_pipelines = run_closest(primary_roads_gdf, pipeline_gdf, 200)

    return nearest_pipelines