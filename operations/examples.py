#Import libraries. Make sure to import necessary libraries like pandas as pd, numpy as np, shapely, if required. 

# Data manipulation
import pandas as pd
import numpy as np
import geopandas as gpd

# Geospatial operations
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely.ops import unary_union, nearest_points
from shapely.affinity import translate, rotate, scale
from shapely import wkt, wkb

from operations.get_hydrolines_as_gdf import run as get_hydrolines
from operations.get_pipelines_as_gdf import run as get_pipelines
from operations.spatial_func import run_intersection

def final_gdf():
    # Step 1: Load GeoDataFrames
    hydro_gdf = get_hydrolines()
    pipeline_gdf = get_pipelines()

    # Step 2: Compute intersection using the utility function
    intersected = run_intersection(hydro_gdf, pipeline_gdf)

    return intersected
