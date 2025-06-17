import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import pandas as pd

def run_intersection(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    return gpd.overlay(gdf1, gdf2, how="intersection")

def run_touches(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    mask = gdf1.geometry.apply(lambda x: gdf2.geometry.touches(x).any())
    return gdf1[mask]

def run_crosses(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    mask = gdf1.geometry.apply(lambda x: gdf2.geometry.crosses(x).any())
    return gdf1[mask]

def run_within(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    mask = gdf1.geometry.apply(lambda x: gdf2.geometry.within(x).any())
    return gdf1[mask]

def run_closest(gdf1, gdf2, distance):
    
    # Ensure both have same CRS
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    # Buffer gdf1 for spatial range filtering (optional but faster)
    buffered = gdf1.copy()
    buffered["geometry"] = buffered.geometry.buffer(distance)

    # Clip gdf2 for faster processing
    clipped = gpd.clip(gdf2, buffered)

    if clipped.empty:
        print("⚠️ No matches found within the given distance.")
        return gpd.GeoDataFrame(columns=gdf1.columns, crs=gdf1.crs)

    # Perform nearest join
    joined = gpd.sjoin_nearest(gdf1, clipped, how="left", distance_col="min_distance")

    return joined

def run_minimum_distance(gdf1, gdf2):
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    data = []
    for idx1, geom1 in gdf1.geometry.items():
        min_dist = float("inf")
        closest_geom = None
        for idx2, geom2 in gdf2.geometry.items():
            dist = geom1.distance(geom2)
            if dist < min_dist:
                min_dist = dist
                closest_geom = geom2
        data.append({
            "source_index": idx1,
            "target_geom": closest_geom,
            "min_distance": min_dist
        })

    return gpd.GeoDataFrame(data, geometry="target_geom", crs=gdf1.crs)

def run_buffer(gdf, distance):
    """
    Buffers geometries in a projected CRS (EPSG:3400) and returns in geographic CRS (EPSG:4326).
    """
    # Save original CRS
    original_crs = gdf.crs

    # Project to a local CRS (EPSG:3400) for accurate distance buffering
    gdf_proj = gdf.to_crs(epsg=3400)
    gdf_proj['geometry'] = gdf_proj.geometry.buffer(distance)

    # Reproject back to original CRS or EPSG:4326
    return gdf_proj.to_crs(original_crs)
