import geopandas as gpd
from sqlalchemy import create_engine, text
from shapely import wkb

# Database connection URL
db_url = "postgresql://postgres:pgpassword@localhost:5432/cenogis"

def run():
    """
    Connects to the database, searches for any table containing 'pipeline',
    pulls all rows as a GeoDataFrame, and returns it.
    """
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Step 1: Look for tables with 'pipeline' in their names
        table_query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type='BASE TABLE'
        AND table_name ILIKE '%pipeline%';
        """)
        result = conn.execute(table_query)
        tables = [row[0] for row in result.fetchall()]

    if not tables:
        raise ValueError("❌ No tables found containing 'pipeline'.")

    pipelines_table = tables[0]  # Use the first matching table

    # Step 2: Read entire table using GeoPandas
    sql = f"SELECT * FROM {pipelines_table};"
    gdf = gpd.read_postgis(sql, engine, geom_col='geometry')

    # Optional: Ensure CRS
    if gdf.crs is None:
        gdf.set_crs(epsg=4269, inplace=True)

    print(f"✅ Loaded {len(gdf)} rows from '{pipelines_table}' as GeoDataFrame.")
    return gdf
