# File: app2.py
import pydeck as pdk
import streamlit as st
import re
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from sqlalchemy import create_engine, inspect, text
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
import atexit
from shapely import wkb
from shapely.wkt import loads
import json
from query_check import analyze_query
from clarify_agent import clarify_intent_with_user
from operations.solution_graph_agent import generate_knowledge_graph_from_query
from operations.execute import generate_and_execute_code

example_queries = """
        -- Example 1: Find Wells That Are Inside Township Boundaries
        SELECT 
            w.*,
            t.*
        FROM wells w
        JOIN township t 
        ON ST_Within(w.geometry, t.geometry);

        -- Example 2: Show the Hydrolines That Cross Townships.
        SELECT 
            h.*,
            t.*
        FROM hydrolines h
        JOIN township t 
        ON ST_Intersects(h.geometry, t.geometry);

        -- Example: Calculate the shortest distance between each well and hydroline.
        SELECT 
            w.*, 
            h.*, 
            ST_Distance(w.geometry, h.geometry) AS distance_meters
        FROM wells w
        JOIN LATERAL (
            SELECT h.*, h.geometry AS hydroline_geometry  -- Rename geometry column
            FROM hydrolines h
            ORDER BY w.geometry <-> h.geometry
            LIMIT 1
        ) h ON true;
         """


# Streamlit Page Configuration
st.set_page_config(page_title="Ceno-GIS", page_icon="🗃️", layout="wide")
def extract_schema(db_url):
    """Extract schema from the PostgreSQL database without accessing data."""
    engine = create_engine(db_url)
    inspector = inspect(engine)

    schema_info = []
    
    for table_name in inspector.get_table_names():
        schema_info.append(f"Table: {table_name}")

        # Extract columns and types
        columns = inspector.get_columns(table_name)
        for column in columns:
            schema_info.append(f"  - {column['name']} ({column['type']})")

        # Extract foreign keys (if any)
        foreign_keys = inspector.get_foreign_keys(table_name)
        if foreign_keys:
            for fk in foreign_keys:
                schema_info.append(f"  - Foreign Key: {fk['constrained_columns']} → {fk['referred_table']}({fk['referred_columns']})")

        # Extract indexes
        indexes = inspector.get_indexes(table_name)
        if indexes:
            for idx in indexes:
                schema_info.append(f"  - Index: {idx['name']} (Columns: {', '.join(idx['column_names'])})")
    
    return "\n".join(schema_info)

def setup_llm_chain(schema, chat_history=None, last_error=None):
    """Set up the LangChain components using Ollama with conversational memory."""
    llm = Ollama(model="qwen2.5-coder:7b")  # Uses Ollama for local inference

    # ✅ Format chat history into a readable format for LLM
    chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) if chat_history else "No prior conversation."

    # ✅ Include error feedback if any
    error_feedback = f"\n🚨 Previous SQL error detected: {last_error}" if last_error else ""

    prompt_template = f"""
    You are an AI assistant that answers questions conversationally and in a funny manner.
    Apart from general answers, you are also an expert in writing spatial SQL queries. 

    VERY IMPORTANT QUERY RULES to STRICTLY FOLLOW:
    - ALWAYS use `SELECT *` instead of selecting specific columns.
    - If performing a `JOIN`, ensure that all columns from both tables are included using `table_name.*`.
    - When performing spatial operations like `ST_Intersects`, `ST_DWithin`, or `ST_Buffer`, apply them without omitting columns.
    - If filtering based on conditions, retain all columns (`*`) in the `SELECT` statement.

    **Refer to the previous conversation:**
    {chat_history_text}

    **Refer to errors encountered in SQL query generation:**
    {error_feedback}

    You have access to the following **database schema** which has geometry columns:
    {schema}

    """

    prompt = PromptTemplate(
        input_variables=["schema", "example_queries", "chat_history", "error_feedback"], 
        template=prompt_template,
    )

    return LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True))




def generate_sql_query(chain, schema, question, example_queries):
    """Generate SQL query based on the schema and user question."""
    return chain.run(schema=schema,question=question, example_queries=example_queries)

# -------------------- SQL Query Execution --------------------
def clean_sql_query(sql_response):
    """Extracts and cleans the SQL query from the response by removing backticks."""
    sql_pattern = r"```sql\n(.*?)\n```"
    match = re.search(sql_pattern, sql_response, re.DOTALL)
    
    if match:
        return match.group(1).strip()  # Extract SQL part
    return sql_response.strip()  # Return the response if no backticks found
def validate_sql_query(sql_query, schema, last_error=None):
    """Check if the generated query only uses valid tables & columns from the schema."""
    
    schema_tables = [line.split(":")[1].strip() for line in schema.split("\n") if "Table:" in line]
    
    for table in schema_tables:
        if table.lower() in sql_query.lower():
            return True  # Query is valid

    error_msg = "🚨 Error: The generated query uses unknown tables or columns!"
    
    # Log error for debugging
    st.error(error_msg)

    # Pass the error message to LLM for correction
    return error_msg  # Returning the error instead of False


def execute_sql_query(db_url, sql_query):
    """Executes the SQL query and returns a GeoDataFrame."""
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        query = text(sql_query)
        result = conn.execute(query)
        
        # Extract column names
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in result]

    # Convert to DataFrame
    gdf = gpd.GeoDataFrame(data)

    # ✅ Detect and set the geometry column
    geometry_column = None
    for col in gdf.columns:
        if "geometry" in col.lower():
            geometry_column = col
            break

    if geometry_column:
        try:
            # ✅ Convert WKB geometry to Shapely objects
            gdf[geometry_column] = gdf[geometry_column].apply(lambda x: wkb.loads(x, hex=True) if x else None)

            # ✅ Set geometry and CRS
            gdf.set_geometry(geometry_column, inplace=True)
            gdf.set_crs(epsg=4326, inplace=True)  # Assuming WGS 84 CRS
        except Exception as e:
            st.error(f"⚠️ Error processing geometry: {str(e)}")
    else:
        st.warning("⚠️ No geometry column found in query result.")

    return gdf



def visualize_geodata(gdf):
    """Visualize geospatial data on a Folium map in Streamlit, detecting geometry column dynamically."""

    if gdf.empty:
        st.warning("⚠️ No spatial data found for this query.")
        return

    # ✅ Detect and set the geometry column dynamically
    geometry_column = None
    for col in gdf.columns:
        if "geometry" in col.lower():  # Look for columns containing "geometry"
            geometry_column = col
            break

    if not geometry_column:
        st.error("❌ No geometry column found in the GeoDataFrame.")
        return

    # ✅ Convert WKT strings to Shapely geometries before setting geometry
    gdf[geometry_column] = gdf[geometry_column].apply(lambda x: loads(x) if isinstance(x, str) else x)

    # ✅ Now set the detected geometry column
    gdf.set_geometry(geometry_column, inplace=True)

    # ✅ Drop missing geometries
    gdf = gdf.dropna(subset=[geometry_column])

    # ✅ Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs(epsg=4269, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4269)

    # ✅ Convert **Timestamp columns** to strings before JSON conversion
    for col in gdf.select_dtypes(include=["datetime64[ns]"]).columns:
        gdf[col] = gdf[col].astype(str)

    # ✅ Convert to GeoJSON (fixed serialization error)
    geojson_data = json.loads(gdf.to_json())

    # ✅ Calculate map center
    valid_geometries = gdf[geometry_column].dropna()
    if valid_geometries.empty:
        st.error("❌ No valid geometries to display.")
        return

    center_lat = valid_geometries.centroid.y.mean()
    center_lon = valid_geometries.centroid.x.mean()

    # ✅ Create a Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # ✅ Add geometries to the map
    for _, row in gdf.iterrows():
        geom = row[geometry_column]
        popup_info = row.get("nid", "No ID")  # Customize popup

        if geom.is_empty:
            continue

        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            folium.GeoJson(geom, popup=popup_info, style_function=lambda x: {"color": "blue", "weight": 2}).add_to(m)
        elif geom.geom_type in ["LineString", "MultiLineString"]:
            folium.GeoJson(geom, popup=popup_info, style_function=lambda x: {"color": "red", "weight": 3}).add_to(m)
        elif geom.geom_type in ["Point", "MultiPoint"]:
            folium.Marker(location=[geom.y, geom.x], popup=popup_info, icon=folium.Icon(color="green")).add_to(m)

    # ✅ Render map in Streamlit
    #folium_static(m)
    st.components.v1.html(folium.Figure().add_child(m).render(), height=500)


# -------------------- Modify Existing Query --------------------
def modify_existing_query(previous_sql, user_input):
    """Modify the previous SQL query based on user instructions."""
    if "add river" in user_input.lower():
        # ✅ Modify query to add rivers using JOIN or UNION
        return f"""
        {previous_sql.strip()} 
        JOIN calgaryriver r 
        ON ST_Intersects(geometry, r.geometry);
        """
    elif "add buffer" in user_input.lower():
        # ✅ Modify query to add buffer distance
        return f"""
        {previous_sql.strip()} 
        WHERE ST_Buffer(geometry, 500) IS NOT NULL;
        """
    elif "increase distance" in user_input.lower():
        # ✅ Modify distance threshold dynamically
        return re.sub(r"(ST_DWithin\(.*?,.*?, )(\d+)(\))", lambda m: f"{m.group(1)}{int(m.group(2)) + 500}{m.group(3)}", previous_sql)

    return previous_sql  # Default: Return the same query if no modification detected


def main():
    """Prompt-based interface to ask spatial queries and visualize results."""

    st.title("🧭 Ceno-GeoSearch")
    st.markdown("Search spatial insights using natural language prompts.")

    # Sidebar for database
    st.sidebar.header("Database Connection")
    db_url = "postgresql://postgres:pgpassword@localhost:5432/cenogis"
    st.sidebar.code(db_url)

    # Extract schema once
    if "schema" not in st.session_state:
        with st.spinner("Extracting schema from database..."):
            st.session_state.schema = extract_schema(db_url)

    st.sidebar.subheader("Extracted Schema")
    st.sidebar.text_area("Schema", st.session_state.schema, height=300)

    # Prompt input field
    user_input = st.text_input("📌 Enter your geospatial query:", placeholder="e.g., Buffer all wells by 100 meters")

    if user_input:
        st.markdown("---")

        result = analyze_query(user_input)
        st.markdown(f"**🗣️ You asked:** _{result['Query']}_")
        st.markdown(f"**🔎 Detected Intent:** `{result['Predicted']}`")
        st.markdown(f"**🤖 Intent Validator:** _{result['LLM Verification']}_")

        if result['Predicted'] in ('geospatial', 'non-geospatial'):
            st.subheader("📌 Generating Workflow Graph...")
            solution_graph = generate_knowledge_graph_from_query(user_input)
            st.write(solution_graph)

            st.subheader("⚙️ Running Geospatial Operation")
            gdf = generate_and_execute_code(user_query=user_input, solution_graph=solution_graph)

            if isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
                st.success("✅ Geospatial Task Completed")
                st.markdown(f"🔍 Total records: **{len(gdf)}**")

                # Simplify geometry for rendering
                gdf_simplified = gdf.copy()
                gdf_simplified["geometry"] = gdf_simplified["geometry"].simplify(tolerance=0.0005, preserve_topology=True)

                # Convert geometry to short WKT for display
                gdf_display = gdf_simplified.copy()
                gdf_display["geometry"] = gdf_display["geometry"].apply(lambda geom: geom.wkt[:250] if geom else None)

                st.subheader("📊 Tabular Results")

                # Row preview selector
                limit = st.slider("Rows to preview:", min_value=100, max_value=len(gdf), step=100, value=min(500, len(gdf)))
                st.dataframe(gdf_display.head(limit))

                # CSV download
                st.download_button("📥 Download CSV", gdf_display.to_csv(index=False), file_name="data.csv", mime="text/csv")

                # Map visualization
                st.subheader("🗺️ Visualization")
                visualize_geodata(gdf_simplified)

            else:
                st.warning("⚠️ No data returned. Please refine your query.")

    # Footer
    st.markdown("<div style='text-align:center; color:#B0B0B0; margin-top:30px;'>© 2024 Ceno-GeoSearch</div>", unsafe_allow_html=True)

# -------------------- Run the App --------------------
if __name__ == "__main__":
    main()

