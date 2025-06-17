# Ceno-GeoSearch: Natural Language Interface for Spatial SQL Analytics

Ceno-GeoSearch is a prompt-driven geospatial assistant that leverages **LangChain**, **Ollama (Qwen2.5-Coder)**, and a **solution graph agent** to translate natural language queries into executable spatial SQL workflows. It supports **PostGIS-enabled PostgreSQL** databases and enables users to visualize results dynamically in Streamlit using Folium.

---

## Features

- **Natural Language to SQL**: Generates PostGIS-compatible spatial queries from prompts
- **Solution Graph Reasoning**: Constructs a task graph from prompts and chooses functions accordingly
- **Schema-Aware Validation**: Ensures SQL queries align with actual table/column definitions
- **Interactive GIS Visuals**: Uses Folium for rendering points, lines, polygons on a web map
- **Qwen2.5-Coder LLM (via Ollama)**: Open-source coding LLM optimized for local, secure inference

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com) (installed and running locally)
- PostgreSQL with PostGIS
- GeoPandas, Folium, LangChain, SQLAlchemy

---

   
