from typing import Literal
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langgraph.prebuilt import create_react_agent


schema="""
    Database Schema:

    Table: spatial_ref_sys
    - srid (INTEGER)
    - auth_name (VARCHAR)
    - auth_srid (INTEGER)
    - srtext (VARCHAR)
    - proj4text (VARCHAR)

    Table: pipelines
    - pipeline_i (TEXT)
    - approval_n (BIGINT)
    - line_numbe (BIGINT)
    - alternatea (TEXT)
    - sourceuniq (TEXT)
    - sourcealtu (TEXT)
    - linename (TEXT)
    - substance_ (TEXT)
    - substance1 (TEXT)
    - substanc_1 (TEXT)
    - substanc_2 (TEXT)
    - licensee_c (TEXT)
    - licensee_n (TEXT)
    - licensee_a (TEXT)
    - operator_c (TEXT)
    - operator_n (TEXT)
    - operator_a (TEXT)
    - area_offic (TEXT)
    - regauthori (TEXT)
    - fedregulat (BIGINT)
    - regulatory (TEXT)
    - provstate (TEXT)
    - from_facil (TEXT)
    - from_locat (TEXT)
    - fsv1–fsv8 (TEXT)
    - to_facilit (TEXT)
    - to_locatio (TEXT)
    - tsv1–tsv8 (TEXT)
    - status (TEXT)
    - environmen (TEXT)
    - permitappr (TEXT)
    - permitexpi (TEXT)
    - licenseapp (TEXT)
    - constructi (TEXT)
    - length (DOUBLE PRECISION)
    - stress_lev (BIGINT)
    - material (TEXT)
    - type (TEXT)
    - grade (TEXT)
    - code (TEXT)
    - yield (DOUBLE PRECISION)
    - joints (TEXT)
    - internal_p (TEXT)
    - external_p (TEXT)
    - outside_di (DOUBLE PRECISION)
    - wall_thick (DOUBLE PRECISION)
    - max_op_pre (BIGINT)
    - h2s_conten (DOUBLE PRECISION)
    - attributes (TEXT)
    - attributem (TEXT)
    - attributed (TIMESTAMP)
    - featureid (TEXT)
    - teamname (TEXT)
    - districtna (TEXT)
    - fieldname (TEXT)
    - networknam (TEXT)
    - globalid (TEXT)
    - created_by (TEXT)
    - created (TIMESTAMP)
    - last_modif (TEXT)
    - last_mod_1 (TIMESTAMP)
    - st_length_ (DOUBLE PRECISION)
    - geometry (GEOMETRY) — geospatial column

    Table: wells
    - uwi (TEXT)
    - formatted_ (TEXT)
    - surf_locat–surf_loc_8 (TEXT)
    - well_name (TEXT)
    - well_num (TEXT)
    - country (TEXT)
    - province_s (TEXT)
    - field_id (TEXT)
    - field_name (TEXT)
    - pool_id (TEXT)
    - pool_name (TEXT)
    - current_cl (TEXT)
    - status_typ (TEXT)
    - current_st (TEXT)
    - previous_s (TEXT)
    - original_s (TEXT)
    - licensee_c (TEXT)
    - licensee_n (TEXT)
    - licensee_a (TEXT)
    - operator_c (TEXT)
    - operator_n (TEXT)
    - operator_a (TEXT)
    - drill_td (DOUBLE PRECISION)
    - final_td (DOUBLE PRECISION)
    - plugback_d (DOUBLE PRECISION)
    - whipstock_ (DOUBLE PRECISION)
    - ground_ele (DOUBLE PRECISION)
    - kb_elev (DOUBLE PRECISION)
    - completion (TEXT)
    - final_dril (TEXT)
    - onprod_dat (TEXT)
    - oninject_d (TEXT)
    - rig_releas (TEXT)
    - spud_date (TEXT)
    - current__1 (TEXT)
    - abandonmen (TEXT)
    - surface_ab (TEXT)
    - plot_label (TEXT)
    - plot_symbo (TEXT)
    - plot_name (TEXT)
    - profile_ty (TEXT)
    - location_q (TEXT)
    - surf_latit (DOUBLE PRECISION)
    - surf_longi (DOUBLE PRECISION)
    - base_latit (DOUBLE PRECISION)
    - base_longi (DOUBLE PRECISION)
    - attributes (TEXT)
    - attributem (TEXT)
    - attributed (TIMESTAMP)
    - teamname (TEXT)
    - districtna (TEXT)
    - fieldname (TEXT)
    - networknam (TEXT)
    - globalid (TEXT)
    - created_by (TEXT)
    - created (TIMESTAMP)
    - last_modif (TEXT)
    - last_mod_1 (TIMESTAMP)
    - geometry (GEOMETRY) — geospatial column

    Table: township
    - pcuid (TEXT)
    - pcpuid (TEXT)
    - pcname (TEXT)
    - pctype (TEXT)
    - pcclass (TEXT)
    - cmauid (TEXT)
    - cmapuid (TEXT)
    - cmaname (TEXT)
    - cmatype (TEXT)
    - pruid (TEXT)
    - prname (TEXT)
    - st_area_sh (DOUBLE PRECISION)
    - st_length_ (DOUBLE PRECISION)
    - geometry (GEOMETRY) — polygon geometry

    Table: primary_roads
    - feature_id (TEXT)
    - md_tempora, md_tempo_1 (TEXT)
    - md_horiz_p, md_horiz_1 (DOUBLE PRECISION)
    - closing_pe (INTEGER)
    - exit_numbe (TEXT)
    - political_ (INTEGER)
    - road_juris, road_jur_1 (TEXT)
    - is_nationa, is_trans_c, number_of_ (INTEGER)
    - road_class (INTEGER)
    - geobase_ni (TEXT)
    - route_name–route_na_7 (TEXT)
    - road_segme, road_seg_1 (TEXT)
    - route_numb, route_nu_1, route_nu_2 (TEXT)
    - speed_rest, road_surfa, is_paved, traffic_di (INTEGER)
    - of_municip, of_munic_1 (TEXT)
    - official_p–official_4 (TEXT)
    - of_directi–of_direc_3 (TEXT)
    - of_street_–of_stree_8 (TEXT)
    - official_s (TEXT)
    - first_hous, last_house, first_ho_3, last_hou_3 (BIGINT)
    - other house fields (TEXT, INTEGER)
    - numbering_, numbering1, ref_system, ref_syst_1, digitizing, digitizi_1 (INTEGER)
    - road_struc, road_str_1–road_str_3 (TEXT/INTEGER)
    - map_select (INTEGER)
    - st_length_ (DOUBLE PRECISION)
    - geometry (GEOMETRY) — geospatial column

    Table: hydrolines
    - category (TEXT)
    - type (TEXT)
    - name (TEXT)
    - provstate (TEXT)
    - st_length_ (DOUBLE PRECISION)
    - geometry (GEOMETRY) — geospatial column
    """
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="qwen2.5-coder:7b")
# 1. SQL Agent
sql_writer_agent = create_react_agent(
    llm,
    tools=[],
    prompt=f"""You are a SQL generation assistant. Convert the user query into a valid SQL command that DOES NOT CHANGE OR DELETE ANYTHING IN THE DATABASE based on the database schema: {schema}"""
)

def sql_writer_node(state: MessagesState) -> Command[Literal["validator"]]:
    result = sql_writer_agent.invoke(state)

    last_msg = result["messages"][-1]
    response = last_msg if isinstance(last_msg, str) else last_msg.content

    result["messages"][-1] = HumanMessage(
        content=response,
        name="sql_writer"
    )
    print("SQL Writer:", response)

    return Command(update={"messages": result["messages"]}, goto="validator")

# 2. Validator Agent
validator_agent = create_react_agent(
    llm,
    tools=[],
    prompt=f"""You are a SQL validator. Check if the SQL query is valid against the provided schema and that the given command will not modify the database in anyways. If not, return reasons and flag is_valid: false. Schema:{schema}"""
)

def validator_node(state: MessagesState) -> Command[Literal["interrupt", "executor"]]:
    result = validator_agent.invoke(state)

    # Handle case where last message might be a string instead of a message object
    last_msg = result["messages"][-1]
    response = last_msg if isinstance(last_msg, str) else last_msg.content

    is_valid = "true" in response.lower()
    print("Validator:", response)
    
    return Command(
        update={"messages": result["messages"]}, goto="executor" if is_valid else "interrupt"
    )

# 3. Interrupt Handler (Human-in-the-loop)
def interrupt_node(state: MessagesState) -> Command[Literal["sql_writer"]]:
    user_input = input("\nPlease enter a revised query:\n> ")
    new_message = HumanMessage(content=user_input)
    updated_messages = state["messages"] + [new_message]
    return Command(update={"messages": updated_messages}, goto="sql_writer")
# 4. Dummy Executor Node
def executor_node(state: MessagesState) -> Command[Literal["__end__"]]:
    print(">>> LANGGRAPH END NODE REACHED <<<")
    return Command(update=state, goto=END)

# 5. Build the Graph
builder = StateGraph(MessagesState)
builder.add_node("sql_writer", sql_writer_node)
builder.add_node("validator", validator_node)
builder.add_node("interrupt", interrupt_node)
builder.add_node("executor", executor_node)

builder.set_entry_point("sql_writer")
builder.set_finish_point("executor")

graph = builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    user_input = input()

    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    from IPython.display import Image, display
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # You can put your exception handling code here
        pass
    # print("\nFinal Messages:")
    # for msg in result["messages"]:
    #     name = msg.name if msg.name else "user"
    #     print(f"{name}: {msg.content}")
