import os
import ast
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

def list_available_modules_with_signatures(folder_path):
    """
    List all .py files and their functions with arguments, excluding certain ones.
    """
    exclude_files = {
        "__init__.py", "solution_graph_agent.py",
        "execute.py", "generated_task.py", "examples.py"
    }

    module_functions = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename not in exclude_files:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        arg_list = [arg.arg for arg in node.args.args]
                        signature = f"{node.name}({', '.join(arg_list)})"
                        module_functions.append({
                            "file": filename,
                            "function": node.name,
                            "signature": signature
                        })

            except Exception as e:
                module_functions.append({
                    "file": filename,
                    "function": "ERROR",
                    "signature": f"❌ Error reading file ({e})"
                })

    return module_functions


def generate_knowledge_graph_from_query(user_query, module_folder="operations"):
    """
    Generates a knowledge graph (nodes and edges) using an LLM based on the query and Python module metadata.
    """
    llm = Ollama(model="qwen2.5-coder:7b")
    modules_info = list_available_modules_with_signatures(module_folder)

    module_context = "\n".join([
        f"- `{item['file']}` → `{item['signature']}`"
        for item in modules_info
    ])

    prompt = PromptTemplate(
        input_variables=["query", "modules"],
        template="""
You are a Knowledge Graph generator. Given the user query and a list of available Python functions from local files, do the following:

1. Select the relevant modules and functions required to fulfill the user's query.
2. Define a **knowledge graph** with:
    - **Nodes**: Representing modules/functions that should be used.
    - **Edges**: Representing the data flow or dependency between modules.

Each node must be based on the real files/functions listed below. Do not make up new ones. Use the function signatures to understand dependencies.

### User Query:
{query}

### Available Modules and Functions:
{modules}

### Output Format (JSON preferred if possible):
{{
  "nodes": [
    {{"id": "get_wells_as_gdf.run()", "description": "Extract wells from database"}},
    {{"id": "spatial_func.run_buffer(gdf, distance)", "description": "Buffer geometries by distance"}}
  ],
  "edges": [
    {{"from": "get_wells_as_gdf.run()", "to": "spatial_func.run_buffer(gdf, distance)"}}
  ]
}}

"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    kg_output = chain.run(query=user_query, modules=module_context)
    return kg_output
