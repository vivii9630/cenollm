import os
import re
import sys
import importlib
import ast
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import traceback
import importlib.util

# Load LLM
llm = Ollama(model="qwen2.5-coder:7b")


def extract_function_definitions(file_path):
    """Extract functions with arguments and their bodies from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    parsed = ast.parse(source)
    function_snippets = []

    for node in parsed.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            args = [arg.arg for arg in node.args.args]
            func_code = ast.get_source_segment(source, node)
            snippet = f"\n### {func_name}({', '.join(args)}):\n{func_code}"
            function_snippets.append(snippet)

    return "\n".join(function_snippets)


# Extract reusable spatial functions
spatial_func_path = os.path.join("operations", "spatial_func.py")
spatial_context = extract_function_definitions(spatial_func_path)
def extract_examplecode(path):
    """
    Reads and returns all the code from a Python file (e.g., examples.py).
    
    Parameters:
        path (str): Full path to the examples.py file.
        
    Returns:
        str: Entire code as a string.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# ❌ Error reading {path}: {str(e)}"
example_func_path = os.path.join("operations", "examples.py")
example_for_generation = extract_examplecode(example_func_path)

# Optimized PromptTemplate
prompt = PromptTemplate(
    input_variables=["user_query", "solution_graph", "spatial_context","example_for_generation"],
    template="""
You are an expert geospatial coding agent. Your job is to generate **only Python code** using existing modules and reusable spatial functions below.

### User Query:
{user_query}

### Solution Graph:
{solution_graph}

### Available Spatial Functions (You can reuse these):
{spatial_context}

🧠 Rules:
1. Compose logic in a single function `final_gdf()` which returns a GeoDataFrame `gdf`.
2. Use `operations.module_name.run()` to import and fetch data.
3. You MUST use functions from the spatial context when relevant.
4. DO NOT add any explanations, markdown, or extra print statements.
5. Final output must be valid Python code only (no markdown).
6. Make sure to import **all extra** libraries that are required - pandas, numpy etc.

For example, if the {user_query} is to find the intersection between pipelines and hydrolines, the code should be generated as shown in:
{example_for_generation}

"""
)

# LangChain Chain
chain = LLMChain(llm=llm, prompt=prompt)


def clean_generated_code(code):
    """Removes ```python or ``` from code blocks."""
    return re.sub(r"```(?:python)?", "", code).strip("` \n")


def generate_and_execute_code(user_query, solution_graph):
    """Generate, write, and run a geospatial task script based on query + solution graph."""
    
    # Step 1: Generate Python code from LLM
    raw_code = chain.run(
        user_query=user_query,
        solution_graph=solution_graph,
        spatial_context=spatial_context,
        example_for_generation=example_for_generation
    )
    code = clean_generated_code(raw_code)

    # Step 2: Save the code into operations/generated_task.py
    generated_path = os.path.join("operations", "generated_task.py")
    with open(generated_path, "w", encoding="utf-8") as f:
        f.write(code)

    # Step 3: Safely import and execute final_gdf from generated_task.py
    try:
        # Dynamically import generated_task.py using importlib.util
        spec = importlib.util.spec_from_file_location("generated_task", generated_path)
        generated_task = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_task)

        # Check that final_gdf exists
        if not hasattr(generated_task, "final_gdf"):
            raise AttributeError("The module does not contain a 'final_gdf()' function.")

        gdf = generated_task.final_gdf()
        print(gdf)
        return gdf

    except Exception as e:
        error_details = traceback.format_exc()
        raise RuntimeError(f"❌ Error executing generated code:\n\n{error_details}")
