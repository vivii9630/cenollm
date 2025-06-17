# File: clarify_agent.py

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Function to clarify user intent with multi-turn interaction

def clarify_intent_with_user(query):
    llm = Ollama(model="qwen2.5-coder:7b")

    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
You are a helpful assistant responsible for clarifying user queries before classifying them.
The original classification between 'geospatial' and 'non-geospatial' was uncertain or contradictory.

Start a clarification conversation with the user to gather more context and details.

Original query: "{query}"

Your task:
1. Politely ask for more context if the query is too vague.
2. Ask questions that will help identify if spatial/geometric operations are implied (e.g., maps, buffers, distances).
3. End by deciding if it is 'geospatial' or 'non-geospatial' and explain your decision clearly.

Begin your clarification now.
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(query=query)

"""
# Optional helper if used from CLI or for testing
if __name__ == "__main__":
    sample_query = "Get data about pipelines near rivers"
    clarification = clarify_intent_with_user(sample_query)
    print(clarification)
"""