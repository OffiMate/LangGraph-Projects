"""
In this one, we are creating a simple agent
that takes a llm as an engine, to make
decisions for the output and inputs.
This makes langchain models to be just more than
the output machine we know
"""
from dotenv import load_dotenv
import os
import json
from langchain_community.tools import BraveSearch
import pprint

# setting up the environment variables for the logging traces
load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("BRAVE_SEARCH_API_KEY")
# then set up the tavily logins. Ideally this is a websearch tool.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# step 2
"""
creating the tool that we want to use. In our case, the tool of use here
is Tavily -- the search engine. Ideally this is what the consider the first
reseach engine for AI - used by langchains and LLMs.
"""
search = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
search_query = search.invoke("What do you after having sex?")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(search_query)
# the list below is a list of all tools that will be used here, and they will be under
# the name of the variable named = tools.
tools = [search]
