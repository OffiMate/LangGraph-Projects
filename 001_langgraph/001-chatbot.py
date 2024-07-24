"""
In this one, we are creating a simple agent
that takes a llm as an engine, to make
decisions for the output and inputs.
This makes langchain models to be just more than
the output machine we know
"""
from dotenv import load_dotenv
import os
from langchain_community.tools import BraveSearch
from langchain_google_genai import ChatGoogleGenerativeAI
import pprint
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage


# setting up the environment variables for the logging traces
load_dotenv()
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("BRAVE_SEARCH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# then set up the tavily logins. Ideally this is a websearch tool.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# step 2
"""
creating the tool that we want to use. In our case, the tool of use here
is Tavily -- the search engine. Ideally this is what the consider the first
reseach engine for AI - used by langchains and LLMs.
"""
search_tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
search_query_tool = search_tool.invoke("What do you after having sex?")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(search_query_tool)

# the list below is a list of all tools that will be used here, and they will be under
# the name of the variable named = tools.
tools = [search_tool]

# step 3
"""
calling that llm on the tool we created above - called search_query_tool
"""
# declare our model in use
model = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY, temperature=0.5)

# create the agent using the model, and tools using create_react_agent() method
# agent_executor = create_react_agent(model, tools) // check line 56

# now that we created an agent, we want to save memory and have it remembering us
"""
the idea is to have a 1. checkpointer and 2. and id for that checkpointer.
That's how the llm and agent in general remembers what to consider.
"""
memory = SqliteSaver.from_conn_string(":memory:")
# then add memory to the agent
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {
    "configurable": {
        "thread_id": "abcd123"
    }
}

for conversation in agent_executor.stream(
        {
            "messages": [
                HumanMessage(
                    content="Hello, I am a young girl new to the art of teenage hood"
                )
            ]
        }, config
):
    print(conversation)
    print("------")
