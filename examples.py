from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

model = ChatOpenAI(temperature=0, streaming=True)
import random


@tool
async def where_cat_is_hiding() -> str:
    """Where is the cat hiding right now?"""
    return random.choice(["under the bed", "on the shelf"])


@tool
async def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "bed" in place:  # For under the bed
        return "socks, shoes and dust bunnies"
    if "shelf" in place:  # For 'shelf'
        return "books, penciles and pictures"
    else:  # if the agent decides to ask about a different place
        return "cat snacks"
    
# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-tools-agent")
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant"),
  ("placeholder", "{chat_history}"),
  ("human", "{input}"),
  ("placeholder", "{agent_scratchpad}"),
])


tools = [get_items, where_cat_is_hiding]
agent = create_openai_tools_agent(
    model.with_config({"tags": ["agent_llm"]}), tools, prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

# Note: We use `pprint` to print only to depth 1, it makes it easier to see the output from a high level, before digging in.
import pprint

# Define an async function to run the agent
async def run_agent():
    chunks = []
    async for chunk in agent_executor.astream(
        {"input": "what's items are located where the cat is hiding?"}
    ):
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
        # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                print(f"Tool Result: `{step.observation}`")
        # Final result
        elif "output" in chunk:
            print(f'Final Output: {chunk["output"]}')
        else:
            raise ValueError()
        print("---")

# Run the async function
import asyncio
asyncio.run(run_agent())