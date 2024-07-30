from langchain.agents import Tool
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import YouTubeSearchTool
from langchain.agents import initialize_agent
# Initialize tools
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain_community.utilities import GoogleSearchAPIWrapper
import os
os.environ["GOOGLE_CSE_ID"] = "66297c07f31dc4c6d"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDE0ErQezoMG9H33i7IZ6x663yLzUe7_hA"

llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key = os.environ["OPEN_AI_KEY"],
                 temperature = 0.3)

google_search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=google_search.run,
)
# search = DuckDuckGoSearchRun()
# search_tool = Tool(name = "search_tool", description="search the net", func= search.run)
yt = YouTubeSearchTool()
yt_tool = Tool(name="youtube", description="youtube search for video", func = yt.run)

tools = [search_tool, yt_tool]
agent = initialize_agent(tools=tools, llm=llm, agent = "zero-shot-react-description",
                         verbose=True)

# st.title("First Agent")
# prompt = st.text_input("Input The promt here")
prompt = "get the latest news about France Olympics"

if prompt:
    response = agent.run(prompt)
    print(response)
#     st.write(response)
# search = SerpAPIWrapper()
# tools = [
#     Tool(
#         name="search",
#         func=search.run,
#         description="Useful for answering questions about current events with targeted queries.",
#     ),
#     WriteFileTool(),  # Tool for writing files
#     ReadFileTool(),   # Tool for reading files
# ]