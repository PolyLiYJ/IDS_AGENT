# from llm_agents import Agent, ChatLLM, PythonREPLTool, SerpAPITool, GoogleSearchTool, HackerNewsSearchTool
from pyexpat import model
import joblib
from langchain.tools import tool
# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
import pandas as pd
from langchain import hub
import numpy as np
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
import pprint
import asyncio
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import re
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import argparse
import os
with open('openai_key_mine.txt', 'r') as file:
    # Read the first line from the file
    first_line = file.readline()
    os.environ["OPENAI_API_KEY"] = first_line
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI

from database import MyKnowledgeBase
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langchain.agents import create_json_chat_agent, create_openai_tools_agent
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings 
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_experimental.chat_models import Llama2Chat

from uuid import uuid4
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_experimental.tools import PythonREPLTool
from langchain_community.llms import Ollama
# from langchain_community.tools.tavily_search import TavilySearchResults
# from app.models.chat import NewChatItem, RoleEnum, ChatItem, ChatBody
# router = APIRouter(
#     prefix="/v1/chat",
#     tags=["Chat"],
#     responses={404: {"description": "Not found"}},
# )


import json




# os.environ["GOOGLE_CSE_ID"] = ""
# os.environ["GOOGLE_API_KEY"] = ""
# DataProcessingTool
# Context extraction
# FeatureSelectionTool
# Scikit-learn API /Pretrain model Tool
# huggingface model
# Arkiv Tool
# 结合Snort，Suricata进行实时流量监控和入侵响应。
# 调用XAI解释
# knowledge 用来解释分类结果


# class CustomToolException(Exception):
#     """Custom LangChain tool exception."""

#     def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
#         super().__init__()
#         self.tool_call = tool_call
#         self.exception = exception


# def tool_custom_exception(msg: AIMessage, config: RunnableConfig) -> Runnable:
#     try:
#         return complex_tool.invoke(msg.tool_calls[0]["args"], config=config)
#     except Exception as e:
#         raise CustomToolException(msg.tool_calls[0], e)


# def exception_to_messages(inputs: dict) -> dict:
#     exception = inputs.pop("exception")

#     # Add historical messages to the original input, so the model knows that it made a mistake with the last tool call.
#     messages = [
#         AIMessage(content="", tool_calls=[exception.tool_call]),
#         ToolMessage(
#             tool_call_id=exception.tool_call["id"], content=str(exception.exception)
#         ),
#         HumanMessage(
#             content="The last tool call raised an exception. Try calling the tool again with corrected arguments. Do not repeat mistakes."
#         ),
#     ]
#     inputs["last_output"] = messages
#     return inputs

async def get_message(promptMsg, tools):
    instructions = """You are an assistant."""
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    llm1 = Ollama(model="llama3")
    # tavily_tool = TavilySearchResults()
    tools = []

    agent = create_react_agent(llm=llm1, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res = agent_executor.invoke(
        {"input": promptMsg}
    )

    yield res.encode()

# @router.post("/completions")
# async def get_chat(chatBody: ChatBody):
#     return StreamingResponse(
#         get_message(chatBody.promptMsg),
#         status_code=200,
#         media_type="text/event-stream",
#     )


# Define your custom error handling function
def custom_error_handler(error) -> str:
    if error is None:
        return "Received None as an Action Input, please provide a valid input."
    else:
        return str(error)

def string_to_dataframe(traffic):
    """Convert a formatted string back into a DataFrame."""
    # Split the string into key-value pairs
    pairs = traffic.split(', ')
    
    # Create a dictionary from the key-value pairs
    record_dict = {}
    for pair in pairs:
        key, value = pair.split(': ')
        record_dict[key] = value
    
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame([record_dict])
    
    return df

def get_list_dim(lst):
    if isinstance(lst, list):
        if not lst:  # 空列表
            return 1
        return 1 + get_list_dim(lst[0])
    else:
        return 0
    
def retriever_qa_creation(folder):
    kb = MyKnowledgeBase(pdf_source_folder_path=folder)
    # ollama_model_name = "llama2:7b-chat-q6_K"
    # embedder = OllamaEmbeddings(model=ollama_model_name)
    # chroma_db_database = 'db-llama'
    # retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    
    # # vector_db = kb.initiate_document_injection_pipeline(embedder)
    # # retriever = vector_db.as_retriever()

    # llm = Ollama(model=ollama_model_name) #, temperature=0.1)
    
    embedder = OpenAIEmbeddings()
    chroma_db_database = "db"
    retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) 

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever = retriever)
    return qa

retriever_qa_chain = retriever_qa_creation(folder = "/Users/yanjieli/program/IDS_AGENT/documents")

@tool
async def knowledge_retriever(query:str)->str:
    """search the knowledge of different attacks or information about the network"""
    return retriever_qa_chain.run(query)

class DataIDInput(BaseModel):
    file_path: str = Field(description="should be a file path")
    flow_id: str = Field(description="should be a flow ID")

@tool("load_data_ID", args_schema = DataIDInput, return_direct=True)
async def load_data_ID(file_path: str, flow_id: str) -> str:
    """input flow_id and output traffic features"""
    data = pd.read_csv(file_path)
    columns_to_drop = [ "Label"]
    data.drop(columns=columns_to_drop, inplace=True)
    record = data[data["Flow ID"] == flow_id]
        
    if not record.empty:
        # Convert the single row DataFrame to a dictionary
        record_dict = record.iloc[0].to_dict()
        # Convert the dictionary to a string
        record_str = ', '.join(f'{key}: {value}' for key, value in record_dict.items())
        return record_str
    else:
        print(f"No record found for Flow ID {flow_id}")
        return f"No record found for Flow ID {flow_id}"


class DataLineInput(BaseModel):
    # file_path: str = Field(description="The file path to the CSV file")
    line_number: int = Field(description="The line number in the CSV file")
    
@tool("load_data_line", args_schema=DataLineInput, return_direct=True)
async def load_data_line(line_number: int) -> str:
    """Load traffic features from the specified line number in a CSV file."""
    # Read the CSV file
    file_path = "/Users/yanjieli/program/IDS_AGENT/dataset/CICIoT2023/test_set_small.csv"
    data = pd.read_csv(file_path)
    columns_to_drop = ["label", "final_label"]
    data.drop(columns=columns_to_drop, inplace=True)
    
    # Select the specified line number
    record = data.iloc[[line_number]]
    
    if not record.empty:
        # Convert the single row DataFrame to a dictionary
        record_dict = record.iloc[0].to_dict()
        # Convert the dictionary to a string
        record_str = ', '.join(f'{key}: {value}' for key, value in record_dict.items())
        return record_str
    else:
        print(f"No record found for line number {line_number}")
        return None

class DataProcessingInput(BaseModel):
    # traffic_features: str = Field(description="The input is a full traffic features string, including multiple features.")
    # file_path: str = Field(description="should be a file path")
    line_number: int = Field(description="should be a line_number")


@tool("data_preprocessing", args_schema = DataProcessingInput, return_direct=True)
async def data_preprocessing(line_number: int) -> list:
    """input flow_id and output preprocessed traffic features"""
    file_path = "/Users/yanjieli/program/IDS_AGENT/dataset/CICIoT2023/test_set_small.csv"
    data = pd.read_csv(file_path)
    columns_to_drop = [ "label", "final_label"]
    data.drop(columns=columns_to_drop, inplace=True)
    record = data.iloc[[line_number]]
    
    """Preprocess the traffic record by selecting and scaling the features. This should be done after loading the data."""
    # columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Connection Type']
    # df.drop(columns=columns_to_drop, inplace=True)
    
    # Handle missing values
    record.replace([np.inf, -np.inf], 0, inplace=True)
    
    selector = joblib.load('/Users/yanjieli/program/IDS_AGENT/models_CICIoT/kbest_selector.joblib')
    scaler = joblib.load('/Users/yanjieli/program/IDS_AGENT/models_CICIoT/scaler.joblib')
    
    
    # Transform the input data
    X_selected = selector.transform(record)
    X_scaled = scaler.transform(X_selected)
    X_scaled = np.round(X_scaled, 4)
    
    return X_scaled.tolist()

# 定义Softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减去max(x)是为了数值稳定性
    return exp_x / exp_x.sum(axis=1, keepdims=True)


class ClassifyInput(BaseModel):
    modelname: str = Field(description="This is a classfication model name")
    traffic_features: list = Field(description="This is the preprocessed data to be classified")
    top_k: int = Field(default=3, description="Number of top predictions to return")
    
@tool("classify", args_schema=ClassifyInput, return_direct=True)
async def classifier(modelname: str, traffic_features: list, top_k: int = 3) -> str:
    """Classify the traffic features using a machine learning model to determine if the traffic record is an attack."""
    
    # Load the model
    model = joblib.load(f"models_CICIoT/{modelname}.joblib")
    
    # Ensure traffic_features is a 2D array for prediction
    if isinstance(traffic_features[0], list):  # Already 2D
        traffic_features_array = np.array(traffic_features)
    else:  # Convert to 2D
        traffic_features_array = np.array([traffic_features])
    
    # Make predictions
    label_encoder = joblib.load('models_CICIoT/label_encoder.joblib')
    labels = label_encoder.classes_

    probabilities = None
    
    # Get the score/probability if possible
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(traffic_features_array)
    elif hasattr(model, "decision_function"):
        decision_values = model.decision_function(traffic_features_array)
        probabilities = softmax(decision_values)
    
    if probabilities is not None:
        # Get the top k predictions and their corresponding labels
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:][0][::-1]
        top_k_labels = labels[top_k_indices]
        top_k_probabilities = probabilities[0][top_k_indices]

        # Format the output
        results = [
            f"{label}: {prob:.4f}" for label, prob in zip(top_k_labels, top_k_probabilities)
        ]
        return "Top predictions: " + ", ".join(results)
    else:
        return "Model does not support probability output."

# example = load_data_line("/Users/yanjieli/program/IDS_AGENT/dataset/ACIIoT/test_set_small.csv", 1)
# X_scaled = data_preprocessing(example)
# pred = classify("Decision Tree", X_scaled)
# print(pred)

def ids_system_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a cyber security expert who processes traffic records to classify whether they are attacks. You must follow these steps:"),
            ("assistant", """
            1. Load the traffic features using the `load_data_line` tool.
            2. Preprocess the loaded features using the `data_preprocessing` tool.
            3. Use the preprocessed data to classify the record with the `classify` tool.
            4. Summarize the results and explain your reasoning. 
            """),
            ("placeholder", "{chat_history}"), #important
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    return prompt
    

def chat_json_prompt():
    system = '''Assistant is a large language model trained by OpenAI.
    Assistant is designed to be able to assist with network intrusion detection task, 
    from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
    As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this  knowledge to provide accurate and informative responses to a wide range of questions.             
    Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
    Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 
    Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

    human = '''TOOLS
    ------
    Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

    {tools}

    RESPONSE FORMAT INSTRUCTIONS
    ----------------------------

    When responding to me, please output a response in one of two formats:

    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:

    ```json
    {{
        "action": string, \ The action to take. Must be one of {tool_names}
        "action_input": string \ The input to the action
    }}
    ```

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    ```json
    {{
        "action": "Final Answer",
        "final answer": string \ Benign or the attack name, according to the prediction of different classifiers.
        "action_input": string \ You should put what you want to return to use here
    }}
    ```

    USER'S INPUT
    --------------------
    Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

    {input}'''
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return prompt

def load_few_shot(file_path = "few_shot.txt"):
    with open(file_path, "r") as file:
        few_shot = file.read()
    return few_shot

def react_prompt():
    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question, should be a classification of the input traffic.

    When output the final answer, use this Markdown code snippet formatted in the following schema:
    %%%json
    {{
        "action": "Final Answer",
        "final prediction": string \ Benign or the attack name, according to the prediction of different classifiers.
        "action_input": string \ You should put what you want to return to use here
    }}
    %%%
    
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}

    '''

    prompt = PromptTemplate.from_template(template)
    return prompt
    
    

async def run_agent(file_path = "dataset/CICIoT2023/test_set_small.csv",
                    model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression", "MLP", "Support Vector Classifier","Decision Tree"],
                    attitude ="Balanced",
                    # llm_model_name = "gpt-3.5-turbo-0125"
                    llm_model_name = "gpt-4o"
                    ):
    # or gpt-3.5-turbo-0125/gpt-4o
    
    #llm = ChatOpenAI(model=llm_model_name, temperature=0)
    
    #ollama_model_name = "llama2:7b-chat-q6_K"
    #ollama_model_name= "llama3.1"
    #llm = Ollama(model=ollama_model_name) #, temperature=0.1)
    
    # jondurbin/airoboros-l2-7b-gpt4-m2.0
    # xtuner/Llama-2-7b-qlora-msagent-react
    llm = HuggingFacePipeline.from_model_id(
         model_id="jondurbin/airoboros-l2-7b-gpt4-m2.0",
         task="text-generation",
         device=-1,  # -1 for CPU
         batch_size=1,  # adjust as needed based on GPU map and model size.
         model_kwargs={"temperature": 0, "max_length": 1024},
     )
    
    # llm = Llama2Chat(llm = llama2)
    

    # When there are discrepancy/disagreements for different models, 
    # you can search from the google or Wiki to get more information about the difference of these attacks to help you to make decision.
    # Determine the number of lines in the file
    df = pd.read_csv(file_path)
    labels  = df["label"].unique()

    true_labels = []
    predicted_labels = []
    flow_ids = []
    
    predicts_analysis = []
    
    # define agent
    os.environ["GOOGLE_CSE_ID"] = "66297c07f31dc4c6d"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDE0ErQezoMG9H33i7IZ6x663yLzUe7_hA"
    # search = SerpAPIWrapper()
    # search_tool = Tool(
    #         name="Search",
    #         func=search.run,
    #         description="useful for when you need to answer questions about current events",
    #     ), 
    google_search = GoogleSearchAPIWrapper(k=5)
    # search_tool = Tool(
    #     name="google_search",
    #     description="Search Google for attack informations.",
    #     func=google_search.run,
    # )
    # Define a Pydantic model for the tool's arguments
    class GoogleSearchInput(BaseModel):
        query: str
        
    search_tool = StructuredTool(
        name="google_search",
        description="Search Google for informations.",
        func=google_search.run,  # Assuming this is the correct function for the tool
        args_schema=GoogleSearchInput
    )
    
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # tools = [load_data_line, data_preprocessing, classifier, search_tool, wiki_tool]
    tools = [load_data_line, data_preprocessing, classifier, search_tool, wiki_tool, knowledge_retriever]
    message_history = RedisChatMessageHistory(
        url="redis://127.0.0.1:6379/0", ttl=600, session_id="my-session"
    )
    
    #prompt = hub.pull("hwchase17/openai-tools-agent")
    # agent = create_openai_tools_agent(llm, tools, prompt)
    
    # prompt = hub.pull("hwchase17/react-chat-json")
    # agent = create_json_chat_agent(llm, tools, prompt)
    
    # prompt = ids_system_prompt()
    # agent = create_tool_calling_agent(llm, tools, prompt)
    
    # prompt = react_prompt()
    instructions = """You are an assistant."""
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    #prompt = hub.pull("langchain-ai/react-agent-template")
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True,  handle_parsing_errors=True)
    
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    few_shot_example = load_few_shot("few_shot.txt")
    
    for line_number in range(183, 190):
        # Define the prompt for the agent
        # flow_id = df.at[line_number,"Flow ID"]
        # If there is a significant discrepancy in the model predictions, consider the possibility of `Unknown` attacks. 
        if "Recon" not in df.at[line_number,"label"]:
            continue

        if attitude == "Aggresive":
            attitude_details = "discover the attack at the first time"
        if attitude == "Conservative":
            attitude_details = "donot alert unless you are very sure"
        if attitude == "Balanced":
            attitude_details = "balance the false alarm rate and the missing alarm rate"
        
        prompt_text = f"""
        Is the network traffic from the line number {line_number} an attack or normal traffic? 
        You are tasked with performing intrusion detection. Follow these steps for the detection process:

        1. Load the traffic features from the CSV file using tools. You can use the load_data_line tool to obtain the complete traffic_string.
        2. Select features and normalize the full feature strings using tools. This can be done using data_preprocessing tool.
        3. Finally, load classifiers for classification.
        You should classify this traffic until you reach a final classification. You can use multiple classifiers, including {model_name}. 
        4. At the end, you should summarize the results from these classifiers and provide a final result.

        The predicted_label should be one of the {labels}.
        The final output format should be: 
        %%%json
        {{
            'line_number': line_number, 
            'analysis':  str,  \here is the Analysis, 
            'predicted_label_top_1': str, \BenignTraffic or An Attack type,
            'predicted_label_top_2': str, \BenignTraffic or An Attack type,
            'predicted_label_top_3': str, \BenignTraffic or An Attack type,
        }}
        %%%
        
        """
        # Summarize the classification with {attitude} attitude, which means {attitude_details}.
        # When there are discrepancy/disagreements for different models, you can search from vector database/google/wiki to get more information about the difference of attacks to help you to make decision.
        # For example, you can call the IDS knowledge database `what is port scan attack`.


        # Here is an example
       
        #For example, when one model classify the data as DDoS, and another classify the data as DoS, you can search on google or wiki for "The difference between DoS and DDoS".
        
        # 4. Additionally, you need to explain the results based on the classification outcomes. If you are uncertain about the classification, consider using online resources like Wikipedia.
        # 5. Finnaly, give suggestions for intrusion response. You can access google and Wikipedis to summarize information.
        
        # """
        # You should follow these steps:
        # 1. Load the data using the `load_data_line` tool.
        # 2. Data Processing: Use the `data_preprocessing` tool for this.
        # 3. Load multiple Classifier Models: Use the provided pretrained ML models, including {model_name}.
        # 4. Based on the classification results, give the final classification, and explain the classification reasons if an example is classified as malicious.
        # Your goal is to accurately classify network traffic data and provide explanations for malicious classifications.
        # """
        
        print(prompt_text)


        # tools = [load_data_line, data_preprocessing, classify, PythonREPLTool(), google_search_tool]
       


        # Run the agent with the provided prompt
        # response = agent_executor.invoke({"input": prompt_text})
        # print(response)

        chunks = []
        f = open("response_results_CICIoT_0830_vectordb.txt","a")
    
        # async for chunk in agent_executor.astream({"input": prompt_text}):
        async for chunk in agent_with_chat_history.astream({"input": prompt_text},config={"configurable": {"session_id": "<foo>"}}):
            chunks.append(chunk)
            print("------")
            
            # print(chunk["messages"])
            if "actions" in chunk:
                for action in chunk["actions"]:
                    # print(f"Calling Tool: {action.tool} with input {action.tool_input}")
                    f.write(f"Calling Tool: {action.tool} with input {action.tool_input}" + "\n")
            elif "output" in chunk:
                response = chunk["output"]
                f.write(str(response) + "\n")
                print(f'Final Output: {chunk["output"]}')
                # Regular expression to find the final classification
                # Extract the JSON block from the text
                start = response.find("%%%json") + len("%%%json")
                end = response.find("%%%", start)
                match = response[start:end].strip()
                if response.find("%%%json") >= 0:
                    pred_dict = json.loads(match)
                    final_classification = pred_dict['predicted_label_top_1']
                    print(f"Final Classification: {final_classification}")
                    predicted_labels.append(final_classification)
                    data_line = df.iloc[line_number]
                    true_label = data_line['label']
                    true_labels.append(true_label)
                    pred_dict["true_label"] = true_label
                    predicts_analysis.append(pred_dict)
                    with open("results/classification_results_CICIoT_0830-vectordb.txt", "w") as f_class:
                        f_class.write(str(predicts_analysis))
                    print(f"line_number: {line_number}, True Label: {true_label}, Predicted Label: {final_classification}\n")
                else:
                    print("Final Classification not found.")
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"Tool Result: {step.observation}")
                    f.write(f"Tool Result: {step.observation}" + "\n")
            else:
                raise ValueError()
            print("--------------")

    # Compute the confusion matrix
    # Remove the first row and first column

    
    cm = confusion_matrix(true_labels, predicted_labels)    
    print(cm)   
    # print(classification_report(true_labels, predicted_labels, target_names=np.unique(true_labels)))
    f.write("\nSummary:\n")
    f.write(f"Total Records: {len(true_labels)}\n")
    
    # Write the confusion matrix to the file
    f.write("Confusion Matrix:\n")
    np.savetxt(f, cm, fmt='%d')

    # Calculate and print the classification report
    report = classification_report(true_labels, predicted_labels, target_names=labels)
    print(report)   
    f.write(report)
    
        
    # Optionally, display the confusion matrix
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_labels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.show()
        # Save the confusion matrix figure
    plt.savefig("confusion_matrix.png")  # Save as a PNG file
    plt.close()  # Close the plot to avoid display if running in a script
    
    # Load the label encoder
    # label_encoder = joblib.load('label_encoder.joblib')
    
    # # Encode the true and predicted labels
    # true_labels_encoded = label_encoder.transform(true_labels)
    # predicted_labels_encoded = label_encoder.transform(predicted_labels)
    
    # # Calculate the average accuracy and weighted F1 score
    # avg_accuracy = accuracy_score(true_labels_encoded, predicted_labels_encoded)
    # weighted_f1 = f1_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
    
    # print(f"Average Accuracy: {avg_accuracy:.4f}")
    # print(f"Weighted F1 Score: {weighted_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run intrusion detection agent.')
    parser.add_argument('--file_path', type=str, default="dataset/CICIoT2023/test_set_small.csv", help='Path to the CSV file containing the traffic data.')
    parser.add_argument('--model_num', type=int, default=6, help='List of models to use for classification.')
    parser.add_argument('--attitude', type=str, default="Balanced", help='Classification attitude (Aggressive, Conservative, Balanced).')

    args = parser.parse_args()
    if args.model_num == 3:
        model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression"]
    if args.model_num == 5:
        model_name = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Random Forest", "Support Vector Classifier"] 
    if args.model_num == 6:
        model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression", "MLP", "Support Vector Classifier","Decision Tree"]
    
    asyncio.run(run_agent(file_path=args.file_path, model_name=model_name, attitude=args.attitude))
