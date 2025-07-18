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
with open('openai_key.txt', 'r') as file:
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

from react_parser import MultiInputReactOutputParser
from scipy.spatial.distance import euclidean
from functools import partial

import json

MEMORY_PATH = "longmemory/success_example_test_set0910.json"

def write_success_example_to_log(text_list, filename):
    # Name of the file to write
    # file_name = "results/Recon_success_examples.txt"

    # Open the file in write mode
    with open(filename, 'a') as file:
        # Write each string from the list to the file
        file.write(text_list + '\n')
        file.write("**************************************")

    print(f"Strings have been written to {filename}")

def custom_error_handler(error) -> str:
    if error is None:
        return "Received None as an Action Input, please provide a valid input."
    else:
        return str(error)

def load_traffic(filepath, line_number):
    # file_path = "/Users/yanjieli/program/IDS_AGENT/dataset/CICIoT2023/test_set_small.csv"
    data = pd.read_csv(filepath)
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
        print("No record for " + str(line_number))

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
    
def retriever_qa_creation(folder, chroma_db_database, init = False):
    kb = MyKnowledgeBase(pdf_source_folder_path=folder)
    # ollama_model_name = "llama2:7b-chat-q6_K"
    # embedder = OllamaEmbeddings(model=ollama_model_name)
    # chroma_db_database = 'db-llama'
    # retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    
    # vector_db = kb.initiate_document_injection_pipeline(embedder)
    # # retriever = vector_db.as_retriever()

    # llm = Ollama(model=ollama_model_name) #, temperature=0.1)
    
    embedder = OpenAIEmbeddings()
    if init:
        kb.initiate_document_injection_pipeline(embedder, chroma_db_database)
    retriever = kb.return_retriever_from_persistant_vector_db(embedder, chroma_db_database)
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff", retriever = retriever)
    return qa

retriever_qa_chain = retriever_qa_creation(folder = "/Users/yanjieli/program/IDS_AGENT/documents/Recon", chroma_db_database = "db" )
retriever_success_samples = retriever_qa_creation(folder = "/Users/yanjieli/program/IDS_AGENT/longmemory/text", chroma_db_database = "db-long-memory", init=False  )

def custom_sort_criteria(action_list, feature):
    """
    Extracts the 'thought' and 'observation' from entries in the output_list where the step is 'data_preprocessing'.
    
    Parameters:
    output_list (list): The list of outputs containing processed features and other information.
    
    Returns:
    list: A list of dictionaries containing 'thought' and 'observation' for data preprocessing steps.
    """
    preprocessing_steps = []

    # Loop through each entry in the output_list
    for entry in action_list:
        if "action" in entry and entry["action"]["tool"] == "classifier":
            # Extract the thought and observation for data preprocessing
            return euclidean(entry["action"]["tool_input"]["traffic_features"], feature)
        
def extract_actions(json_data):
    actions_list = []
    
    for index, step in enumerate(json_data):
        if 'action' in step:
            action_info = f"Action: {step['action']['tool']}"
            if step['action']['tool'] == 'knowledge_retriever':
                action_info += f", Query: {step['action']['tool_input']['query']}"
                # action_info += f", Observation: {json_data[index+1]['observation']}"
            if step['action']['tool'] == 'classifier':
                action_info += f", model name: {step['action']['tool_input']['modelname']}"
            actions_list.append(action_info)
    
    return actions_list

class find_closest_samples_Input(BaseModel):
    processed_feature_list: list = Field(description="The processed features to compare.")
    num_samples: int =  Field(description="The number of closest samples to return (default is 3).")
    
@tool("load_memory", args_schema=find_closest_samples_Input, return_direct=False)
async def load_memory(processed_feature_list:list, num_samples:int) -> list:
    """
    Finds the closest samples from the memory based on the Euclidean distance 
    between processed features and returns their details.
    
    Parameters:
    processed_feature_list (list): The processed features to compare.
    num_samples (int): The number of closest samples to return (default is 5).
    
    Returns:
    list: A list of dictionaries containing the closest samples with their details.
    """

    # Initialize an empty list to store the distances


    # Open the JSON file and load the list
    with open(MEMORY_PATH, "r") as file:
        my_list = json.load(file)
    new_list = []
    for i in range(len(my_list)):
        if custom_sort_criteria(my_list[i],processed_feature_list) is not None:
            new_list.append(my_list[i])
        
    partial_sort_criteria = partial(custom_sort_criteria, feature=processed_feature_list)    
    sorted_list = sorted(new_list, key=partial_sort_criteria)

    # Get the closest `num_samples` entries
    closest_samples = sorted_list[:num_samples]

    # Create a list to store the results
    results = []

    # Loop through the closest samples and extract the required fields
    for sample in closest_samples:
        result = {
            "processed_features": sample[3]["observation"],
            "action trace":extract_actions(sample),
            "final_answer": sample[-1]["final_classification"]
        }
        results.append(result)

    return results

# print(load_memory.invoke({"processed_feature_list":[-0.1517, -0.3088, -0.5131, -0.3166, -0.3113, 2.6909, 1.0781, 0.5586, -0.3267, 0.8611, -0.5179, -0.4433, -0.0022, -0.2031, 0.2497, 1.6121, 0.7173, 0.5773, 1.6121, 3.5835],"num_samples":3}))


from langchain_text_splitters import CharacterTextSplitter

@tool
async def knowledge_retriever(query:str)->str:
    """search the knowledge of different attacks, classifiers, or information about the network"""
    return retriever_qa_chain.run(query)


@tool
async def long_memory_retriever(query:str)->str:
    """search 1-5 related previous sucess examples, return the examples in including the original features, preprocess features, analysis and finial predictions."""
    return retriever_success_samples.run(query)

class DataIDInput(BaseModel):
    file_path: str = Field(description="should be a file path")
    flow_id: str = Field(description="should be a flow ID")

@tool("load_data_ID", args_schema = DataIDInput, return_direct=False)
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
    file_path: str = Field(description="The file path to the CSV file")
    line_number: int = Field(description="The line number in the CSV file")
    
@tool("data_loader", args_schema=DataLineInput, return_direct=False)
async def data_loader(file_path: str, line_number: int) -> str:
    """Load traffic features from the the file_path and a specified line_number in a CSV file."""
    # Read the CSV file
    # file_path = "/Users/yanjieli/program/IDS_AGENT/dataset/CICIoT2023/test_set_small.csv"
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
    traffic_features: str = Field(description="The input is a full traffic string, such as 'flow_duration: 1.633075261, Header_Length: 5198.7, Protocol Type: 6.6, ..., Weight: 244.6'")
    # file_path: str = Field(description="should be a file path")
    # line_number: int = Field(description="should be a line_number")


@tool("data_preprocessing", args_schema = DataProcessingInput, return_direct=False)
async def data_preprocessing(traffic_features: str) -> list:
    """input traffic_features and output preprocessed traffic features as a list
    """
    # file_path = "/Users/yanjieli/program/IDS_AGENT/dataset/CICIoT2023/test_set_small.csv"
    # data = pd.read_csv(file_path)
    # columns_to_drop = [ "label", "final_label"]
    # data.drop(columns=columns_to_drop, inplace=True)
    # record = data.iloc[[line_number]]
    record = string_to_dataframe(traffic_features)
    
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
    
    return X_scaled.tolist()[0]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    return exp_x / exp_x.sum(axis=1, keepdims=True)

class ClassifyInput(BaseModel):
    modelname: str = Field(description="This is a classfication model name")
    traffic_features: list = Field(description="This is the preprocessed data to be classified")
    top_k: int = Field(default=3, description="Number of top predictions to return")
    
@tool("classifier", args_schema=ClassifyInput, return_direct=False)
async def classifier(modelname: str, traffic_features: list, top_k: int = 3) -> str:
    """Classify the traffic features using a machine learning model to determine if the traffic record is an attack.
    params:
        modelname
        traffic_features
    """
    
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
            ("system", """
             You are a helpful assisstant. You are ask to do intruction dection using machine learning models and web knowledge..          
             You should firstly generate a plan to load, process and classify the data, and then excute the plan step by step.
             For each step, you should call one tool from the tools.
             ```json
                {{
                    'line_number': line_number, 
                    'analysis':  str,  \here is the Analysis, 
                    'predicted_label_top_1': str,
                    'predicted_label_top_2': str, 
                    'predicted_label_top_3': str, 
                }}
            ```
             """),
            ("placeholder", "{chat_history}"), #important
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    return prompt
    

def chat_json_prompt():
    system = '''Assistant is a large language model.
    Assistant is designed to be able to assist with network intrusion detection task, 
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
        "action_input": \ The input params to the fuction, should be in json format
    }}
    ```

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    ```json
    {{
        "action": "Final Answer",
        "action_input": {{
            'line_number': line_number, 
            'analysis':  str,  \here is the Analysis, 
            'predicted_label_top_1': str,
            'predicted_label_top_2': str, 
            'predicted_label_top_3': str, 
        }} \ You should put what you want to return to use here
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
    template = '''You are a helpful assisstant. You are ask to do intruction dection using machine learning models and web knowledge.          
    You should firstly generate a plan to load, process and classify the data, and then excute the plan step by step.
    ### Plan Example:
    # 1. Load the traffic features from the CSV file using tools. You can use the load_data_line tool to obtain the complete traffic_string.
    # 2. Select features and normalize the full feature strings using tools. This can be done using data_preprocessing tool.
    # 3. Finally, load classifiers for classification. I will use multiple classifiers. 
    # 4. At the end, I summarize the results from these classifiers and provide a final result.
    The final output format is 
    Final Answer:
    ``` json
    {{
        'line_number': line_number, 
        'analysis':  str,  \here is the Analysis, 
        'predicted_label_top_1': str,
        'predicted_label_top_2': str, 
        'predicted_label_top_3': str, 
    }}
    ```
    You have access to the following tools:

    {tools}

    Strictly Use the following format (Donot add any sign like # or * before and after the Thought/Action/Action Input/Observation):

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action, should be a json format
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question, should be in a json format including line_number, analysis and prediction.
    
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}

    '''

    prompt = PromptTemplate.from_template(template)
    return prompt
    
    
async def run_agent(file_path = "dataset/CICIoT2023/test_set_small.csv",
                    model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression", "MLP", "Support Vector Classifier","Decision Tree"],
                    attitude ="Balanced",
                    #llm_model_name = "gpt-3.5-turbo-0125"
                    llm_model_name = "gpt-4o-mini",
                    memory_path = "longmemory/success_example_test_set0910.json"
                    ):

    llm = ChatOpenAI(model=llm_model_name, temperature=0)

    # When there are discrepancy/disagreements for different models, 
    # you can search from the google or Wiki to get more information about the difference of these attacks to help you to make decision.
    # Determine the number of lines in the file
    df = pd.read_csv(file_path)
    labels  = df["label"].unique()

    true_labels = []
    predicted_labels = []
    flow_ids = []
    
    # predicts_analysis = []
    with open("results/classification_results_CICIoT_0910.txt", "r") as file:
        predict_analysis = json.load(file)
    predicted_index = []
    for i in range(0, len(predict_analysis)):
        predicted_index.append(predict_analysis[i]["line_number"])
    linenumbers = [i for i in range(len(df)) if i not in predicted_index]
    print(linenumbers)
    
    
    # define agent
    os.environ["GOOGLE_CSE_ID"] = "66297c07f31dc4c6d"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDE0ErQezoMG9H33i7IZ6x663yLzUe7_hA"
    google_search = GoogleSearchAPIWrapper(k=5)

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
    tools = [data_loader, data_preprocessing, classifier, knowledge_retriever, search_tool, wiki_tool, load_memory]
    message_history = RedisChatMessageHistory(
        url="redis://127.0.0.1:6379/0", ttl=600, session_id="0904"
    )
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # prompt = hub.pull("hwchase17/react-chat-json")
    # prompt = chat_json_prompt()
    # agent = create_json_chat_agent(llm, tools, prompt)
    
    # prompt = ids_system_prompt()
    # agent = create_tool_calling_agent(llm, tools, prompt)
    
    # prompt = react_prompt()
    # agent = create_react_agent(llm, tools, prompt, stop_sequence = True)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps = True, handle_parsing_errors=True)
    
    # agent_with_chat_history = RunnableWithMessageHistory(
    #     agent_executor,
    #     # This is needed because in most real world scenarios, a session id is needed
    #     # It isn't really used here because we are using a simple in memory ChatMessageHistory
    #     lambda session_id: message_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )
    
    # few_shot_example = "Here is some examples :\n" +load_few_shot("few_shot.txt")
    few_shot_example = ""
    
    react_log = []
    # load memory
    with open(memory_path, "r") as file:
        success_list = json.load(file)
    with open("longmemory/all_example_test_set0910.json", "r") as file:
        final_list = json.load(file)

    for line_number in linenumbers:
        react_log_example = ""
        
        # Define the prompt for the agent
        # flow_id = df.at[line_number,"Flow ID"]
        # If there is a significant discrepancy in the model predictions, consider the possibility of `Unknown` attacks. 
        
        if attitude == "Aggresive":
            attitude_details = "discover the attack at the first time"
        if attitude == "Conservative":
            attitude_details = "donot alert unless you are very sure"
        if attitude == "Balanced":
            attitude_details = "balance the false alarm rate and the missing alarm rate"
        
        prompt_text = f"""
        You are a helpful assisstant that can implement multi-step tasks, such as intrusion detection. I will give you the traffic features, you are asked to classfiy it using tools.
        The final output mush be in a json format according to the classifier results. You should follow the plan:
        1. Load the traffic features from the CSV file using tools. You can use the data_loader tool to obtain the complete traffic_string.
        2. Preprocessing the feature. This can be done using data_preprocessing tool. Input the tarffic string in original format.
        3. load the previous sucess examples from the long memory. This can be done using the load_memory tool, return the top 2 results.
        3. load classifiers for classification.This can be done using classifier tool. You can use multiple classifiers, 
        The tool params include the classifier name, which must be one from {model_name}.  and the preprocessed features.
        4. When there are discrepancy/disagreements for different models or you are unsure about the classification results, search from long_memory_retriver\knowledge_retriever\Wikipedia\Google to get more information to help you to make decision.
        There are some examples:
            1)You can first search from the long memory using the long_memory_retriever tool, by query like: "what is the features of xxx attacks"
            2)You can call the knowledge_retriever google or wiki tool 'what is xxx and how to detect it' and 'what is the difference about a potential xxx attack and benign example'.
            3)You can call the knowledge_retriever `what features push the prediction to recon` if you cannot distinguish benign and recon attack.
            4)When one model classify the data as attack A, and another classify the data as attack B, you can search on knowledge_retriever for "The features of A,and the features of B and their difference".
        5. At the end, you should summarize the results from these classifiers and provide a final result.
        Summarize the classification with {attitude} sentitivity, which means {attitude_details}.
        The predicted_label should be original format of classifier prediction.
        The final output format **must** be: 
        Final Answer:
        ```json
        {{
            'line_number': \line_number, should be same with the line number of dataloder's input
            'analysis':  str,  \here is the Analysis, 
            'predicted_label_top_1': str,
            'predicted_label_top_2': str, 
            'predicted_label_top_3': str, 
        }}
        ```
        {few_shot_example}
        
        Now, classsify the traffic from line_number {line_number} in the file {file_path}
        """
        
        print(prompt_text)

        chunks = []
        output_list = []

        async for chunk in agent_executor.astream({"input": prompt_text}):
            chunks.append(chunk)
            current_output = {} 
            # print(chunk)
            print("------")
            # print(chunk["messages"])
            if "actions" in chunk:
                for action in chunk["actions"]:
                    react_log_example += "\n" + "Thought:" + action.log
                    current_output["action"] = {
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                    }
                    current_output["Thought"] = action.log
            elif "output" in chunk:
                response = chunk["output"]
                # Regular expression to find the final classification
                # Extract the JSON block from the text
                start = response.find("```json") + len("```json")
                end = response.find("```", start)
                match = response[start:end].strip()
                if response.find("```json") >= 0:
                    pred_dict = json.loads(match)
                    # f.write(str(pred_dict) + "\n")
                    final_classification = pred_dict['predicted_label_top_1']
                    print(f"Final Classification: {final_classification}")
                    predicted_labels.append(final_classification)
                    data_line = df.iloc[line_number]
                    true_label = data_line['label']
                    true_labels.append(true_label)
                    pred_dict["true_label"] = true_label
                    predict_analysis.append(pred_dict)
                    current_output["final_classification"] = final_classification
                    current_output["true_label"] = true_label
                    current_output["prediction_details"] = pred_dict
                    
                    with open("results/classification_results_CICIoT_0910.txt", "w") as f_class:
                        json.dump(predict_analysis, f_class, indent=4)
                    # print(f"line_number: {line_number}, True Label: {true_label}, Predicted Label: {final_classification}\n")
                    if true_label == final_classification:
                        react_log_example += "Final Answer:" + response
                        write_success_example_to_log(react_log_example, "results/success_examples0910.txt")
                else:
                    print("Final Classification not found.")
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"Tool Result: {step.observation}")
                    # f.write(f"Tool Result: {step.observation}" + "\n")
                    react_log_example += "\n Observation: " + str(step.observation)
                    current_output["observation"] = step.observation
            else:
                raise ValueError()
            print("--------------")
        
            output_list.append(current_output)
            
        if true_label == final_classification:
            success_list.append(output_list)
        # final_list.append(output_list)
            # Save the entire output as JSON after processing all chunks
        json_output_path = memory_path
        with open(json_output_path, "w") as json_file:
            json.dump(success_list, json_file, indent=4)    
        json_output_path = "longmemory/all_example_test_set0910.json"
        with open(json_output_path, "w") as json_file:
            json.dump(final_list, json_file, indent=4)      


    # Compute the confusion matrix
    # Remove the first row and first column

    
    cm = confusion_matrix(true_labels, predicted_labels)    
    print(cm)   
    # Calculate and print the classification report
    report = classification_report(true_labels, predicted_labels, target_names=labels)
    print(report)   
    # f.write(report)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
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
    parser.add_argument('--memory_path', type=str, default="longmemory/success_example_test_set0910.json", help='Path to the CSV file containing the traffic data.')
    parser.add_argument('--model_num', type=int, default=6, help='List of models to use for classification.')
    parser.add_argument('--attitude', type=str, default="Balanced", help='Classification attitude (Aggressive, Conservative, Balanced).')

    args = parser.parse_args()
    if args.model_num == 3:
        model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression"]
    if args.model_num == 5:
        model_name = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Random Forest", "Support Vector Classifier"] 
    if args.model_num == 6:
        model_name = ["Random Forest", "K-Nearest Neighbors", "Logistic Regression", "MLP", "Support Vector Classifier","Decision Tree"]
    
    asyncio.run(run_agent(file_path=args.file_path, model_name=model_name, attitude=args.attitude, memory_path = args.memory_path))
    # run_agent(file_path=args.file_path, model_name=model_name, attitude=args.attitude)
