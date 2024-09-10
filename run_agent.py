<<<<<<< HEAD
# from llm_agents import Agent, ChatLLM, PythonREPLTool, SerpAPITool, GoogleSearchTool, HackerNewsSearchTool
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
from langchain.agents import AgentExecutor
from langchain_community.utilities import WikipediaAPIWrapper
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from langchain_experimental.tools import PythonREPLTool
# from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
import re
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import argparse
import os
with open('openai_key.txt', 'r') as file:
    # Read the first line from the file
    first_line = file.readline()
    os.environ["OPENAI_API_KEY"] = first_line


# os.environ["GOOGLE_CSE_ID"] = ""
# os.environ["GOOGLE_API_KEY"] = ""
=======
from llm_agents import Agent, ChatLLM, PythonREPLTool, SerpAPITool, GoogleSearchTool, HackerNewsSearchTool



>>>>>>> 101711d2b704910742fc4eda7ce368b674c43e15
# DataProcessingTool
# Context extraction
# FeatureSelectionTool
# Scikit-learn API /Pretrain model Tool
# huggingface model
# Arkiv Tool
# 结合Snort，Suricata进行实时流量监控和入侵响应。
<<<<<<< HEAD
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

class DataLineInput(BaseModel):
    file_path: str = Field(description="should be a file path")
    flow_id: str = Field(description="should be a flow ID")

@tool("load_data_line", args_schema = DataLineInput, return_direct=True)
async def load_data_line(file_path: str, flow_id: str) -> str:
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

# load_data_line.invoke({"file_path":"/Users/yanjieli/program/IDS_AGENT/dataset/ACIIoT/test_set_small.csv",
#                         "line_number": 1})

class DataProcessingInput(BaseModel):
    # traffic_features: str = Field(description="The input is a full traffic features string, including multiple features.")
    file_path: str = Field(description="should be a file path")
    flow_id: str = Field(description="should be a flow ID")


@tool("data_preprocessing", args_schema = DataProcessingInput, return_direct=True)
async def data_preprocessing(file_path: str, flow_id: str) -> list:
    """input flow_id and output preprocessed traffic features"""
    data = pd.read_csv(file_path)
    columns_to_drop = [ "Label"]
    data.drop(columns=columns_to_drop, inplace=True)
    record = data[data["Flow ID"] == flow_id]
    
    """Preprocess the traffic record by selecting and scaling the features. This should be done after loading the data."""
    df = record
    # df = string_to_dataframe(traffic_features)
    # Drop unnecessary columns
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Connection Type']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Handle missing values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    selector = joblib.load('/Users/yanjieli/program/IDS_AGENT/models/selector.joblib')
    scaler = joblib.load('/Users/yanjieli/program/IDS_AGENT/models/scaler.joblib')
    
    # Transform the input data
    X_selected = selector.transform(df)
    X_scaled = scaler.transform(X_selected)
    
    return X_scaled.tolist()


class ClassifyInput(BaseModel):
    modelname: str = Field(description="This is a classfication model name")
    traffic_features: list = Field(description="This is the preprocessed data to be classified")
    
@tool("classify", args_schema = ClassifyInput, return_direct=True)
async def classifier(modelname: str, traffic_features: list) -> str:
    """Classify the traffic features using a machine learning model to determine if the traffic record is an attack."""
    # Load the model
    model = joblib.load(f"models/{modelname}.joblib")
    
    # Make predictions
    if get_list_dim(traffic_features) == 1:
        traffic_features = [traffic_features]
    predictions = model.predict(np.array(traffic_features))
    label_encoder = joblib.load('models/label_encoder.joblib')
    string_predictions = label_encoder.inverse_transform(predictions)
    return string_predictions.tolist()[0]

# example = load_data_line("/Users/yanjieli/program/IDS_AGENT/dataset/ACIIoT/test_set_small.csv", 1)
# X_scaled = data_preprocessing(example)
# pred = classify("Decision Tree", X_scaled)
# print(pred)



async def run_agent(file_path = "dataset/ACIIoT/test_set_small.csv",
                    model_name = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression"],
                    attitude ="Balanced"):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
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
   
    # Determine the number of lines in the file
    df = pd.read_csv(file_path)
    labels  = df["Label"].unique()

    true_labels = []
    predicted_labels = []
    flow_ids = []
    for line_number in range(len(df)):
        # Define the prompt for the agent
        flow_id = df.at[line_number,"Flow ID"]
        # If there is a significant discrepancy in the model predictions, consider the possibility of `Unknown` attacks. 
        # if df.at[line_number,"Label"] != "UDP Flood":
        #     continue

        if attitude == "Aggresive":
            attitude_details = "discover the attack at the first time"
        if attitude == "Conservative":
            attitude_details = "donot alert unless you are very sure"
        if attitude == "Balanced":
            attitude_details = "balance the false alarm rate and the missing alarm rate"
        
        prompt_text = f"""
        Is the network traffic from the file {file_path} with the Flow ID {flow_id} an attack or normal traffic? 
        You are tasked with performing intrusion detection. Follow these steps for the detection process:

        1. Load the traffic features from the CSV file using tools. You can use the load_data_line tool to obtain the complete traffic_string.
        2. Select features and normalize the full feature strings using tools. This can be done using data_preprocessing tool.
        3. Finally, load classifiers for classification.
        You should classify this traffic until you reach a final classification. You can use multiple classifiers, including {model_name}. 
        At the end, you should summarize the results from these classifiers and provide a final result.
        Summarize the classification with {attitude} attitude, which means {attitude_details}.
        The final output format should be: **Final Classification: Classification results, such as Benign or UDP Flood**
        """
        
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

        # Initialize the LLM
        # os.environ["GOOGLE_CSE_ID"] = "66297c07f31dc4c6d"
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDE0ErQezoMG9H33i7IZ6x663yLzUe7_hA"
        # google_search = GoogleSearchAPIWrapper(k=1)
        # search_tool = Tool(
        #     name="google_search",
        #     description="Search Google for recent results.",
        #     func=google_search.run,
        # )
        # Get the prompt
        # prompt = hub.pull("hwchase17/openai-functions-agent")
        # prompt = hub.pull("hwchase17/openai-tools-agent")
        # print(prompt.messages) -- to see the prompt

        
        
        # api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
        # wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

            # tools = [load_data_line, data_preprocessing, classify, PythonREPLTool(), google_search_tool]
        tools = [load_data_line, data_preprocessing, classifier]

        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True,  handle_parsing_errors=True)


        # Run the agent with the provided prompt
        # response = agent_executor.invoke({"input": prompt_text})
        # print(response)

        chunks = []
        responses = []

        async for chunk in agent_executor.astream({"input": prompt_text}):
            chunks.append(chunk)
            print("------")
            # print(chunk["messages"])
            if "actions" in chunk:
                for action in chunk["actions"]:
                    print(f"Calling Tool: {action.tool} with input {action.tool_input}")
            elif "output" in chunk:
                response = chunk["output"]
                print(f'Final Output: {chunk["output"]}')
                            # Regular expression to find the final classification
                match = re.search(r"Final Classification\s*(.*?)\n", response)
                if match:
                    final_classification = match.group(1)
                    for label in labels:
                        if label in final_classification:
                            print(f"Final Classification: {final_classification}")
                            predicted_labels.append(label)
                            data = pd.read_csv(file_path)
                            data_line = data[data["Flow ID"] == flow_id]
                            true_label = data_line['Label'].values[0]
                            true_labels.append(true_label)
                            flow_ids.append(flow_id)
                            # Write to the file
                            with open("results/classification_results_0815.txt", "a") as f_class:
                                f_class.write(f"Flow ID: {flow_id}, True Label: {true_label}, Predicted Label: {final_classification}\n")
                else:
                    print("Final Classification not found.")
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"Tool Result: {step.observation}")
            else:
                raise ValueError()
            print("---")

        with open ("response_results.txt","w") as f:    
            f.write(str(flow_id) + "\n")
            f.write(str(response) + "\n")
            

    # Compute the confusion matrix
    # Remove the first row and first column

    
    cm = confusion_matrix(true_labels, predicted_labels)    
    print(cm)   
    # print(classification_report(true_labels, predicted_labels, target_names=np.unique(true_labels)))
    with open("classification_results.txt", "a") as f:
        f.write("\nSummary:\n")
        f.write(f"Total Records: {len(true_labels)}\n")
        
        # Write the confusion matrix to the file
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')
        
    modified_confusion_matrix = cm[1:, 1:] # remove the APR attack
    labels = np.unique(true_labels)
    modified_labels = labels[1:]

    # Generate synthetic y_true and y_pred based on the modified confusion matrix
    y_true = []
    y_pred = []

    for i, row in enumerate(modified_confusion_matrix):
        for j, count in enumerate(row):
            y_true.extend([modified_labels[i]] * count)
            y_pred.extend([modified_labels[j]] * count)

    # Calculate and print the classification report
    report = classification_report(y_true, y_pred, target_names=modified_labels)
    print(report)   
    
        
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
    parser.add_argument('--file_path', type=str, default="dataset/ACIIoT/test_set_small.csv", help='Path to the CSV file containing the traffic data.')
    parser.add_argument('--model_num', type=int, default=5, help='List of models to use for classification.')
    parser.add_argument('--attitude', type=str, default="Aggressive", help='Classification attitude (Aggressive, Conservative, Balanced).')

    args = parser.parse_args()
    if args.model_num == 3:
        model_name = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression"]
    if args.model_num == 5:
        model_name = ["Decision Tree", "K-Nearest Neighbors", "Logistic Regression", "Random Forest", "Support Vector Classifier"] 
    
    asyncio.run(run_agent(file_path=args.file_path, model_name=model_name, attitude=args.attitude))
=======
#调用XAI解释
# knowledge 用来解释分类结果

if __name__ == '__main__':
    # prompt = input("Enter a question / task for the agent: ")
    prompt = '''You are a cyber security expert and ask to do intrusion detection based on provided files. 
                The files is at '/home/yanjieli/IDS_Agent/train_set.csv'.
                You should follow the following steps: 
                1. Data processing. You should write a python code to process the traffic data. For example, remove the nan value and ID field.
                2. Select important features. You can use online papers/website to collect knowledge.
                3. Load multiple classifier models. Here are pretrained ML models.
                4. Real time classfication, load the test file line by line.
                4. Based on the classfication results, explain the classfication reasons if the example is be classified as malicous.'''

    agent = Agent(llm=ChatLLM(), tools=[DataProcessingTool(), PythonREPLTool(), SerpAPITool(), GoogleSearchTool(), HackerNewsSearchTool()])
    result = agent.run(prompt)

    print(f"Final answer is {result}")
>>>>>>> 101711d2b704910742fc4eda7ce368b674c43e15
