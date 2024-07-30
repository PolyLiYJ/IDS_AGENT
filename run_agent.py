from llm_agents import Agent, ChatLLM, PythonREPLTool, SerpAPITool, GoogleSearchTool, HackerNewsSearchTool



# DataProcessingTool
# Context extraction
# FeatureSelectionTool
# Scikit-learn API /Pretrain model Tool
# huggingface model
# Arkiv Tool
# 结合Snort，Suricata进行实时流量监控和入侵响应。
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
