import re
import json
from typing import Union, Dict

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser

FINAL_ANSWER_ACTION = "Final Answer:"
ACTION_INPUT_REGEX = r"Action\s*:\s*(.*?)\s*Action\s*Input\s*:\s*({.*?})"

class MultiInputReactOutputParser(AgentOutputParser):
    """Parses LLM outputs to handle network traffic classification tasks.
    
    Expects the output to be in the following format:
    
    ```
    I need to follow the steps provided to classify the network traffic from line number 183 as either an attack or normal traffic. I should load the traffic features, preprocess the data, and then use multiple classifiers to determine the final classification.
    
    Action: load_data_line
    Action Input: {"line_number": 183}
    ```
    
    The output will be parsed into an AgentAction with the `Action Input` as a JSON object.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        match = re.search(ACTION_INPUT_REGEX, text, re.DOTALL)
        if match:
            print("find action!!!")
            action = match.group(1).strip()
            print("Action::::", action)
            
            action_input_str = match.group(2).strip()
            print("Action_input str::::", action_input_str)
            # Convert the action input to a JSON object
            try:
                action_input: Dict = json.loads(action_input_str)
            except json.JSONDecodeError as e:
                raise OutputParserException(
                    f"Invalid JSON in Action Input: {action_input_str}",
                    observation=str(e),
                    llm_output=text,
                    send_to_llm=True,
                )

            return AgentAction(
                tool=action,      
                tool_input=action_input,  
                log=text           
            )

        raise OutputParserException(
            f"Could not parse LLM output: `{text}`",
            observation="Invalid format for action and action input.",
            llm_output=text,
            send_to_llm=True,
        )

    @property
    def _type(self) -> str:
        return "network-traffic-classification"



