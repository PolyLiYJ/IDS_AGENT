from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

hf = HuggingFacePipeline.from_model_id(
    model_id="jondurbin/airoboros-l2-7b-gpt4-m2.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))