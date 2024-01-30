import os
import config
import openai

from langchain_community.llms import AzureOpenAI


def run_demo():

    #openai.api_key = config.azure_dev["openai_key"] #os.getenv("AZURE_OPENAI_KEY")
    #openai.api_base = config.azure_dev["endpoint"] #os.getenv("AZURE_OPENAI_ENDPOINT") 
    #openai.api_type = 'azure'
    #openai.api_version = '2023-05-15' # this may change in the future

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = config.azure_dev["endpoint"]
    os.environ["OPENAI_API_KEY"] = config.azure_dev["openai_key"]

    llm = AzureOpenAI(
        deployment_name=config.azure_dev["deployment_name"],
        model_name="text-davinci-002",
    )

    demo_llm(llm)


def demo_llm(llm):

    print("Call azureopenai demo_llm")

    result = llm("Tell me a fun fact about Pluto.")
    print(result)
    #print(f"llm_output: {llm.llm_output}")