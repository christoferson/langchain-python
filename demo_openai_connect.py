import os
import requests
import json
import openai
import config

from langchain_community.llms import AzureOpenAI

def run_demo():

    openai.api_key = config.azure_dev["openai_key"] #os.getenv("AZURE_OPENAI_KEY")
    openai.api_base = config.azure_dev["endpoint"] #os.getenv("AZURE_OPENAI_ENDPOINT") 
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future

    deployment_name = config.azure_dev["deployment_name"] #This will correspond to the custom name you chose for your deployment when you deployed a model. 

    # Send a completion call to generate an answer
    print('Sending a test completion job')
    start_phrase = 'Write a tagline for an ice cream shop. '
    response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=50)
    text = response['choices'][0]['text'].replace('\n', ' ').replace(' .', '.').strip()
    print(start_phrase + " : " + text)

    # Azure OpenAI

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    #os.environ["OPENAI_API_BASE"] = "..."
    os.environ["OPENAI_API_KEY"] = config.azure_dev["openai_key"]

    llm = AzureOpenAI(
        deployment_name=config.azure_dev["deployment_name"],
        model_name="text-davinci-002",
    )
    result = llm("Tell me a fun fact about Pluto.")
    print(result)
    #print(f"llm_output: {llm.llm_output}")
