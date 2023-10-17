import langchain
from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain.cache import InMemoryCache

from langchain import PromptTemplate

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    # Enable Cache 
    langchain.llm_cache = InMemoryCache()

    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "Tell me a fun fact about Pluto."
    prompts = ["Tell me a fun fact about Pluto.", "Tell me a fun fact about Venus."]
    system_message = "You are a 3 year old kid who is an introvert."

    #demo_prompt_template_no_input(bedrock_runtime, model_id, model_kwargs)
    demo_prompt_template_single_input(bedrock_runtime, model_id, model_kwargs)


def demo_prompt_template_no_input(bedrock_runtime, model_id, model_kwargs):

    print(f"Call Bedrock demo_prompt_template_no_input")

    prompt_template = PromptTemplate(input_variables = [], template = "Tell me a fact.")
    #print(prompt_template.format())

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        HumanMessage(content = prompt_template.format())
    ])

    print(result.content)

    

def demo_prompt_template_single_input(bedrock_runtime, model_id, model_kwargs):

    print(f"Call Bedrock demo_prompt_template_single_input")

    prompt_template = PromptTemplate(input_variables = ["Topic"], template = "Tell me a fact about {Topic}.")
    #print(prompt_template.format())

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        HumanMessage(content = prompt_template.format(Topic="Skylark"))
    ])

    print(result.content)


def demo_chat_generate_with_system_message(bedrock_runtime, model_id, model_kwargs, prompt, system_message):

    print(f"Call Bedrock demo_chat_generate_with_system_message model_id={model_id}")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm.generate([
        [ SystemMessage(content = system_message), HumanMessage(content = prompt) ]
    ])

    for generation in result.generations:
        print(generation[0].text)
        print("")