import langchain
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain.cache import InMemoryCache

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

    #demo_chat(bedrock_runtime, model_id, model_kwargs, prompt)
    demo_chat_with_system_message(bedrock_runtime, model_id, model_kwargs, prompt, system_message)
    #demo_chat_generate_with_system_message(bedrock_runtime, model_id, model_kwargs, prompt, system_message)


def demo_chat(bedrock_runtime, model_id, model_kwargs, prompt):

    print(f"Call Bedrock demo_chat")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        HumanMessage(content = prompt)
    ])

    print(result.content)

    

def demo_chat_with_system_message(bedrock_runtime, model_id, model_kwargs, prompt, system_message):

    print(f"Call Bedrock demo_chat_with_system_message")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    result = llm([
        SystemMessage(content = system_message),
        HumanMessage(content = prompt)
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