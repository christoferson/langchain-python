from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "Tell me a fun fact about Pluto."
    prompts = ["Tell me a fun fact about Pluto.", "Tell me a fun fact about Venus."]

    demo_chat(bedrock_runtime, model_id, model_kwargs, prompt)


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

    
