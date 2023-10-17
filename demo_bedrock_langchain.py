from langchain.llms import Bedrock

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "Tell me a fun fact about Pluto."

    demo_llm(bedrock_runtime, model_id, model_kwargs, prompt)


def demo_llm(bedrock_runtime, model_id, model_kwargs, prompt):

    print("Call demo_llm")

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm(prompt) #output = llm.predict(prompt)

    print(output)
