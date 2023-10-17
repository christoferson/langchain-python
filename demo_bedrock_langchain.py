from langchain.llms import Bedrock

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "Tell me a fun fact about Pluto."
    prompts = ["Tell me a fun fact about Pluto.", "Tell me a fun fact about Venus."]

    #demo_llm(bedrock_runtime, model_id, model_kwargs, prompt)
    #demo_llm_predict(bedrock_runtime, model_id, model_kwargs, prompt)
    demo_llm_generate(bedrock_runtime, model_id, model_kwargs, prompts)


def demo_llm(bedrock_runtime, model_id, model_kwargs, prompt):

    print("Call demo_llm")

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm(prompt)

    print(output)

def demo_llm_predict(bedrock_runtime, model_id, model_kwargs, prompt):

    print("Call demo_llm_predict")

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm.predict(prompt)

    print(output)

def demo_llm_generate(bedrock_runtime, model_id, model_kwargs, prompts):

    print("Call demo_llm_generate")

    llm = Bedrock(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    output = llm.generate(prompts)

    #print(output.schema())
    print(output.llm_output)

    for generation in output.generations:
        print(generation[0].text)
        print("")

    
