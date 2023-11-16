from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat

from langchain.prompts import PromptTemplate

from langchain.schema.output_parser import StrOutputParser

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"
    model_id = "anthropic.claude-v1"
    model_id = "meta.llama2-13b-chat-v1" #NotImplementedError: Provider meta model does not support chat.
    model_id = "cohere.command-text-v14"

    model_kwargs = { "temperature": 0.0 }
    model_kwargs_meta = { "temperature": 0.0, "top_p" : 0.9, "max_gen_len": 512 }
    prompt = "Tell me a fun fact about Pluto."
    prompts = ["Tell me a fun fact about Pluto.", "Tell me a fun fact about Venus."]

    #demo_invoke_model_meta(bedrock_runtime, model_id, prompt)
    demo_llm(bedrock_runtime, model_id, model_kwargs_meta, prompt)
    #demo_llm_predict(bedrock_runtime, model_id, model_kwargs, prompt)
    #demo_llm_generate(bedrock_runtime, model_id, model_kwargs, prompts)
    #demo_llm_chain(bedrock_runtime, model_id, model_kwargs, prompt)

def demo_invoke_model_meta(bedrock_runtime, model_id, prompt):

    print(f"Call demo_invoke_model_meta model_id={model_id} prompt={prompt}")

    request = {
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_gen_len": 512
    }

    response = bedrock_runtime.invoke_model(modelId = model_id, accept='application/json', contentType='application/json', body = json.dumps(request))

    response_body_json = json.loads(response["body"].read())
    print("*************************************************")
    print(response_body_json)
    completion = response_body_json['generation']
    print("*************************************************")
    print(f"Answer: {completion}")
    print("*************************************************")

