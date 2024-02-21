from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat

from langchain.prompts import PromptTemplate

from langchain.schema.output_parser import StrOutputParser

import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    #demo_invoke_model_meta(bedrock_runtime, model_id, prompt)
    
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

