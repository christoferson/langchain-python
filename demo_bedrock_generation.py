import json

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "What is the diameter of Earth?"


    demo_invoke_model_anthropic_claude(bedrock_runtime, model_id)


def demo_invoke_model_anthropic_claude(bedrock_runtime, model_id = "anthropic.claude-v1"):

    print("Call demo_invoke_model_anthropic_claude")

    prompt="""\n\nHuman: What is the diameter of the earth?
        Assistant:
    """

    request = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 300,
        "max_tokens_to_sample": 2048,
        "stop_sequences": []
        }

    response = bedrock_runtime.invoke_model(modelId = model_id, body = json.dumps(request))

    response_body_json = json.loads(response["body"].read())

    print(f"Answer: {response_body_json['completion']}")


def run_demoxxx(bedrock_runtime, model_id):

    prompt_data = """
    Command: Write an email from Bob, Customer Service Manager, to the customer "John Doe" 
    who provided negative feedback on the service provided by our customer support 
    engineer"""

    body = json.dumps({
    "inputText": prompt_data, 
    "textGenerationConfig":{
        "maxTokenCount":2048,
        "stopSequences":[],
        "temperature":0,
        "topP":0.9
        }
    }) 



    modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'
    outputText = "\n"

    response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    outputText = response_body.get('results')[0].get('outputText')

    email = outputText[outputText.index('\n')+1:]
    print(email)