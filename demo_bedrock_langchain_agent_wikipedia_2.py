from langchain_community.vectorstores import Chroma

from langchain_community.chat_models import BedrockChat

from langchain.agents import load_tools, initialize_agent, AgentType

import wikipedia

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    demo_agents_wikipedia_connect()


def demo_agents_wikipedia_connect():

    print("Call demo_agents_wikipedia_connect")

    wikipedia.set_lang("ja")
    search_response = wikipedia.search("Hunter x Hunter")
    print("--------------------------------------")
    print(len(search_response))
    print(search_response)
    print("--------------------------------------")

    if search_response:
        wiki_page = wikipedia.page(search_response[0])
        wiki_content = wiki_page.content[:4000]
        response_string = wiki_content
    else:
        response_string = "その単語は登録されていません。"
    
    print("--------------------------------------")
    print(response_string)
    print("--------------------------------------")

    


