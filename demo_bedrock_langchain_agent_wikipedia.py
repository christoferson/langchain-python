import os
import config

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat

from langchain.agents import load_tools, initialize_agent, AgentType

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    #demo_agents_wikipedia_connect()
    demo_agents_wikipedia(bedrock_runtime, prompt="When was Albert Einstein born? What is the year multiplied by 5?")
    #demo_agents_wikipedia(bedrock_runtime, prompt="When was Hunter x Hunter first aired? What is that year multiplied by 5?")
    #demo_agents_wikipedia_2(bedrock_runtime, prompt="When was Hunter x Hunter first aired? Was it the same year as when Fairy Tale was aired?")


def demo_agents_wikipedia_connect():

    print("Call demo_agents_wikipedia_connect")

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    result = wikipedia.run("HUNTER X HUNTER")

    print(result)


def demo_agents_wikipedia(bedrock_runtime,
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        prompt = ""):

    print("Call demo_agents_wikipedia")


    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    #result = wikipedia.run("HUNTER X HUNTER")

    #print(result)

    tools = load_tools(['llm-math', 'wikipedia'], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True, max_iterations = 2)

    result = agent.run(prompt)

    print(result)



def demo_agents_wikipedia_2(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        prompt = ""):

    print("Call demo_agents_wikipedia_2")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    #wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    tools = load_tools(['wikipedia'], llm=llm) #load_tools(['llm-math', 'wikipedia'], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True, max_iterations = 2)

    result = agent.run(prompt)

    print(result)
