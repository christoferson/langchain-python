import os
import config

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain, SequentialChain

from langchain.text_splitter import CharacterTextSplitter


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.agents import load_tools, initialize_agent, AgentType, Tool

from langchain.chains import LLMMathChain
from langchain_community.utilities import SerpAPIWrapper

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    serpapi_api_key = config.serpapi["api_key"]
    # Set SerpApi ApiKey
    os.environ["SERPAPI_API_KEY"] = serpapi_api_key

    demo_agents_tools_defnition_method1(bedrock_runtime, prompt="When was Albert Einstein born? What is the year multiplied by 5?")
    #demo_agents_serpapi(bedrock_runtime, prompt="What is the temperature in Tokyo today? What is the temperature multiplied by 5?")
    #demo_agents_serpapi(bedrock_runtime, prompt="When was Albert Einstein born? What is the year multiplied by 5?")


def demo_agents_tools_defnition_method1(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        prompt = ""):

    print("Call demo_agents_tools_defnition_method1")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    search = SerpAPIWrapper()

    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        #Tool(
        #    name = "Search",
        #    func=search.run,
        #    description="useful for when you need to answer questions about current events"
        #),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True, max_iterations = 2)

    result = agent.run(prompt)

    print(result)

def demo_agents_serpapi(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        prompt = ""):

    print("Call demo_agents_serpapi")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    search = SerpAPIWrapper()

    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = load_tools(['llm-math'], llm=llm) #tools = load_tools(['serpapi', 'llm-math'], llm=llm)

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True, max_iterations = 2)

    result = agent.run(prompt)

    print(result)
