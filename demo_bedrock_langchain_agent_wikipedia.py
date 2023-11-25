import os
import config

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.vectorstores import Chroma

from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain, SequentialChain

from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.agents import load_tools, initialize_agent, AgentType

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    demo_agents_wikipedia_connect(bedrock_runtime, prompt="When was Albert Einstein born? What is the year multiplied by 5?")


def demo_agents_wikipedia_connect(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        prompt = ""):

    print("Call demo_agents_wikipedia_connect")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    result = wikipedia.run("HUNTER X HUNTER")

    print(result)

    #tools = load_tools(['llm-math'], llm=llm) #tools = load_tools(['serpapi', 'llm-math'], llm=llm)

    #agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

    #result = agent.run(prompt)

    #print(result)
