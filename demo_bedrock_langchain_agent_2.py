import sys
import config
import os
from langchain.llms import Bedrock
from langchain.chains import LLMChain
from langchain.agents import XMLAgent
from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun

from langchain.document_loaders import WebBaseLoader

def web_page_reader(url: str) -> str:
    loader = WebBaseLoader(url)
    content = loader.load()[0].page_content
    return content

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    print("OK")