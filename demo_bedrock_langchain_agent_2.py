import sys
import config
import os
from langchain import hub
from langchain_community.llms import Bedrock
from langchain.chains import LLMChain
from langchain.agents import XMLAgent, create_xml_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.document_loaders import WebBaseLoader

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

    prompt = "今日の日付と今日の日本のニュースを教えてください。"
    #prompt = "次のページを要約してください https://aws.amazon.com/jp/bedrock/"
    #prompt = "次のサイトを要約して https://aws.amazon.com/bedrock/"
    #prompt = "インターネットを検索せずに答えてください。すきやきの作り方を教えてください"
    #run_demo_agent_search(bedrock_runtime, prompt)

    search = DuckDuckGoSearchRun()
    result = search.run("Dog Images")
    print(result)

def run_demo_agent_search(bedrock_runtime, prompt):
    
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="duckduckgo-search",
            func=search.run,
            description="このツールはWeb上の最新情報を検索します。引数で検索キーワードを受け取ります。最新情報が必要ない場合はこのツールは使用しません。",
        ),
        Tool(
            name = "WebBaseLoader",
            func=web_page_reader,
            description="このツールは引数でURLを渡された場合に内容をテキストで返却します。引数にはURLの文字列のみを受け付けます。"
        )
    ]

    llm = Bedrock(
        client = bedrock_runtime,
        model_id="anthropic.claude-v2",
        model_kwargs={"max_tokens_to_sample": 2000}, # 最大トークン数は大きめに
        verbose=True
    )

    chain = LLMChain(
        llm=llm,
        prompt=XMLAgent.get_default_prompt(),
        output_parser=XMLAgent.get_default_output_parser()
    )

    agent = XMLAgent(tools=tools, llm_chain=chain)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    agent_prompt = f"特に言語の指定が無い場合はあなたは質問に対して日本語で回答します。{prompt}"
    print(f"Invoking Agent with prompt: {agent_prompt}")
    answer = agent_executor.invoke({"input": agent_prompt})
    print(answer['output'])


def run_demo_agent_search_v2(bedrock_runtime, prompt):
    
    search = DuckDuckGoSearchRun()

    tools = [
        Tool(
            name="duckduckgo-search",
            func=search.run,
            description="このツールはWeb上の最新情報を検索します。引数で検索キーワードを受け取ります。最新情報が必要ない場合はこのツールは使用しません。",
        ),
        Tool(
            name = "WebBaseLoader",
            func=web_page_reader,
            description="このツールは引数でURLを渡された場合に内容をテキストで返却します。引数にはURLの文字列のみを受け付けます。"
        )
    ]

    llm = Bedrock(
        client = bedrock_runtime,
        model_id="anthropic.claude-v2",
        model_kwargs={"max_tokens_to_sample": 2000}, # 最大トークン数は大きめに
        verbose=True
    )
    """
    chain = LLMChain(
        llm=llm,
        prompt=XMLAgent.get_default_prompt(),
        output_parser=XMLAgent.get_default_output_parser()
    )

    agent = XMLAgent(tools=tools, llm_chain=chain)
    """
    agent = create_xml_agent(llm, tools, hub.pull("hwchase17/xml-agent-convo"))
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    agent_prompt = f"特に言語の指定が無い場合はあなたは質問に対して日本語で回答します。{prompt}"
    print(f"Invoking Agent with prompt: {agent_prompt}")
    answer = agent_executor.invoke({"input": agent_prompt})
    print(answer['output'])