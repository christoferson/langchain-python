import logging

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader
from langchain.document_loaders import TextLoader

from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings

from langchain.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

USC_CHROMA_DB_PATH = "vectordb/chromadb/usc.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_id = "amazon.titan-embed-text-v1"

    model_kwargs = { "temperature": 0.0 }

    #demo_load_embed_save(bedrock_runtime)
    prompt = "What is the meaning behind the name New York?"
    #prompt = "Etymology New York"

    setup_loggers()

    #demo_chain(bedrock_runtime)
    demo_chain_simple_sequential(bedrock_runtime)
    #usc_vectordb_contextual_retriever(bedrock_runtime, prompt = "What is the 1st Amendment?")



def setup_loggers():
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

def demo_chain(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, 
                            prompt = "Pluto"):

    print("Call demo_chain")

    human_prompt = HumanMessagePromptTemplate.from_template("Tell me a trivia about {topic}")

    chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt])

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    llm_chain = LLMChain(llm=llm, prompt=chat_prompt_template)

    result = llm_chain.run(topic = prompt)

    print(result)

def demo_chain_simple_sequential(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, 
                            prompt = "Cheesecake"):

    print("Call demo_chain_simple_sequential")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    prompt_1 = ChatPromptTemplate.from_template("Give me a simple bullet point outline for a blog post on {topic}")
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = ChatPromptTemplate.from_template("Write a blog post using this {outline}.")
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    llm_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

    result = llm_chain.run(prompt)

    print(result)


def usc_vectordb_contextual_retriever(bedrock_runtime : str, 
                                          embedding_model_id = "amazon.titan-embed-text-v1", 
                                          llm_model_id = "anthropic.claude-instant-v1", 
                                          llm_model_kwargs = { "temperature": 0.0 }, 
                                          prompt = None):

    print("Call usc_vectordb_contextual_retriever")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    vectordb = Chroma(embedding_function = embeddings, persist_directory = USC_CHROMA_DB_PATH)

    retriever = vectordb.as_retriever()

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                           base_retriever=retriever)

    similar_docs = compression_retriever.get_relevant_documents(prompt, kwargs={ "k": 2 })

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")

        