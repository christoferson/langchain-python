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
from langchain.chains import LLMChain

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

    #demo_vectordb_similarity_search(bedrock_runtime, model_id, prompt)
    #demo_vectordb_retriever(bedrock_runtime, model_id, prompt)
    #demo_vectordb_multiquery_retriever(bedrock_runtime, "anthropic.claude-instant-v1", model_kwargs, prompt)
    #demo_vectordb_multiquery_retriever_m2(bedrock_runtime, prompt = prompt)
    #demo_vectordb_contextual_retriever(bedrock_runtime, prompt = prompt)

    # US Constitution
    demo_chain(bedrock_runtime)
    #usc_load_embed_save(bedrock_runtime=bedrock_runtime)
    #usc_vectordb_contextual_retriever(bedrock_runtime, prompt = "What is the 1st Amendment?")



def setup_loggers():
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

def demo_chain(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, ):

    print("Call demo_chain")

    human_prompt = HumanMessagePromptTemplate.from_template("Tell me a trivia about {topic}")

    chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt])

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    llm_chain = LLMChain(llm=llm, prompt=chat_prompt_template)

    result = llm_chain.run(topic = "Pluto")

    print(result)

def usc_load_embed_save(bedrock_runtime : str, embedding_model_id='amazon.titan-embed-text-v1'):

    print("Call usc_load_embed_save")

    document_loader = TextLoader("documents/us_constitution.txt")

    data = document_loader.load()

    #text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)

    data_chunked = text_splitter.split_documents(data)
    
    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    vectordb = Chroma.from_documents(data_chunked, embeddings, persist_directory=USC_CHROMA_DB_PATH)

    vectordb.persist()


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

        