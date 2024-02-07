import logging

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import TextLoader

from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

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
    demo_vectordb_multiquery_retriever_m2(bedrock_runtime, prompt = prompt)


def setup_loggers():
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

def demo_load_embed_save(bedrock_runtime : str, model_id='amazon.titan-embed-text-v1'):

    print("Call demo_load_embed_save")

    document_loader = TextLoader("documents/demo.txt")

    data = document_loader.load()

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)

    data_chunked = text_splitter.split_documents(data)
    
    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    vectordb = Chroma.from_documents(data_chunked, embeddings, persist_directory=CHROMA_DB_PATH)

    vectordb.persist()

    #result = embeddings.embed_documents([text.page_content for text in data_chunked])

    #print(f"Embeddings.Length: {len(result)}")


def demo_vectordb_similarity_search(bedrock_runtime : str, model_id, prompt):

    print("Call demo_vectordb_similarity_search")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    similar_docs = vectordb.similarity_search(prompt, k=2)

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")

def demo_vectordb_retriever(bedrock_runtime : str, model_id, prompt):

    print("Call demo_vectordb_retriever")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    retriever = vectordb.as_retriever()

    similar_docs = retriever.get_relevant_documents(prompt, kwargs={ "k": 2 })

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")



def demo_vectordb_multiquery_retriever(bedrock_runtime : str, model_id, model_kwargs, prompt):

    print("Call demo_vectordb_multiquery_retriever")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = "amazon.titan-embed-text-v1"
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = "anthropic.claude-instant-v1",
        model_kwargs = model_kwargs,
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    retriever = vectordb.as_retriever()

    multiquery_retriever = MultiQueryRetriever.from_llm(retriever = retriever, llm = llm)

    # Error since questions contain empty elements
    similar_docs = multiquery_retriever.get_relevant_documents(prompt)

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")

#
# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        #print(f"Lines: {lines}")
        for line in lines:
            if (len(line)==0):
                lines.remove(line)
            else:
                print(f"Line: {line}")
        #print(f"Lines: {lines}")
        return LineList(lines=lines)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}""",
)

def demo_vectordb_multiquery_retriever_m2(bedrock_runtime : str, 
                                          embedding_model_id = "amazon.titan-embed-text-v1", 
                                          llm_model_id = "anthropic.claude-instant-v1", 
                                          llm_model_kwargs = { "temperature": 0.0 }, prompt = None):

    print("Call demo_vectordb_multiquery_retriever")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    retriever = vectordb.as_retriever()

    # Create MultiQueryRetriever to perform Similarity Search

    output_parser = LineListOutputParser()

    llm_chain = LLMChain(llm = llm, prompt = QUERY_PROMPT, output_parser = output_parser)

    multiquery_retriever = MultiQueryRetriever(
        retriever = vectordb.as_retriever(), llm_chain = llm_chain, parser_key = "lines"
    )  # "lines" is the key (attribute name) of the parsed output

    similar_docs = multiquery_retriever.get_relevant_documents(prompt, kwargs={ "k": 2 })

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")