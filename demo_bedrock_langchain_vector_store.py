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


CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_id = "amazon.titan-embed-text-v1"

    #demo_load_embed_save(bedrock_runtime)
    prompt = "What is the origin of the name New York?"
    prompt = "Etymology New York"
    #demo_vectordb_similarity_search(bedrock_runtime, model_id, prompt)
    demo_vectordb_retriever(bedrock_runtime, model_id, prompt)



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