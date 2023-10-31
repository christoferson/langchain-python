import config

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

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth

from langchain.vectorstores import OpenSearchVectorSearch

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    llm_model_id = "anthropic.claude-instant-v1"
    embedding_model_id = "amazon.titan-embed-text-v1"

    demo_access_opensearch(session)
    #demo_access_opensearch_2(session, bedrock_runtime)
    #demo_load_embed_save(bedrock_runtime)
    prompt = "What is the origin of the name New York?"
    prompt = "Etymology New York"
    #demo_vectordb_similarity_search(bedrock_runtime, model_id, prompt)
    #demo_vectordb_retriever(bedrock_runtime, model_id, prompt)

bookinfo_index_settings = {
  "mappings": {
    "properties": {
      "id":{
        "type": "keyword"
      },   
      "bookName":{
        "type": "text",
        "analyzer": "kuromoji"
      },
      "description":{
        "type": "text",
        "analyzer": "kuromoji"
      },      
      "publicationDate":{
        "type": "date",
      } 
    }
  }
}

def demo_access_opensearch(session):
    print("Call demo_access_opensearch")
    client = session.client('opensearchserverless')
    result = client.list_collections()
    print(result)

    service = 'aoss'
    region = 'us-east-1'
    credentials = session.get_credentials()
    print(credentials.access_key)
    #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"]
    print(host)
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )


    query = {
      "size": 20,
      "fields": ["title", "plot"],
      "_source": False,
      "query": {
        "knn": {
          "v_title": {
            "vector": "123",
            "k": 1
          }
        }
      }
    }
    
    print (query)
    
    response = client.search(
        body = query,
        index = "bookinfo"
    )

    print (response)

    index_name = "bookinfo"
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=bookinfo_index_settings)
        print("index created")
    else:
        print("index already exist")

def demo_access_opensearch_2(session, bedrock_runtime, embedding_model_id="amazon.titan-embed-text-v1"):
    
    print("Call demo_access_opensearch_2")

    ###
    document_loader = TextLoader("documents/demo.txt")

    data = document_loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)

    data_chunked = text_splitter.split_documents(data)

    print("Loaded documents")
    ###

    client = session.client('opensearchserverless')

    service = 'aoss'
    region = 'us-east-1'
    credentials = session.get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)
    host = config.aws["opensearch_serverless_endpoint"] 

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    print("Ready")

    docsearch = OpenSearchVectorSearch.from_documents(
        data_chunked,
        embeddings,
        opensearch_url=host,
        http_auth=awsauth,
        timeout=30,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        index_name="test-index-using-aoss",
        engine="faiss",
    )

    print("Searching")

    docs = docsearch.similarity_search(
        "What is feature selection",
        efficient_filter=filter,
        k=200,
    )



    print(docs)

def demo_load_embed_save(bedrock_runtime : str, embedding_model_id='amazon.titan-embed-text-v1'):

    print("Call demo_load_embed_save")

    document_loader = TextLoader("documents/demo.txt")

    data = document_loader.load()

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)

    data_chunked = text_splitter.split_documents(data)
    
    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
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