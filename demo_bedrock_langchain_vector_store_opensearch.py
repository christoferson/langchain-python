import config
from time import sleep

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

    #demo_boto3_opensearch(session)
    #demo_access_opensearch(session)
    #demo_access_opensearch_1(session)
    #demo_boto3_bucket_delete(session)
    #demo_access_opensearch_2(session, bedrock_runtime)
    #demo_access_opensearch_search(session)
    demo_access_opensearch_search_2(session)
    #demo_load_embed_save(bedrock_runtime)


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


housing_index_settings = {
    "settings": {
      "index.knn": "true"
   },
   "mappings": {
      "properties": {
         "housing-vector": {
            "type": "knn_vector",
            "dimension": 3
         },
         "title": {
            "type": "text"
         },
         "price": {
            "type": "long"
         },
         "location": {
            "type": "geo_point"
         }
      }
   }
}


def demo_boto3_bucket_delete(session):
    print("Call demo_boto3_bucket_delete")

    BUCKETS = [
        "kendra-basic-kendradatasourcebucket2-55ycr3ekk2lr",
        "kendra-basic-kendradatasourcebucket-ubmynaqy7o3m",
    ]

    s3 = session.resource("s3")
    for bucket_name in BUCKETS:
        # For each bucket, create a resource object with boto3
        bucket = s3.Bucket(bucket_name)
        # Delete all of the objects in the bucket
        bucket.object_versions.delete()

        # Delete the bucket itself!
        #bucket.delete()

def demo_boto3_opensearch(session):
    print("Call demo_boto3_opensearch")

    client = session.client('opensearchserverless')

    print("\n\nList Collections")
    result = client.list_collections( maxResults=5)
    print(result['collectionSummaries'])


def demo_access_opensearch(session):

    print("Call demo_access_opensearch")

    service = 'aoss'
    region = config.aws["region_name"]
    credentials = session.get_credentials()
    #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"]

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

#    index_name = "bookinfo"
#    if not client.indices.exists(index=index_name):
#        client.indices.create(index=index_name, body=bookinfo_index_settings)
#        print("index created")
#    else:
#        print("index already exist")


def demo_access_opensearch_1(session):

    print("Call demo_access_opensearch_1")

    service = 'aoss'
    region = config.aws["opensearch_serverless_region"]
    credentials = session.get_credentials()
    #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"]

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )

    index_name = "bookinfo"
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=bookinfo_index_settings)
        print("index created")
    else:
        print("index already exist")


def demo_access_opensearch_search(session):

    print("Call demo_access_opensearch_search")

    region = config.aws["opensearch_serverless_region"]
    credentials = session.get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"]

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )

    #info = client.info()
    #print(f"{info['version']['distribution']}: {info['version']['number']}")

    index_name = "movies"
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)
        #client.indices.create(index=index_name, body=bookinfo_index_settings)

    try:
        # index data
        document = {"director": "Bennett Miller", "title": "Moneyball", "year": 2011}
        #client.index(index=index_name, body=document, id="1") #'Document ID is not supported in create/index operation request'
        client.index(index=index_name, body=document)

        # wait for the document to index
        sleep(1)

        # search for the document
        results = client.search(body={"query": {"match": {"director": "Bennett Miller"}}})
        print(f"Search Results: {results}")
        for hit in results["hits"]["hits"]:
            print(hit["_source"])

        # delete the document
        #client.delete(index=index_name, id="1") #(400, 'illegal_argument_exception', 'Invalid external document id:[1] for index type: [VECTORSEARCH].')
    finally:
        # delete the index
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)


def demo_access_opensearch_search_2(session):

    print("Call demo_access_opensearch_search_2")

    region = config.aws["opensearch_serverless_region"]
    credentials = session.get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"]

    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )

    index_name = "housing"
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=housing_index_settings)

    sleep(1)

    try:
        document = {
            "housing-vector": [
                10,
                20,
                30
            ],
            "title": "2 bedroom in downtown Seattle",
            "price": "2800",
            "location": "47.71, 122.00"
        }
        indexed_document = client.index(index=index_name, body=document)
        print(indexed_document)
        print("----------------------------------------------------------")

        # wait for the document to index
        sleep(1)

        # search for the document
        search_query = {
            "knn": {
                "housing-vector": {
                    "vector": [
                        10,
                        20,
                        30
                    ],
                    "k": 5
                }
            }
        }
        results = client.search(index=index_name, size=5, body={"query": search_query })
        print(f"Search Results: {results}")
        for hit in results["hits"]["hits"]:
            print(hit["_source"])

        # delete the document
        #client.delete(index=index_name, id="1") #(400, 'illegal_argument_exception', 'Invalid external document id:[1] for index type: [VECTORSEARCH].')
    finally:
        # delete the index
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)


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

    region = config.aws["region_name"]
    credentials = session.get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region, service="aoss")
    host = config.aws["opensearch_serverless_endpoint"] 

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    print("Ready")

    index_name = "bookinfo"

    docsearch = OpenSearchVectorSearch.from_documents(
        data_chunked,
        embeddings,
        opensearch_url=host,
        http_auth=awsauth,
        timeout=30,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        index_name="index_name",
        #engine="faiss",
    )

    print("Searching")

    print("-----------------------------------")

    docs = docsearch.similarity_search(
        "What is feature selection",
        efficient_filter=filter,
        k=200,
    )

    print(docs)

    print("-----------------------------------")

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