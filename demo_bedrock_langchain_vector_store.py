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

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_id = "amazon.titan-embed-text-v1"

    demo_load_embed_save(bedrock_runtime)



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

    vectordb = Chroma.from_documents(data_chunked, embeddings, persist_directory="vectordb/chromadb/demo.db")

    vectordb.persist()

    #result = embeddings.embed_documents([text.page_content for text in data_chunked])

    #print(f"Embeddings.Length: {len(result)}")
