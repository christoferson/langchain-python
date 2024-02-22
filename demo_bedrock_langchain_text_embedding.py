from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_id = "amazon.titan-embed-text-v1"

    demo_load_and_embed(bedrock_runtime)



def demo_text_embedding(bedrock_runtime, model_id='amazon.titan-embed-text-v1'):

    print("Call demo_text_embedding")
    
    text = "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor."

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    result = embeddings.embed_query(text)

    print(len(result))


def demo_load_and_embed(bedrock_runtime : str, model_id='amazon.titan-embed-text-v1'):

    print("Call demo_text_embedding")

    document_loader = CSVLoader("documents/demo.csv")

    data = document_loader.load()
    
    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = model_id
    )

    result = embeddings.embed_documents([text.page_content for text in data])

    print(f"Embeddings.Length: {len(result)}")
