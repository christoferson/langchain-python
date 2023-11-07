from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"

    demo_load_txt_and_transform(bedrock_runtime)



def demo_load_txt_and_transform(bedrock_runtime):

    print("Call demo_load_txt_and_transform")

    with open("documents/demo.txt", encoding="utf-8") as file:
        file_text = file.read()

    #print(file_text)
    print(len(file_text))
    print(len(file_text.split()))

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
    
    texts = text_splitter.create_documents([file_text])

    print(texts[0])

