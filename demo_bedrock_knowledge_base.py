import config
from time import sleep
import random
import json

from opensearchpy import helpers

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
    bedrock_agent_runtime = session.client('bedrock-agent-runtime')

    llm_model_id = "anthropic.claude-instant-v1"
    embedding_model_id = "amazon.titan-embed-text-v1"

    user_prompt = "How many ammendments are there in the US constitution?"
    demo_bedrock_knowledge_base(session, bedrock_runtime, bedrock_agent_runtime, embedding_model_id, user_prompt)
    #demo_bedrock_knowledge_base_search(session, bedrock_runtime, bedrock_agent_runtime, embedding_model_id)

## 

def demo_bedrock_knowledge_base(session, bedrock_runtime, bedrock_agent_runtime,
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_knowledge_base | user_prompt={user_prompt}")

    # Knowledge Base
    knowledge_base_id = config.bedrock_kb["id"]

    # Select the model to use - Currently Anthropic is Supported
    model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2'
    #model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-instant-v1'

    # Construct the Prompt
    language = "en"
    prompt = f"""\n\nHuman:
    Answer the question after [Question] corectly. 
    [Question]
    {user_prompt}
    Assistant:
    """

    if language == "ja":
        prompt = f"""\n\nHuman:
        [質問]に適切に答えてください。回答は日本語に翻訳してください。  
        [質問]
        {user_prompt}
        Assistant:
        """

    response = bedrock_agent_runtime.retrieve_and_generate(
        input={
            'text': prompt,
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': model_arn,
            }
        }
    )

    print("Received response:" + json.dumps(response, ensure_ascii=False))

    response_output = response['output']['text']

    print(f"--------------------------------------")
    print(f"Question: {user_prompt}")
    print(f"Answer: {response_output}")
    print(f"--------------------------------------")



## 

def demo_bedrock_knowledge_base_search(session, bedrock_runtime, bedrock_agent_runtime,
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_knowledge_base_search | user_prompt={user_prompt}")

    # Knowledge Base
    knowledge_base_id = config.bedrock_kb["id"]

    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId = knowledge_base_id,
        retrievalQuery={
            'text': user_prompt,
        },
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': 3
            }
        }
    )

    print("Received response:" + json.dumps(response, ensure_ascii=False, indent=1))

    print(f"--------------------------------------")

    for i, retrievalResult in enumerate(response['retrievalResults']):
        uri = retrievalResult['location']['s3Location']['uri']
        excerpt = retrievalResult['content']['text'][0:75]
        score = retrievalResult['score']
        print(f"{i} RetrievalResult: {score} {uri} {excerpt}")
        
    print(f"--------------------------------------")



