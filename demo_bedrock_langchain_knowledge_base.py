import config
import json

from langchain.chains import RetrievalQA
from langchain.chat_models import BedrockChat
from langchain.retrievers import AmazonKnowledgeBasesRetriever

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")
    bedrock_agent_runtime = session.client('bedrock-agent-runtime')

    llm_model_id = "anthropic.claude-instant-v1"
    embedding_model_id = "amazon.titan-embed-text-v1"
    llm_model_id = "anthropic.claude-v2"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    user_prompt = "How many ammendments are there in the US constitution?"
    user_prompt = "When was the 15th amendment ratified and what does it say?"
    #user_prompt = "What is AMICUS BRIE?"
    #user_prompt = "How is AMICUS BRIE different from AMICUS BRIEF?"
    demo_bedrock_langchain_knowledge_base(session, bedrock_runtime, bedrock_agent_runtime, llm_model_id, embedding_model_id, user_prompt)

## 

def demo_bedrock_langchain_knowledge_base(session, 
                                bedrock_runtime, 
                                bedrock_agent_runtime,
                                llm_model_id="anthropic.claude-v2",
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_langchain_knowledge_base | llm_model_id={llm_model_id} | user_prompt={user_prompt}")

    # Knowledge Base
    knowledge_base_id = config.bedrock_kb["id"]

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id=llm_model_id,
        model_kwargs={
            "temperature": 0,
            "max_tokens_to_sample":1024
        }
    )


    retriever = AmazonKnowledgeBasesRetriever(
        client=bedrock_agent_runtime,
        knowledge_base_id=knowledge_base_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 4
            }
        }
    )


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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True
    )

    query = "Does Amazon Bedrock support PrivateLink?"
    result = qa.run(prompt)

    #print("Received response:" + json.dumps(result, ensure_ascii=False))


    print(f"--------------------------------------")
    print(f"Question: {user_prompt}")
    print(f"Answer: {result}")
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



