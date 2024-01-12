import config
import json

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader
from langchain.document_loaders import TextLoader
from langchain.chat_models import BedrockChat
from langchain.embeddings import BedrockEmbeddings

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")
    bedrock_agent_runtime = session.client('bedrock-agent-runtime')

    llm_model_id = "anthropic.claude-instant-v1"
    llm_model_id="anthropic.claude-v2"
    embedding_model_id = "amazon.titan-embed-text-v1"

    user_prompt = "How many ammendments are there in the US constitution?"
    user_prompt = "When was the 15th amendment ratified and what does it say?"
    #user_prompt = "アメリカ憲法修正第 15 条はいつ批准されましたか? それには何が書かれていますか?"
    #user_prompt = "What is AMICUS BRIE?"
    #user_prompt = "How is AMICUS BRIE different from AMICUS BRIEF?"
    #demo_bedrock_knowledge_base(session, bedrock_runtime, bedrock_agent_runtime, llm_model_id, embedding_model_id, user_prompt)
    demo_bedrock_knowledge_base_v2(session, bedrock_runtime, bedrock_agent_runtime, llm_model_id, embedding_model_id, user_prompt)
    #demo_bedrock_knowledge_base_search(session, bedrock_runtime, bedrock_agent_runtime, embedding_model_id)

## 

def demo_bedrock_knowledge_base(session, 
                                bedrock_runtime, 
                                bedrock_agent_runtime,
                                llm_model_id="anthropic.claude-v2",
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_knowledge_base | llm_model_id={llm_model_id} | user_prompt={user_prompt}")

    # Knowledge Base
    knowledge_base_id = config.bedrock_kb["id"]

    # Select the model to use - Currently Anthropic is Supported
    model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2'
    #model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-instant-v1'

    # Construct the Prompt
    language = "ja"

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


def demo_bedrock_knowledge_base_v2(session, 
                                bedrock_runtime, 
                                bedrock_agent_runtime,
                                llm_model_id="anthropic.claude-v2",
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_knowledge_base_v2 | llm_model_id={llm_model_id} | user_prompt={user_prompt}")

    # Knowledge Base
    knowledge_base_id = config.bedrock_kb["id"]

    # Select the model to use - Currently Anthropic is Supported
    model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2'
    #model_arn = 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-instant-v1'

    # Construct the Prompt
    prompt = f"""\n\nHuman: {user_prompt}
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





"""
Question: アメリカ憲法修正第 15 条はいつ批准されましたか? それには何が書かれていますか?
Answer: アメリカ合衆国憲法修正第15条は1869年2月26日に議会を通過し、1870年2月3日に批准されました。 
この修正条項では、人種、肌の色、または以前の隷属状態に基づいて、アメリカ合衆国またはいか
なる州も、市民の投票権を否定または制限することを禁止しています。
"""

"""
Question: When was the 15th amendment ratified and what does it say?
Answer: The 15th Amendment was ratified on February 3, 1870. The 15th Amendment states that the right of citizens 
to vote shall not be denied or abridged based on race, color, or previous condition of servitude.
"""