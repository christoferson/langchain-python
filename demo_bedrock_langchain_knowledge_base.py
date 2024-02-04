import config

from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
#from langchain.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

def run_demo(session):

    bedrock = session.client('bedrock')
    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")
    bedrock_agent_runtime = session.client('bedrock-agent-runtime')

    llm_model_id = "anthropic.claude-v2"
    embedding_model_id = "amazon.titan-embed-text-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    user_prompt = "How many ammendments are there in the US constitution?"
    user_prompt = "When was the 15th amendment ratified and what does it say?"
    #user_prompt = "アメリカ憲法修正第 15 条はいつ批准されましたか? それには何が書かれていますか?"
    demo_bedrock_langchain_knowledge_base_v2(session, bedrock_runtime, bedrock_agent_runtime, llm_model_id, embedding_model_id, user_prompt)

## 

def demo_bedrock_langchain_knowledge_base(session, 
                                bedrock_runtime, 
                                bedrock_agent_runtime,
                                llm_model_id="anthropic.claude-v2",
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_langchain_knowledge_base | llm_model_id={llm_model_id} | user_prompt={user_prompt}")

    knowledge_base_id = config.bedrock_kb["id"]

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = {
            "temperature": 0,
            "max_tokens_to_sample": 1024
        }
    )


    retriever = AmazonKnowledgeBasesRetriever(
        client = bedrock_agent_runtime,
        knowledge_base_id = knowledge_base_id,
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": 2
            }
        }
    )


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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True
    )

    result = qa.run(prompt)

    print(f"--------------------------------------")
    print(f"Question: {user_prompt}")
    print(f"Answer: {result}")
    print(f"--------------------------------------")

####


def demo_bedrock_langchain_knowledge_base_v2(session, 
                                bedrock_runtime, 
                                bedrock_agent_runtime,
                                llm_model_id="anthropic.claude-v2",
                                embedding_model_id="amazon.titan-embed-text-v1",
                                user_prompt="What is the first ammendment?"):
    
    print(f"Call demo_bedrock_langchain_knowledge_base_v2 | llm_model_id={llm_model_id} | user_prompt={user_prompt}")

    knowledge_base_id = config.bedrock_kb["id"]

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = {
            "temperature": 0,
            "max_tokens_to_sample": 1024
        }
    )


    retriever = AmazonKnowledgeBasesRetriever(
        client = bedrock_agent_runtime,
        knowledge_base_id = knowledge_base_id,
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": 2
            }
        }
    )


    # Construct the Prompt
    prompt = f"""\n\nHuman:{user_prompt}
    Assistant:
    """

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True
    )

    result = qa.run(prompt)

    print(f"--------------------------------------")
    print(f"Question: {user_prompt}")
    print(f"Answer: {result}")
    print(f"--------------------------------------")


"""
Question: アメリカ憲法修正第 15 条はいつ批准されましたか? それには何が書かれていますか?
Answer:  アメリカ憲法修正第15条は1870年2月3日に批准されました。その内容は以下の通りです:

「合衆国市民の投票権は、合衆国もしくはいずれかの州によって、人種、皮膚の色、あるいは奴隷制度の既往の状態に基づいて否認または制限されることはない。

議会は、この条項を適切な立法によって執行する権限を有する。」

要するに、人種や肌の色に関係なく、すべてのアメリカ市民に平等な投票権が保障されることを定めています。
"""


"""
Question: When was the 15th amendment ratified and what does it say?
Answer:  The 15th Amendment to the U.S. Constitution was ratified on February 3, 1870. The text of the 15th Amendment states:

SECTION 1
The right of citizens of the United States to vote shall not be denied or abridged by the United States or by any State on account of race, color, or previous condition of servitude.   

SECTION 2
The Congress shall have the power to enforce this article by appropriate legislation.

In summary, the 15th Amendment prohibits the federal government and state governments from denying citizens the right to vote based on race, color, or previous status as slaves. It gives Congress the power to enforce this prohibition through legislation.
"""