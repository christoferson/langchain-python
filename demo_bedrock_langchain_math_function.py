from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import LLMMathChain
from langchain_community.utilities import SerpAPIWrapper

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    demo_langchain_math(bedrock_runtime)

def transform_function(inputs : dict) -> dict:
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output' : lower_case_text}


def demo_langchain_math(bedrock_runtime, 
                                embedding_model_id = "amazon.titan-embed-text-v1", 
                                llm_model_id = "anthropic.claude-instant-v1", 
                                llm_model_kwargs = { "temperature": 0.0 }, ):

    print("Call demo_langchain_math")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    prompt = "What is 5 plus 12?"
    prompt = "What is 17 raised to the power of 11."

    prompt_template = PromptTemplate(input_variables = [], template = prompt)

    chain = prompt_template | llm

    result = chain.invoke({"foo": "bar"})

    #print(result)

    print(result.content)
    print()
    print(f"17 exp 11: {17**11}")
    print(f"17 exp 10: {17**10}")
    print(f"17 exp 9: {17**9}")
    print(f"17 exp 8: {17**8}")
    print(f"17 exp 7: {17**7}")
    print(f"17 exp 6: {17**6}")
    print(f"17 exp 5: {17**5}")
    print(f"17 exp 4: {17**4}")
    print(f"17 exp 3: {17**3}")

    llm_math = LLMMathChain.from_llm(llm)
    result = llm_math(prompt)
    print(result)