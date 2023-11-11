from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

from typing import Optional
#from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel, Field

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    #demo_chain_function_calling(bedrock_runtime, model_kwargs)

    demo_chain_function_calling_pydantic(bedrock_runtime, model_kwargs)


class Scientist():
    
    def __init__(self,first_name,last_name):
        self.first_name = first_name
        self.last_name = last_name

json_schema = {
                #"$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "Scientist",
                "type": "object",
              }

def transform_function(inputs : dict) -> dict:
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output' : lower_case_text}


def demo_chain_function_calling(bedrock_runtime, model_kwargs):

    print("Call demo_chain_function_calling")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = "anthropic.claude-instant-v1",
        model_kwargs = model_kwargs,
    )

    template = "Name a famous {country} scientist."

    chat_prompt = ChatPromptTemplate.from_template(template)

    chain = create_structured_output_chain(json_schema, llm, chat_prompt, verbose=True)

    result = chain.run(country='American')

    print(result['output'])


class Dog(BaseModel):
    """Identifying information about a dog."""

    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

    #class Config:
     #   arbitrary_types_allowed = True

def demo_chain_function_calling_pydantic(bedrock_runtime, model_kwargs):

    print("Call demo_chain_function_calling_pydantic")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = "anthropic.claude-instant-v1",
        model_kwargs = model_kwargs,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a world class algorithm for extracting information in structured formats."),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    chain = create_structured_output_chain(Dog, llm, prompt)
    chain.run("Harry was a chubby brown beagle who loved chicken")
    # -> Dog(name="Harry", color="brown", fav_food="chicken")

    #print(result['output'])