from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

from langchain.text_splitter import CharacterTextSplitter

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    demo_load_txt_and_transform(bedrock_runtime, model_kwargs)

def transform_function(inputs : dict) -> dict:
    text = inputs['text']
    only_review_text = text.split('REVIEW:')[-1]
    lower_case_text = only_review_text.lower()
    return {'output' : lower_case_text}


def demo_load_txt_and_transform(bedrock_runtime, model_kwargs):

    print("Call demo_load_txt_and_transform")

    with open("documents/demo.txt", encoding="utf-8") as file:
        file_text = file.read()

    transform_chain = TransformChain(input_variables=['text'], output_variables=['output'], transform=transform_function)

    template = "Create a 3 sentence summary of the review:\n{review}"

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = "anthropic.claude-instant-v1",
        model_kwargs = model_kwargs,
    )

    prompt = ChatPromptTemplate.from_template(template)

    summary_chain = LLMChain(llm=llm, prompt=prompt, output_key='review_summary')

    sequential_chain = SimpleSequentialChain(chains=[transform_chain, summary_chain], verbose=True)

    result = sequential_chain(file_text)

    #print(result)

    print(result['output'])
