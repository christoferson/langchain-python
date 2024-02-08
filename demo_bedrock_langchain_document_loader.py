from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "What is the diameter of Earth?"


    #demo_load_csv(bedrock_runtime)
    #demo_load_html(bedrock_runtime)
    #demo_load_pdf(bedrock_runtime)
    person_name = "Claude Shannon" #"Tom Cruise"
    question = "When was he born?"
    demo_load_wikipedia(bedrock_runtime, model_id, model_kwargs, person_name, question)


def demo_load_csv(bedrock_runtime):

    print("Call demo_load_csv")

    loader = CSVLoader("documents/demo.csv")

    data = loader.load()

    print(data)

    print(data[0].page_content)

def demo_load_html(bedrock_runtime):

    print("Call demo_load_html")

    loader = BSHTMLLoader("documents/demo.html")

    data = loader.load()

    print(data)

    print(data[0].page_content)


def demo_load_pdf(bedrock_runtime):

    print("Call demo_load_pdf")

    loader = PyPDFLoader("documents/demo.pdf")

    data = loader.load()

    print(data)

    #print(data[0].page_content)


def demo_load_wikipedia(bedrock_runtime, model_id, model_kwargs, person_name, question):

    print("Call demo_load_wikipedia")

    loader = WikipediaLoader(query=person_name, load_max_docs=1)

    data = loader.load()

    print(data)

    context_text = data[0].page_content

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = model_id,
        model_kwargs = model_kwargs,
    )

    template = """Answer this question: 
    {question}
    Here is some extra context: 
    {document}
    """

    human_prompt = HumanMessagePromptTemplate.from_template(template)
    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])

    messages = chat_prompt.format_prompt(question=question, document=context_text).to_messages()

    result = llm(messages)

    print("\n\n")
    print(f"Question: {question}")
    print("\n")
    print(f"Answer: {result.content}")

