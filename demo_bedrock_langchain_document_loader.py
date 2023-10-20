from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader

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
    demo_load_pdf(bedrock_runtime)


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