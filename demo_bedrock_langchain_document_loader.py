from langchain.document_loaders import CSVLoader

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"

    model_kwargs = { "temperature": 0.0 }
    prompt = "What is the diameter of Earth?"


    demo_load_csv(bedrock_runtime)


def demo_load_csv(bedrock_runtime):

    print("Call demo_load_csv")

    loader = CSVLoader("csv/demo.csv")

    data = loader.load()

    print(data)

    print(data[0].page_content)