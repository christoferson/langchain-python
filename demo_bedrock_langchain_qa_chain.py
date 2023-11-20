from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.vectorstores import Chroma

from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

from langchain.text_splitter import CharacterTextSplitter


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    #demo_langchain_qa_chain(bedrock_runtime)
    demo_langchain_qa_chain_with_sources(bedrock_runtime)



def demo_langchain_qa_chain(bedrock_runtime, 
                                embedding_model_id : str = "amazon.titan-embed-text-v1", 
                                llm_model_id : str = "anthropic.claude-instant-v1", 
                                llm_model_kwargs : dict = { "temperature": 0.0 }, ):

    print("Call demo_langchain_qa_chain")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    chain = load_qa_chain(llm, chain_type="stuff")

    question = "What is the origin of the name New York?"

    similar_docs = vectordb.similarity_search(question, k=2)
    print(similar_docs)
    result = chain.run(input_documents = similar_docs, question = question)

    print("*****")
    print(result)


def demo_langchain_qa_chain_with_sources(bedrock_runtime, 
                                embedding_model_id : str = "amazon.titan-embed-text-v1", 
                                llm_model_id : str = "anthropic.claude-instant-v1", 
                                llm_model_kwargs : dict = { "temperature": 0.0 }, ):

    print("Call demo_langchain_qa_chain_with_sources")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)

    chain = load_qa_with_sources_chain(llm, chain_type="stuff")

    question = "What is the origin of the name New York?"

    similar_docs = vectordb.similarity_search(question, k=2)
    print(similar_docs)
    result = chain.run(input_documents = similar_docs, question = question)

    print("*****")
    print(result)