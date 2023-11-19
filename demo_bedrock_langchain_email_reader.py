from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader

from langchain.vectorstores import Chroma

from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain, SequentialChain

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

    email_contents = open('documents/customer_email_spanish.txt', encoding="utf-8").read()
    #email_contents = open('documents/customer_email_japanese.txt', encoding="utf-8").read()

    #print(email_contents)
    #print("*****")

    #demo_langchain_qa_chain(bedrock_runtime)
    demo_read_translate_email(bedrock_runtime, email_contents=email_contents, output_language="Japanese", 
                              llm_model_kwargs = { "temperature": 0.0 })



def demo_read_translate_email(bedrock_runtime, 
                                embedding_model_id : str = "amazon.titan-embed-text-v1", 
                                llm_model_id : str = "anthropic.claude-instant-v1", 
                                llm_model_kwargs : dict = { "temperature": 0.0 }, 
                                email_contents : str = "Sample Email Content",
                                output_language : str = "English"):

    print("Call demo_read_translate_email")

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )
    
    # Detect Language
    template1 = "Return the language this email is written in:\n{email}.\nReturn only the language it was written in."
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=llm, prompt=prompt1, output_key="language")
    
    # Translate from detected language to English
    template2 = "Translate this email from {language} to " + output_language + ". Here is the email:\n" + email_contents
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=llm, prompt=prompt2, output_key="translated_email")
    
    # Return Summary AND the Translated Email
    template3 = "Create a short summary of this email in " + output_language + ":\n{translated_email}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=llm, prompt=prompt3, output_key="summary")
    
    seq_chain = SequentialChain(chains=[chain_1,chain_2,chain_3],
                            input_variables=['email'],
                            output_variables=['language','translated_email','summary'],
                            verbose=True)

    result = seq_chain(email_contents)

    print("*****")
    print(result.keys())

    print("*****")
    print(result["language"])

    print("*****")
    print(result["translated_email"])

    print("*****")
    print(result["summary"])