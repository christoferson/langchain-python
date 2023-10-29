import logging

from langchain.document_loaders import CSVLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WikipediaLoader
from langchain.document_loaders import TextLoader

from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings

from langchain.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from typing import List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

USC_CHROMA_DB_PATH = "vectordb/chromadb/usc.db"

def run_demo(session):

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_id = "amazon.titan-embed-text-v1"

    model_kwargs = { "temperature": 0.0 }

    #demo_load_embed_save(bedrock_runtime)
    prompt = "What is the meaning behind the name New York?"
    #prompt = "Etymology New York"

    setup_loggers()

    emmployee_review = """
        "Employee Information:\n",
        "Name: Joe Schmo\n",
        "Position: Software Engineer\n",
        "Date of Review: July 14, 2023\n",
        "\n",
        "Strengths:\n",
        "Joe is a highly skilled software engineer with a deep understanding of programming languages, algorithms, and software development best practices. His technical expertise shines through in his ability to efficiently solve complex problems and deliver high-quality code.\n",
        "\n",
        "One of Joe's greatest strengths is his collaborative nature. He actively engages with cross-functional teams, contributing valuable insights and seeking input from others. His open-mindedness and willingness to learn from colleagues make him a true team player.\n",
        "\n",
        "Joe consistently demonstrates initiative and self-motivation. He takes the lead in seeking out new projects and challenges, and his proactive attitude has led to significant improvements in existing processes and systems. His dedication to self-improvement and growth is commendable.\n",
        "\n",
        "Another notable strength is Joe's adaptability. He has shown great flexibility in handling changing project requirements and learning new technologies. This adaptability allows him to seamlessly transition between different projects and tasks, making him a valuable asset to the team.\n",
        "\n",
        "Joe's problem-solving skills are exceptional. He approaches issues with a logical mindset and consistently finds effective solutions, often thinking outside the box. His ability to break down complex problems into manageable parts is key to his success in resolving issues efficiently.\n",
        "\n",
        "Weaknesses:\n",
        "While Joe possesses numerous strengths, there are a few areas where he could benefit from improvement. One such area is time management. Occasionally, Joe struggles with effectively managing his time, resulting in missed deadlines or the need for additional support to complete tasks on time. Developing better prioritization and time management techniques would greatly enhance his efficiency.\n",
        "\n",
        "Another area for improvement is Joe's written communication skills. While he communicates well verbally, there have been instances where his written documentation lacked clarity, leading to confusion among team members. Focusing on enhancing his written communication abilities will help him effectively convey ideas and instructions.\n",
        "\n",
        "Additionally, Joe tends to take on too many responsibilities and hesitates to delegate tasks to others. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.\n"
    """

    #demo_chain(bedrock_runtime)
    #demo_chain_simple_sequential(bedrock_runtime)
    demo_chain_sequential(bedrock_runtime, emmployee_review=emmployee_review)



def setup_loggers():
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

def demo_chain(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, 
                            prompt = "Pluto"):

    print("Call demo_chain")

    human_prompt = HumanMessagePromptTemplate.from_template("Tell me a trivia about {topic}")

    chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt])

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    llm_chain = LLMChain(llm=llm, prompt=chat_prompt_template)

    result = llm_chain.run(topic = prompt)

    print(result)

def demo_chain_simple_sequential(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, 
                            prompt = "Cheesecake"):

    print("Call demo_chain_simple_sequential")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    prompt_1 = ChatPromptTemplate.from_template("Give me a simple bullet point outline for a blog post on {topic}")
    chain_1 = LLMChain(llm=llm, prompt=prompt_1)

    prompt_2 = ChatPromptTemplate.from_template("Write a blog post using this {outline}.")
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    llm_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

    result = llm_chain.run(prompt)

    print(result)


def demo_chain_sequential(bedrock_runtime : str, 
                            embedding_model_id = "amazon.titan-embed-text-v1", 
                            llm_model_id = "anthropic.claude-instant-v1", 
                            llm_model_kwargs = { "temperature": 0.0 }, 
                            emmployee_review = ""):

    print("Call demo_chain_simple_sequential")

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    prompt_1 = ChatPromptTemplate.from_template("Give a summary of this employee's performance \n {review}")
    chain_1 = LLMChain(llm=llm, prompt=prompt_1, output_key="review_summary")

    prompt_2 = ChatPromptTemplate.from_template("Identify key employee weekneses in this review summary: \n {review_summary}.")
    chain_2 = LLMChain(llm=llm, prompt=prompt_2, output_key="weakneses")

    prompt_3 = ChatPromptTemplate.from_template("Create a personalized plan to help address and fix these weaknesses: \n {weakneses}.")
    chain_3 = LLMChain(llm=llm, prompt=prompt_3, output_key="final_plan")

    llm_chain = SequentialChain(chains=[chain_1, chain_2, chain_3], verbose=True, input_variables=["review"], 
                                output_variables=["review_summary", "weakneses", "final_plan"])

    result = llm_chain(emmployee_review)

    print(result["review_summary"])
    print(result["weakneses"])
    print(result["weakneses"])

        