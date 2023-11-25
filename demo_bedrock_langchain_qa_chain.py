
from langchain.llms import Bedrock

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
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import warnings


CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

def run_demo(session):

    #warnings.filterwarnings('ignore')

    bedrock = session.client('bedrock')

    bedrock_runtime = session.client('bedrock-runtime', region_name="us-east-1")

    model_id = "anthropic.claude-v1"
    model_id = "anthropic.claude-v2"
    model_id = "anthropic.claude-instant-v1"
    model_kwargs = { "temperature": 0.0, 'max_tokens_to_sample': 200 }

    #demo_langchain_qa_chain(bedrock_runtime)
    #demo_langchain_qa_chain_with_sources(bedrock_runtime)

    
    #query = "How did New York get it's name?" # Infinite Loop
    query = "When did New York get it's name?"
    demo_langchain_retrieval_qa_chain_2(bedrock_runtime, llm_model_id="anthropic.claude-v2", query=query)



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


def demo_langchain_retrieval_qa_chain(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        query : str = ""):

    print(f"Call demo_langchain_retrieval_qa_chain llm_model_id={llm_model_id} ")

    # 1. Create Prompt Template for Claude

    prompt_template = """Human: 
    Text: {context}

    Question: {question}

    Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available.

    Assistant:
    """

    prompt_template = """
    Human: 
    You are a helpful, respectful, and honest assistant, dedicated to providing valuable and accurate information.

    Assistant:
    Understood. I will provide information based on the context given, without relying on prior knowledge.

    Human:
    If you don't see answer in the context just Reply "Sorry , the answer is not in the context so I don't know".

    Assistant:
    Noted. I will respond with "don't know" if the information is not available in the context.

    Human:
    Now read this context and answer the question. 
    {context}

    Assistant:
    Based on the provided context above and information from the retriever source, I will provide a detailed answer to the below question
    {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

    # 2. Instantiate Claude Bedrock

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    # 3. Create RetrievalQA

    chain_type_kwargs = {"prompt": PROMPT}
    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}) #search_kwargs.k defines number of ducuments to search

    #
    similar_docs = retriever.get_relevant_documents(query, kwargs={ "k": 2 })

    print(f"Matches: {len(similar_docs)}")
    for similar_doc in similar_docs:
        print("---------------------------------")
        print(f"{similar_doc.metadata}")
        print(f"{similar_doc.page_content}")
    print("---------------------------------------")
    #


def demo_langchain_retrieval_qa_chain_2(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        query : str = ""):

    print(f"Call demo_langchain_retrieval_qa_chain llm_model_id={llm_model_id} ")

    # 1. Create Prompt Template for Claude

    prompt_template = """
    The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context. If the AI does not know
    the answer to a question, it truthfully says it does not know.

    Current conversation:
    <conversation_history>
    {context}
    </conversation_history>

    Human:
    <human_reply>
    {question}
    </human_reply>

    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

    # 2. Instantiate Claude Bedrock

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = llm_model_id,
        model_kwargs = llm_model_kwargs,
    )

    # 3. Create RetrievalQA

    chain_type_kwargs = {"prompt": PROMPT}
    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}) #search_kwargs.k defines number of ducuments to search

    # Keep track of questions & Answers
    memory = ConversationBufferMemory(ai_prefix="Assistant")

    # Build the chain
    #conversation = ConversationChain(
    #    llm=cl_llm, 
    #    verbose=True, 
    #    memory=memory,
    #    prompt=claude_prompt
    #)

    qa = RetrievalQA.from_chain_type(llm = llm, 
                                 chain_type = "stuff", 
                                 retriever = retriever, 
                                 chain_type_kwargs = chain_type_kwargs, 
                                 return_source_documents = False)
    
    qa = ConversationalRetrievalChain.from_llm(llm = llm, 
                                chain_type = "stuff", 
                                retriever = retriever, 
                            return_source_documents = False,
                            memory = memory)
    # 4. Invoke

    tools = [
        Tool(
            name="newyorktool",
            func=qa.run,
            description="Use when asked about New York."
        ),
    ]

    chat_agent = initialize_agent(
        tools,
        llm=llm,
        agent = "zero-shot-react-description",
        verbose=True,
        system_message="You are a kind assistant. Provide the answer in Japanese",
        max_iterations = 2,
    )

    result = chat_agent.run(query)
    print(result)