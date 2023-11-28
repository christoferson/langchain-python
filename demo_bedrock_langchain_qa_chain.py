
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
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain, StuffDocumentsChain

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
    #demo_langchain_retrieval_qa_chain_2(bedrock_runtime, llm_model_id="anthropic.claude-v2", query=query)
    demo_langchain_retrieval_qa_chain_4(bedrock_runtime, llm_model_id="anthropic.claude-v2", query=query)
    #demo_langchain_converse_retrieval_chain_1(bedrock_runtime, llm_model_id="anthropic.claude-v2", query=query)



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

    qa = RetrievalQA.from_chain_type(llm = llm, 
                                 chain_type = "stuff", 
                                 retriever = retriever, 
                                 chain_type_kwargs = chain_type_kwargs, 
                                 return_source_documents = False)
    
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


def demo_langchain_retrieval_qa_chain_3(bedrock_runtime, 
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



def demo_langchain_retrieval_qa_chain_4(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        query : str = ""):

    print(f"Call demo_langchain_retrieval_qa_chain_4 llm_model_id={llm_model_id} ")

    VerboseFlag = False

    # 1. Create Prompt Template for Claude

    prompt_template = """The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Do not use any XML tags in the answer.

    Use the following context (delimited by <conversation_context></conversation_context>) and 
    the chat history (delimited by <conversation_history></conversation_history>) to answer the question:

    <conversation_context>
    {context}
    </conversation_context>

    <conversation_history>
    {history}
    </conversation_history>

    Based on the above, please answer the following question: {question}

    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'history', 'question']
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

    chain_type_kwargs = {
        "prompt": PROMPT,
        "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question",
                ai_prefix="Assistant",
                human_prefix="Human",
                verbose=VerboseFlag,
        )
    }
    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}) #search_kwargs.k defines number of ducuments to search

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
                                 return_source_documents = False, 
                                 verbose=VerboseFlag)
    

    # 4. Invoke
    #print("-----------------------------------------------------")
    #result = qa.run(query)
    #print(result)

    print("-----------------------------------------------------")
    query_list = [
        query,
        "New York was named in honor of who?",
        "What was the old name of New York in the year 1627?"
    ]

    for query in query_list: 
        result = qa.run(query)
        print(query)
        print(result)



########################################################################################

def demo_langchain_converse_retrieval_chain_1(bedrock_runtime, 
                        embedding_model_id : str = "amazon.titan-embed-text-v1", 
                        llm_model_id : str = "anthropic.claude-instant-v1", 
                        llm_model_kwargs : dict = { "temperature": 0.0 },
                        query : str = ""):

    print(f"Call demo_langchain_retrieval_qa_chain llm_model_id={llm_model_id} ")

    # 1. Create Prompt Template for Claude

    template_qg = """
    次の会話に対しフォローアップの質問があるので、フォローアップの質問を独立した質問に言い換えなさい。
            
    チャットの履歴:
    {chat_history}
            
    フォローアップの質問:
    {question}
            
    言い換えられた独立した質問:"""

    prompt_qg = PromptTemplate(
            template=template_qg,
            input_variables=["chat_history", "question"],
            output_parser=None,
            partial_variables={},
            template_format='f-string',
            validate_template=True,
    )
#####
    prompt_template_qa = """You are a helpful assistant. If the context is not relevant, please answer the question by using your own knowledge about the topic. 

    {context}

    Question: {question}
    Answer:"""

    prompt_qa = PromptTemplate(
            template=prompt_template_qa, 
            input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt_qa}

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

    #combine_docs_chain = StuffDocumentsChain()
    #chain_type_kwargs = {"prompt": PROMPT}
    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}) #search_kwargs.k defines number of ducuments to search

    # Keep track of questions & Answers
    memory = ConversationBufferMemory(ai_prefix="Assistant")
    
    
    # 4. Invoke

    question_generator = LLMChain(llm=llm, prompt=prompt_qg)
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_qa)

    qa = ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
    )

    ##
    questions = [
        "What is the origin of the name New York?",
        "When was the city New York Established?",
    ]
    chat_history = []
    ###
    result = qa.run({"question": query, "chat_history": chat_history})
    print(f"Q: {query}")
    print(f"A: {result}")
    #for question in questions:
    #    print(f"Q: {question}")
    #    result = qa.run({"question": question, "chat_history": chat_history})
    #    print(f"A: {result}")
    #    chat_history.append((question, result))