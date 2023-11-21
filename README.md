# langchain-python

### Langchain examples in Python using AWS Bedrock LLM

##### Basic Examples

<ol>
<li>[langchain](demo_bedrock_langchain.py)</li>
<li>[langchain_chat](demo_bedrock_langchain_chat.py)</li>
</ol>

##### Document Retrievers


##### Vector Store

<ol>
<li>[vector_store](demo_bedrock_langchain_vector_store.py)</li>
</ol>

##### Document Transformation

##### LLM Chains

<ol>
<li>[langchain_chain](demo_bedrock_langchain_chain.py)</li>
</ol>

<hr/>

#### Freeze

pip freeze > requirements.txt

#### Python Package Installation

pip install --upgrade <package-name>

pip install --upgrade langchain

##### Pydantic

pip list | grep langchain
pip list | grep pydantic

##### Document Loader / Retriever

pip install pypdf
pip install wikipedia
pip install tiktoken

##### Vector DB

pip install chromadb

##### Agents - SerpApi

pip install google-search-results

##### OpenSearch

pip install opensearch-py
pip install boto3
pip install botocore
pip install requests-aws4auth

<hr/>

##### Links

- [botocore.client.OpenSearchServiceServerless](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/opensearchserverless.html)

- https://github.com/langchain-ai/langchain

- https://github.com/aws-samples/aws-genai-llm-chatbot

- https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html

- https://catalog.us-east-1.prod.workshops.aws/workshops/f8d2c175-634d-4c5d-94cb-d83bbc656c6a/en-US/41-vector

- https://github.com/opensearch-project/opensearch-py/blob/main/samples/knn/knn-basics.py

- https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html

##### Errors

- opensearchpy.exceptions.AuthorizationException: AuthorizationException(403, 'Forbidden')


##### Blogs

- https://aws.amazon.com/blogs/big-data/amazon-opensearch-services-vector-database-capabilities-explained/


##### Repost

- https://repost.aws/zh-Hant/questions/QUxXol2_SQRb-7iYoouyjl8A/questions/QUxXol2_SQRb-7iYoouyjl8A/aws-opensearch-serverless-bulk-api-document-id-is-not-supported-in-create-index-operation-request?