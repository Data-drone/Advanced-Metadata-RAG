# Databricks notebook source
# MAGIC %pip install mlflow==2.17.2 databricks-vectorsearch databricks-langchain databricks-agents langchain==0.3.7 langchain-community==0.3.7
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Libs
import mlflow
import langchain
import pandas as pd
import time
import os


from databricks.vector_search.client import VectorSearchClient
from mlflow.models import infer_signature
from mlflow.client import MlflowClient

from langchain_community.vectorstores import DatabricksVectorSearch
from databricks_langchain import DatabricksEmbeddings
from databricks_langchain import ChatDatabricks
from langchain_core.prompts import PromptTemplate

# TODO need a fix for pasting all this across
from create_custom_retriver_filter_passthrough import RetrievalQAFilter, VectorStoreRetrieverFilter

import warnings
from warnings import filterwarnings
filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Model  configs
# THOSE ARE OUR EXTENDED LANGCHAIN MODULES TO PASS THE FILTER ARGUMENTS THROUGH THE CHAIN
embedding_model_name = "databricks-gte-large-en"
foundation_model_name = "databricks-meta-llama-3-1-405b-instruct"
vector_search_endpoint = 'one-env-shared-endpoint-14'

catalog = 'brian_ml_dev'
schema = 'poc_dod'
vector_search_index = f'{catalog}.{schema}.silver_refined_dsgl_index'

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()

# scope = "natasha_secrets"
# w.secrets.delete_secret(scope=scope, key="DATABRICKS_TOKEN")
# w.secrets.put_secret(scope=scope, key="DATABRICKS_TOKEN", string_value="foo")
# w.secrets.list_secrets(scope=scope)

#os.environ["DATABRICKS_TOKEN"] = ""
#os.environ["DATABRICKS_HOST"] = ""

# COMMAND ----------

# DBTITLE 1,Setup Embeddings
# LOAD EMBEDDING AND CHAT MODELS
embedding_model = DatabricksEmbeddings(endpoint=embedding_model_name)

# COMMAND ----------

# DBTITLE 1,Setup Model
# openai_model = "openai-4mni "
chat_model = ChatDatabricks(
    endpoint=foundation_model_name, temperature=0.5, max_tokens=2000)


# COMMAND ----------

# DBTITLE 1,Setup Retriever
# insert prompts and questions
def get_retriever_filter(persist_dir: str = None):
    #token = os.environ["DATABRICKS_TOKEN"]
    #host = os.environ["DATABRICKS_HOST"]
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=vector_search_index
    )
    # Adjust text_column that contains chunk based on metadata
    vectorstore = DatabricksVectorSearch(
        index, text_column="llm_parsed_data", embedding=embedding_model, columns=["llm_parsed_data", "section"]
    )
    return vectorstore

# COMMAND ----------

# DBTITLE 1,Setup Prompt
PROMPT = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer. End your response with "Thank you for your attention ML SMEs

{context}

Human: {question}
Chatbot:"""


questions = ["What are audit logs?", "Why do audit logs matter?"]

# COMMAND ----------

# DBTITLE 1,Build & Test Retriever
prompt = PromptTemplate(template=PROMPT, input_variables=[
                  "context", "question"])

# This is to instantiate the VS and can later be overwritten in filter_custom
search_spec = {"num_results": 3} #TO-DO: add score threshold which has been buggy

retriever_custom = VectorStoreRetrieverFilter(vectorstore=get_retriever_filter(),
                                            search_type="similarity",
                                            search_kwargs=search_spec
                                            )

# retriever_custom._get_relevant_documents()
qa = RetrievalQAFilter.from_chain_type(
  llm=chat_model,
  chain_type="stuff",  # TO CHECK
  retriever=retriever_custom,
  chain_type_kwargs={"prompt": prompt},
  return_source_documents=True,
  #verbose=True
)

filter_custom = {"num_results": 5, "filters": {
  "section": "dual_use_cat_6"}}

question_filtered = {
    "query": questions[0],
    "search_kwargs": filter_custom  # Only if search_kwargs is expected
}


result = qa(question_filtered)
print(result)

# COMMAND ----------

# DBTITLE 1,Build Chain Logic
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


chain = (
    {'query': itemgetter("messages") | RunnableLambda(extract_user_query_string),
     'search_kwargs': itemgetter("search_kwargs")}
    | qa
)


# COMMAND ----------

chain.invoke(
    {
        "messages": [{'role': 'human', 'content': 'hi'}],
        "search_kwargs": {"num_results": 5, 
                          "filters": {"section": "dual_use_cat_6"}}
    }
)

# COMMAND ----------

# DBTITLE 1,Setup Model As Code
mlflow.models.set_model(model=chain)


# COMMAND ----------

