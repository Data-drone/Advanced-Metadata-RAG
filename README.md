# README

## Overview

This project contains two main Python scripts that integrate LangChain with Databricks and MLflow to create a custom Retrieval-Augmented Generation (RAG) chain. The scripts allow for the customization of metadata filters in a Langchain vector search module.

### Scripts

1. **create_custom_retriever_filter_passthrough.py**: 
   - This script defines custom classes that extend the LangChain functionality. The custom classes, `RetrievalQAFilter` and `VectorStoreRetrieverFilter`, enable the passing of custom metadata filters through to the vector store, allowing for more refined and relevant document retrieval based on specific conditions.

2. **build_rag_chain.py**:
   - This script uses the custom classes defined in the first script to build and execute a RAG chain. It sets up the environment, retrieves secrets, configures embedding and chat models, and performs retrieval and question answering.  

2. **deploy_rag_chain.py**:
   - This script deploys the rag chain in the previous script using the new Model As Code paradigm. The results are logged and stored using MLflow, and the model is registered for future use. Please note that that there might be compatability issues between MLFlow and the custom Langchain class which will lead to serlialisation issues. In this case, it is recommended to serlaise the code as custom [mlflow pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) module.

   The Model is deplyoed using the Agent module which removes the need to manage secret scopes. Note that currently the chain's signatures don't match what is needed by the review app so the review app won't work.    

## Requirements

- Python 3.7+
- MLflow
- LangChain
- LangChain Community
- Databricks-specific Python packages for vector search and embeddings langchain

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/dyanmic_filtering_databricks_vector_store.git
    cd dyanmic_filtering_databricks_vector_store
    ```

2. Install the required Python packages:
   *To-Do*

3. Set up your Databricks environment with the required secrets:
   - Create a secret scope in Databricks and store your Databricks host and token as secrets.

## Usage

### Step 1: Create Custom Retriever and Filter

The `create_custom_retriever_filter_passthrough.py` script defines two main classes:

- `RetrievalQAFilter`: A custom class extending LangChain's `RetrievalQA` that allows for using search keywords (`search_kwargs`) to filter and retrieve relevant documents based on custom metadata.

- `VectorStoreRetrieverFilter`: A custom class that extends `VectorStoreRetriever`, enabling the integration of metadata-based filtering in vector-based search processes.

### Step 2: Build and Run the RAG Chain

1. Customize the embedding and chat model parameters:
   
   - Set the `embedding_model_name` and `foundation_model_name` to your specific Databricks model endpoints.

2. Define the prompts and questions:
   - Modify the `PROMPT` variable and `questions` list to customize the input for the question-answering task.

3. Execute the `build_rag_chain.py` script:
    ```bash
    python build_rag_chain.py
    ```

   This script will:
   - Initialize the retriever with custom filters.
   - Run the QA chain with the specified questions.

4. Execute the `deploy_rag_chain` notebook.
   
   This script will:
   - Log the the MLflow model.
   - Register the trained model for future use.
   - Deploy to model serving endpoint with agents package