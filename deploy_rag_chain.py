# Databricks notebook source
# MAGIC %pip install mlflow==2.17.2 databricks-vectorsearch databricks-langchain databricks-agents langchain==0.3.7 langchain-community==0.3.7
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup Config Vars
catalog = 'brian_ml_dev'
schema = 'poc_dod'
model_name = 'filter_test_model'


# COMMAND ----------

# DBTITLE 1,Setup Libs
import mlflow
import os
from mlflow.models import infer_signature


# Log the model in MLflow
with mlflow.start_run(run_name='brian_test') as run:

    input_test = {
        "messages": [{'role': 'human', 'content': 'hi'}],
        "search_kwargs": {"num_results": 5, 
                          "filters": {"section": "dual_use_cat_6"}}}

    signature = infer_signature(input_test, "Happy to help!")

    logged_model = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), 'build_rag_chain'
        ),  # Chain code file e.g., /path/to/the/chain.py
        artifact_path="chain",
        input_example=input_test,
        code_paths=['create_custom_retriver_filter_passthrough.py'],
        signature=signature)    


# COMMAND ----------

# Load the model from MLflow
#model_uri = f'runs:/{run.info.run_uuid}'
loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

# Define the input for inference
question_filtered =     {
        "messages": [{'role': 'human', 'content': 'hi'}],
        "search_kwargs": {"num_results": 5, 
                          "filters": {"section": "dual_use_cat_6"}}
    }



# Run inference
result = loaded_model.invoke(question_filtered)

# Print the result
print(result)


# COMMAND ----------

# DBTITLE 1,Deploy chain
from databricks import agents

UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# Use Unity Catalog to log the chain
mlflow.set_registry_uri('databricks-uc')

# Register the chain to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_model.model_uri, name=UC_MODEL_NAME)

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version,
                                )

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

