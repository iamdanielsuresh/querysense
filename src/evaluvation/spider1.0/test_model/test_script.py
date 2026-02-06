import os
from openai import AzureOpenAI

# Configuration
# TODO: Replace with your actual values
ENDPOINT = "https://YOUR_RESOURCE_NAME.openai.azure.com/"
API_KEY = "YOUR_API_KEY"
DEPLOYMENT_NAME = "YOUR_FINETUNED_MODEL_DEPLOYMENT_NAME" 

def get_sql_query(question, db_schema):
    """
    Queries the fine-tuned model to translate a natural language question into SQL.
    """
    
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version="2024-02-01"
    )

    system_message = f"""You are a helpful assistant that translates natural language questions into SQL queries.
The database schema is as follows:
{db_schema}"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question}
            ],
            temperature=0, # Lower temperature for deterministic code generation
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during inference: {str(e)}"

if __name__ == "__main__":
    print("--- Fine-tuned Model Test Script ---")
    
    # 1. Define the schema
    # This is a sample schema. You should replace it with the schema relevant to your test.
    sample_schema = """Table: department, columns: [Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees]
Table: head, columns: [head_ID, name, born_state, age]
Table: management, columns: [department_ID, head_ID, temporary_acting]"""
    
    print(f"\nUsing Schema:\n{sample_schema}\n")

    # 2. Define the question
    question = "How many heads of the departments are older than 56?"
    
    print(f"Question: {question}")
    
    # 3. Run Inference
    if "YOUR_API_KEY" in API_KEY:
        print("\n[WARNING] Please update the ENDPOINT, API_KEY, and DEPLOYMENT_NAME in the script before running.")
    else:
        print("\nSending request to model...")
        sql_query = get_sql_query(question, sample_schema)
        print(f"\nGenerated SQL:\n{sql_query}")
