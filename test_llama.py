import os
from llm_model import TextModel
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME","meta-llama/Llama-3.2-1B") 
MODEL_DIR = os.getenv("MODEL_DIR","Llama-3.2-1B")

if __name__=="__main__":

    query = "what is glunet model ?"
    
    try:
        model = TextModel(model_name=MODEL_NAME, model_dir=MODEL_DIR) 
        # Define the prompt template for generating AI responses
        response = model.model_response(message=query)
        print(response)
    except Exception as e:
        print(f"An error occurred while initializing VectorDatabase: {e}")