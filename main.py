from vector_embedding import VectorDatabase, GenerateEmbeddings
import subprocess
from llm_model import TextModel
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.parse
from dotenv import load_dotenv
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")
app = FastAPI()

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_DIR = os.getenv("MODEL_DIR")  

def get_origins():
    # Get the frontend URL from environment variables with fallback options
    frontend_url = os.getenv("FRONTEND_URL", "https://www.askmeai.de")
    frontend_dns = os.getenv("FRONTEND_DNS", "https://www.askmeai.de")
    dev_url = os.getenv("DEV_URL", "")
    
    # Build the list of origins
    origins = [
        frontend_url,  # Custom domain from environment
         frontend_dns,
         dev_url 
    ]
    
    # Remove any empty origins
    origins = [origin for origin in origins if origin]
    return origins
    # Return the list of origins

# Add CORS middleware with the origins list
origins = get_origins()  # Call the function to get origins
logger.info(f"Allowed origins: {origins}")
# Add CORS middleware with the origins list
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins[-1],  # Use the origins list
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Startup event to log origins
@app.on_event("startup")
async def startup_event():
    #origins = get_origins()  # Reuse the same function to get origins
    # Log the origins with more context
    logger.info("CORS Configuration on Startup:")
    logger.info(f"Number of allowed origins: {len(origins)}")

     # Define the sources for each origin
    origin_sources = {
        "DEV_URL": origins[-1]
    }
    # Log each origin with its source
    for key, item in origin_sources.items():
        logger.info(f"domain -> {item}; Source -> {key}")
    
    # Optional: Validate origins
    for origin in origins:
        try:
            parsed_url = urllib.parse.urlparse(origin)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                logger.warning(f"Invalid origin format: {origin}")
        except Exception as e:
            logger.error(f"Error parsing origin {origin}: {e}", exc_info=True)
    """ This function will run when the FastAPI application starts. """                      
    try:
        global vector_db
        vector_db = VectorDatabase(
            embed_model_loaded = GenerateEmbeddings(
                device=device,
                chunk_length=150 # output chunk length
            ),
            db_name = "milvusdemo",
            collection_name = "rag_collection",
            embed_dim = 768,
            host = "127.0.0.1",
            port = "19530",
            username = "root",
            password = "Milvus",
            token = "root:Milvus"
        )
    except Exception as e:
            print(f"An error occurred while initializing VectorDatabase: {e}")

    try:
        #"tiiuae/falcon-rw-1b" 
        global model
        model = TextModel(model_name=MODEL_NAME,
                            model_dir=MODEL_DIR,
                            device= device,
                            max_tokens = 2096, #max_token=1,28,000
                            temperature = 0.2, 
                            top_p = 0.6,
                            top_k = None,
                            num_return_seq = 2,
                            rep_penalty = 2.5,
                            do_sample = True) 

    except Exception as e:
        print(f"An error occurred while loading text model: {e}")

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health():
    return {"status": "OK"}


@app.get("/search")
async def get_response(query: str = Query(...)) -> dict:
    """
    Handle search queries by generating response using text model
    
    Args:
        query(str): Input query in string format.
    Returns:
        dict: A dictionary containing response under key "result"
    Raises:
        Exeception: If there is an error during model response generation.
    """
    try:
        context_passages = vector_db._search_and_output_query(
        question=query, 
        response_limit=3, 
        json_indent=3)
        #print(context_passages)
        #subprocess.run(['rm', '-rf', 'file_list.txt'], check=True)
        #vector_db._delete_database_and_collection()
        PROMPT = f"""Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {' '.join(context_passages)}

        Question: {query}
        """
        response = model.model_response(message=PROMPT, skip_special_tokens=False)
        #response = model.model_response(message=query)
        #response = f"{query}, How are you ?"
        return {"result": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {"error": "An error occurred while processing your request"}


if __name__=="__main__":
    subprocess.run(['uvicorn', 'main:app', '--host','localhost', '--port', '2000', '--reload-dir','files'], check=True)

       
 