import subprocess
from rag_pipeline import RAGPipeline
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
        global rag_pipeline
        rag_pipeline = RAGPipeline(rag_response_limit=3)
        logger.info("[IMPORTANT] Loaded RAG Pipeline")      
    except Exception as e:
        logger.error(f"An error occurred while loading RAG pipeline: {e}")

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
            question, answer = rag_pipeline.generate_response(query, skip_special_tokens=False)
            #response = model.model_response(message=PROMPT, skip_special_tokens=False)
            return {"result": answer}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": "An error occurred while processing your request"}


if __name__=="__main__":
    subprocess.run(['uvicorn', 'main:app', '--host','localhost', '--port', '2000','--reload', '--reload-dir','files'], check=True)

       
 