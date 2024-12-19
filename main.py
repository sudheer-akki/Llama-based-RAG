import os
import asyncio
import urllib.parse
import uvicorn
from fastapi import FastAPI, Query, Depends
from config import Config
from typing import Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
from watchfiles import awatch

from rag_pipeline import RAGPipeline, TextModelPipeline
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")
config_manager = Config()
load_dotenv()



@dataclass
class AppState:
    rag_pipeline: Optional[RAGPipeline] = None
    text_model: Optional[TextModelPipeline] = None
    vector_db: Optional[any] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Intialize State
    app.state.components = AppState()
    #Intialize text model and vector DB
    try:
        app.state.components.text_model = TextModelPipeline()
        app.state.components.vector_db = app.state.components.text_model._initialize_vector_database()
    except Exception as e:
        logger.info(f"An error occured while loading Text Model Pipeline: {e}")

    #Start File watcher
    watch_task = asyncio.create_task(watch_folder(app))

    yield

    try:
        #cleanup
        watch_task.cancel()  # Stop the file watcher
        await watch_task  # Wait for it to finish
    except asyncio.CancelledError:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(lifespan=lifespan)


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
    #logger.info(f"Allowed origins: {origins}")
    # Remove any empty origins
    return [origin for origin in origins if origin]

def setup_cors(app):
    """Setup CORS middleware with allowed origins."""
    origins = get_origins()
    logger.info(f"Allowed origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
setup_cors(app)

async def watch_folder(app: FastAPI):
    """
    Watch for file changes and reload the RAG pipeline.
    This also triggers an initial execution at the start.
    """
    # Trigger the initial execution to load the RAG pipeline
    logger.info("[IMPORTANT] Loaded RAG Pipeline (Initial Load)")
    app.state.components.rag_pipeline = RAGPipeline(
        vector_db=app.state.components.vector_db
    )
    try:
        async for changes in awatch(config_manager.config.FOLDER):
            app.state.components.rag_pipeline = RAGPipeline(
                vector_db=app.state.components.vector_db
            )
            logger.info("[IMPORTANT] Loaded RAG Pipeline")
    except Exception as e:
        logger.error(f"An error occurred while loading RAG pipeline: {e}")


def get_components() -> AppState:
    """Get application components state."""
    return app.state.components

@app.on_event("startup")
async def startup_event():
    origins = get_origins()  # Reuse the same function to get origins
    # Log the origins with more context
    logger.info("CORS Configuration on Startup:")
    logger.info(f"Starting server with {len(origins)} allowed origins")
    # Validate origins
    for origin in origins:
        try:
            parsed_url = urllib.parse.urlparse(origin)
            if all([parsed_url.scheme, parsed_url.netloc]):
                logger.info(f"Allowed origin: {origin}")
            else:
                logger.warning(f"Invalid origin format: {origin}")
        except Exception as e:
            logger.error(f"Error parsing origin {origin}: {e}", exc_info=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health(components: AppState = Depends(get_components)):
    return {
        "status": "OK",
        "components": {
            "rag_pipeline": app.state.components.rag_pipeline is not None,
            "text_model": app.state.components.text_model is not None,
            "vector_db": app.state.components.vector_db is not None
        }
    }

@app.get("/search")
async def get_response(
    query: str = Query(...),
    components: AppState = Depends(get_components)
    ) -> dict:
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
            if not components.text_model:
                raise ValueError("Text model not initialized")
            
            question, answer = components.text_model.generate_response(
                query=query,
                skip_special_tokens=False
            )
            return {"result": answer}
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"error": "An error occurred while processing your request"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 2000)),
        reload_includes=["config.py"],
        reload=False
    )
 
 