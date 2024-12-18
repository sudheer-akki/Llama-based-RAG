import os
import torch
from dotenv import load_dotenv

# Local module imports
from vector_embedding import VectorDatabase, GenerateEmbeddings
from llm_model import TextModel
from utils import filter_response, PROMPT
from logging_config import setup_logger

# Load environment variables
load_dotenv()

# Configuration constants
class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
    MODEL_DIR = os.getenv("MODEL_DIR", "Llama-3.2-1B")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Embedding model configuration
    EMBEDDING_CONFIG = {
        "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "folder_name": "files",
        "embed_model_save_path": "all-MiniLM-L6-v2",
        "stopwords_file": "stop_words.txt",
        "device": "cpu",
        "remove_stop_words": False,
        "save_text": False,
        "chunk_length": 200,
        "overlap_length": 20
    }

    # Vector DB configuration
    VECTOR_DB_CONFIG = {
        "db_name": "rag_demo",
        "collection_name": "rag_collection",
        "vector_field_dim": 384,
        "host": "127.0.0.1",
        "port": "19530",
        "username": "root",
        "password": "Milvus",
        "token": "root:Milvus",
        "metric_type": "COSINE"
    }

class RAGPipeline:
    def __init__(self, config=Config, rag_response_limit: int = 5,upload_data: bool = False):
        """
        Initialize the RAG (Retrieval-Augmented Generation) Pipeline
        
        Args:
            config (class): Configuration class with pipeline settings
        """
        self.logger = setup_logger(pkgname="rag_database")
        self.config = config
        self.response_limit = rag_response_limit

        # Initialize embedding generator
        try:
            self.gen_embedding = self._initialize_embedding_generator()
        except Exception as e:
            self.logger.error(f"Embedding generator initialization failed: {e}")
            raise

        # Initialize vector database
        try:
            self.vector_db = self._initialize_vector_database()
        except Exception as e:
            self.logger.error(f"Vector database initialization failed: {e}")
            raise

        # Initialize language model
        try:
            self.model = self._initialize_text_model()
        except Exception as e:
            self.logger.error(f"Language model initialization failed: {e}")
            raise

        # Insert data into Vector Database
        try:
            if upload_data:
                self.logger.info("[IMPORTANT] Uploading data into Vector DB")
                self._insert_data_db()
            else:
                self.logger.info("[IMPORTANT] Not uploading new data into Vector DB")
        except Exception as e:
            self.logger.error(f"Data insertion failed: {e}")
            raise


    def _initialize_embedding_generator(self):
        """
        Initialize embedding generator with predefined configurations
        
        Returns:
            GenerateEmbeddings: Configured embedding generator
        """
        return GenerateEmbeddings(**self.config.EMBEDDING_CONFIG)
    
    def _initialize_vector_database(self):
        """
        Initialize vector database with predefined configurations
        
        Returns:
            VectorDatabase: Configured vector database
        """
        vector_db = VectorDatabase(
            embed_model_loaded=self.gen_embedding,
            **self.config.VECTOR_DB_CONFIG
        )
        return vector_db
    
    def _initialize_text_model(self):
        """
        Initialize language model with predefined configurations
        
        Returns:
            TextModel: Configured language model
        """
        return TextModel(
            model_name=self.config.MODEL_NAME,
            model_dir=self.config.MODEL_DIR,
            device=self.config.DEVICE,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.6,
            top_k=None,
            num_return_seq=1,
            rep_penalty=2.5,
            do_sample=True
        )
    
    def _insert_data_db(self):
        return self.vector_db._insert_data()
    
    @staticmethod
    def clean_context(context_passages):
        """
        Clean and format context passages
        
        Args:
            context_passages (list): List of context passages
        
        Returns:
            str: Cleaned context string
        """
        return ''.join([
            str(passage).replace('"', '').replace(',', '').replace("[", "").replace("]", "") 
            for passage in context_passages
        ])
    
    def retrieve_context(self, query, json_indent: int = 3):
        """
        Retrieve context for a given query
        
        Args:
            query (str): Input query
            response_limit (int, optional): Number of context passages. Defaults to 5.
        
        Returns:
            list: Retrieved context passages
        """
        try:
            return self.vector_db._search_and_output_query(
                question=query,
                response_limit=self.response_limit,
                json_indent=json_indent
            )
        except Exception as e:
            self.logger.error(f"Context retrieval error: {e}")
            raise

    def generate_response(self, query: str, skip_special_tokens= False) -> tuple[str]:
        """
        Generate response for a given query
        
        Args:
            query (str): Input query
        
        Returns:
            tuple: Question and answer
        """
        try:
            # Retrieve context
            context_passages = self.retrieve_context(query)
            
            # Prepare prompt
            formatted_prompt = PROMPT.format(
                name="Sora",
                Context=self.clean_context(context_passages),
                query=query
            )
            
            # Generate response
            response = self.model.model_response(
                message=formatted_prompt, 
                skip_special_tokens=skip_special_tokens
            )
            
            # Filter response
            question, answer = filter_response(output_response=response)
            
            return question, answer
        
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            raise

def main():
    """
    Main execution function
    """
    try:
        # Initialize pipeline
        rag_pipeline = RAGPipeline(upload_data = False)
        
        # Example query
        query = "what is Glu-net model and how it works?"
        
        # Generate and print response
        question, answer = rag_pipeline.generate_response(query)
        
        print("Question:", question)
        print("Answer:", answer)
    
    except Exception as e:
        print(f"Pipeline execution error: {e}")

if __name__ == "__main__":
    main()
     