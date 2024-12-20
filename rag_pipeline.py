from dotenv import load_dotenv
from langdetect import detect
from config import Config

# Local module imports
from vector_embedding import VectorDatabase, GenerateEmbeddings
from llm_model import TextModel
from utils import filter_response, PROMPT
from logging_config import setup_logger
config_manager = Config()
# Load environment variables
load_dotenv()

class RAGPipeline:
    def __init__(self, vector_db):
        """
        Initialize the RAG (Retrieval-Augmented Generation) Pipeline
        
        Args:
            config (class): Configuration class with pipeline settings
        """
        self.logger = setup_logger(pkgname="rag_database")
        #vector Database intialized
        self.vector_db = vector_db
        self.config = config_manager
        self.upload_data = self.config.config.EMBEDDING_CONFIG.upload_data
        self.response_limit = self.config.config.VECTOR_DB_CONFIG.response_limit

        # Initialize embedding generator
        try:
            self.gen_embedding = self._initialize_embedding_generator()
        except Exception as e:
            self.logger.error(f"Embedding generator initialization failed: {e}")
            raise

        # Insert data into Vector Database
        try:
            if self.upload_data:
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
        return GenerateEmbeddings(**self.config.config.EMBEDDING_CONFIG)

    def _insert_data_db(self):
        data,_ = self.gen_embedding._make_embeddings()
        return self.vector_db._insert_data(data=data)
    
    def retrieve_context(self, Query, json_indent: int = 3):
        """
        Retrieve context for a given query
        
        Args:
            query (str): Input query
            response_limit (int, optional): Number of context passages. Defaults to 5.
        
        Returns:
            list: Retrieved context passages
        """
        try:
            _, query_embeddings = self.gen_embedding._make_embeddings(Query=Query)
            return self.vector_db._search_and_output_query(
                query_embeddings= query_embeddings,
                json_indent=json_indent
            )
        except Exception as e:
            self.logger.error(f"Context retrieval error: {e}")
            raise
    
    
class TextModelPipeline:
    def __init__(self):
        """
        Initialize the Text Generation Model Pipeline
        
        Args:
            config (class): Configuration class with pipeline settings
        """
        self.logger = setup_logger(pkgname="rag_database")
        self.config = config_manager
        # Initialize vector database
        try:
            self.vector_db = self._initialize_vector_database()
        except Exception as e:
            self.logger.error(f"Vector database initialization failed: {e}")
            raise

        try:
            self.rag_pipeline = RAGPipeline(vector_db=self.vector_db)
        except Exception as e:
            self.logger.error(f"RAG pipeline initialization failed: {e}")
            raise

        # Initialize language model
        try:
            self.model = self._initialize_text_model()
        except Exception as e:
            self.logger.error(f"Text model initialization failed: {e}")
            raise

    def _initialize_text_model(self):
        """
        Initialize language model with predefined configurations
        
        Returns:
            TextModel: Configured language model
        """
        return TextModel(**self.config.config.TEXT_MODEL_CONFIG)
    
    def _initialize_vector_database(self):
        """
        Initialize vector database with predefined configurations
        
        Returns:
            VectorDatabase: Configured vector database
        """
        vector_db = VectorDatabase(
            **self.config.config.VECTOR_DB_CONFIG
        )
        return vector_db
    
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
    
    def generate_response(self, Query: str, skip_special_tokens= False, max_retries=3) -> tuple[str]:
        """
        Generate response for a given query
        
        Args:
            query (str): Input query
        
        Returns:
            tuple: Question and answer
        """
        try:
            # Retrieve context
            context_passages = self.rag_pipeline.retrieve_context(Query=Query) #self.retrieve_context(query)
            # Prepare  prompt
            formatted_prompt = PROMPT.format(
                context=self.clean_context(context_passages),
                query=Query
            )
            retries = 0
            # Generate response
            response = self.model.model_response(
                message=formatted_prompt, 
                skip_special_tokens=skip_special_tokens)
            # Check language and retry if it's not English
            while detect(response) != 'en' and retries < max_retries:
                retries += 1
                self.logger.info(f"Response not in English. Retry {retries}/{max_retries}.")
                response = self.model.model_response(message=formatted_prompt, skip_special_tokens=skip_special_tokens)
                if retries == max_retries and detect(response) != 'en':
                    self.logger.info(f"Max retries reached. Returning non-English response.")
            # Filter response
            question, answer = filter_response(output_response=response)
            return question, answer
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            #raise


def main():
    """
    Main execution function
    """
    try:
        Textmodel = TextModelPipeline()
        # Example query
        user_query = "what is Glunet model and how it works?"
        # Generate and print response
        question, answer = Textmodel.generate_response(Query=user_query,skip_special_tokens=False)
        print("------------------------")
        print("\nQuestion:", question)
        print("Answer:", answer)
    
    except Exception as e:
        print(f"Pipeline execution error: {e}")

if __name__ == "__main__":
    main()
     