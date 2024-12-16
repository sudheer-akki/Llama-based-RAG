import os
import json
from typing import List
import subprocess
from tqdm import tqdm
from data_validation import TextProcess
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download
from pymilvus import MilvusClient, connections, MilvusException, utility, db, DataType
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")

class GenerateEmbeddings:
    def __init__(
        self,
        device: str,
        folder_name: str ="files",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        model_save_path: str = "embedding_model",
        chunk_length: int = 150, 
        overlap_length: int = 10, 
        stopwords_file: str = "stop_words.txt",
        remove_stop_words: bool = False,
        save_text: bool = False,
        ):
        self.save_text = save_text
        self.model_name = embedding_model
        self.model_save_path = model_save_path
        self.text_process = TextProcess(
            folder_name = folder_name, 
            stopwords_file = stopwords_file,
            remove_stop_words = remove_stop_words,
            save_text=save_text
        )
        self.chunked_content = \
        self.text_process._clean_and_chunk_content(
        chunk_size=chunk_length,
        overlap_size=overlap_length
        )
        self.model = self._load_embeddingsmodel(
            saved_model=self.model_save_path,
            device=device)
        #self.make_embeddings()

    def _download_snapshot(self):
        logger.info(f"Downloading {self.model_name} into {self.model_save_path} folder")  
        #model_name = "sentence-transformers/all-mpnet-base-v2"
        try:
            snapshot_download(repo_id=self.model_name,local_dir=self.model_save_path)
        except Exception as e:
            logger.error(f"Error while downloading {self.model_name} into {self.model_save_path}")
            raise Exception(f"Error while downloading {self.model_name} into {self.model_save_path}: {e}") 


    def _load_embeddingsmodel(self, saved_model, device: str):
        """This is a sentence-transformers model: 
        It maps sentences & paragraphs to a 784 dimensional dense vector space and can be used 
        for tasks like clustering or semantic search."""
        if not os.path.exists(os.path.join(os.getcwd(),saved_model)):
            logger.error(f"Model file {saved_model} not found")
            #raise FileNotFoundError(f"Model file {saved_model} not found")
            self._download_snapshot()
        try:
            logger.info(f"Loading embedding model from {saved_model} folder")
            embedding_model = HuggingFaceEmbeddings(
                cache_folder=f"./{saved_model}",
                model_kwargs = {'device':device}, 
                show_progress = True)
            logger.info(f"{saved_model} loaded successfully")
        except IOError as e:
            logger.error(f"[Error] Unable to load model from {saved_model}")
            raise IOError(f"Error loading model from {saved_model}: {e}")
        return embedding_model

    def _make_embeddings(self, query: str = None, embed_save_file: str ="embeddings.txt" ):
        """Making embeddings from chunked text"""
        logger.info("Generating embeddings from chunked data")
        try:
            if query:
                logger.info("Embedding user query")
                content_to_embed = query
            elif isinstance(self.chunked_content, list) and \
                all(isinstance(i, str) for i in self.chunked_content):
                logger.info(f"Embedding chunked content from 'files' folder")
                content_to_embed = self.chunked_content
            else:
                logger.error(f"[Error] No content available to generate embeddings")
                raise Exception(f"[Error] No content available to generate embeddings")
        except Exception as e:
            logger.error(f"[Error] Error in loading content for embeddings: {e}")
            raise Exception(f"[Error] Error in loading content for embeddings \
                             Provide a query or ensure chunked_content is set.: {e}")
        # Assuming HuggingFaceEmbeddings has an `embed` method to process the chunks#
        embeddings = self.model.embed_documents(content_to_embed)
        if self.save_text:
            logger.info(f"Saving embedded content into {embed_save_file}")
            with open(embed_save_file, "w") as f:
                for embed in embeddings:
                    f.write(str(embed))
            logger.info(f"Saved embedded content into {embed_save_file} successfully")
        if embeddings:
            data = [
                {"id": i, "vector": embeddings[i], "text": content_to_embed[i]}
                for i in tqdm(range(len(embeddings)), desc="Creating embeddings")
            ]
            logger.info(f"Data has {len(data)} entities, each with fields: {list(data[0].keys())}")
            logger.info(f"Vector dim: {len(data[0]['vector']) if data[0]['vector'] else 'N/A'}")
        else:
            logger.error("[Error] Embeddings list is empty. No data created.")
            data = [{"id": None, "vector": None, "text": None}]
        return data, embeddings
    

class VectorDatabase:
    def __init__(self, 
                 embed_model_loaded,
                 db_name: str ="milvusdemo",
                 collection_name: str="rag_collection",
                 embed_dim: int= 768,
                 host: str = "127.0.0.1",
                 port: str = "19530",
                 username: str = "root",
                 password: str = "Milvus",
                 token: str = "root:Milvus",
                 metric_type: str = "COSINE"):
        self.db_name = db_name
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self.host = host
        self.port = port
        self.token = token
        self.username = username
        self.password = password
        self.metric_type = metric_type
        self.gen_embedding = embed_model_loaded
        self._initial_connection_setup()

    def _initial_connection_setup(self):
        logger.info(f"Connecting to http://{self.host}:{self.port}")
        connections.connect(host=self.host,port=self.port)
        self._setup_database_and_collection()

    def _connect_client(self):
        try:
            #elf._connect_database()
            # connecting to client
            logger.info("Connecting to Milvus Client")
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}",
                db_name=self.db_name
            )
            #db.create_database(self.db_name)
            #creating schema
            self.schema = self.client.create_schema(
                auto_id = False,
                enable_dynamic_field=True
            )
            self.schema.add_field(field_name="id",datatype=DataType.INT64, is_primary = True, description="primary id")
            self.schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embed_dim, description="vector")
            self.schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=65535, description="text content")

            self.index_params = self.client.prepare_index_params()
            self.index_params.add_index(
                field_name="id",
                index_type="STL_SORT"
            )
            self.index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT", #Quantization-based index; High-speed query & Requires a recall rate as high as possible
                index_name="vector_index",
                metric_type=self.metric_type, #inner product
                params={ "nlist": 128 } #IVF_FLAT divides vector data into nlist cluster units; Range: [1, 65536]; default value: 128
            )
        except Exception as ex:
            logger.error(f"[Error] Unable to connect to Milvus Client: {ex}")
            raise Exception(f"[Error] Unable to connect to Milvus Client: {ex}")

    def _setup_database_and_collection(self):
        """Consolidated method to set up database"""
        logger.info(f"Checking Database: {self.db_name}")
        try:
            #self._connect_database()
            #connecting to database
            # Check if database exists and create if not
            existing_dbs = db.list_database()
            logger.info(f"[Before] Existing Database: {existing_dbs}")
            if self.db_name not in existing_dbs:
                logger.info(f"Creating Database: {self.db_name}")
                db.create_database(self.db_name)
            else:
                logger.info(f"Loading pre-existed database: {self.db_name}")
                # Switch to the database
                db.using_database(self.db_name)
            logger.info(f"[After] Existing Database: {db.list_database()}")

        except MilvusException as e:
            logger.info(f"[Error] Database connection failed :{e}")
            raise Exception(f"[Error] Database connection failed :{e}")
        self._connect_client()
        # Create collection and insert data
        self._create_collection()

    def _connect_database(self):
        """Establish connection to the database"""
        logger.info(f"Establishing database connection...!!!")
        try:
            #databases = self.client.list_databases()
            #print("Available Databases:", databases)
            logger.info(f"Connecting to {self.db_name}")
            connections.connect(
                alias=self.db_name,
                host=self.host,
                port=self.port
            )
            logger.info(f"{self.db_name} connection has established successfully")
        except MilvusException as e:
            logger.error(f"Unable to connect to Database: {e}")
            raise Exception(f"[Error] Unable to connect to Database: {e}")
    
    def _listout_collections(self):
        """List all collections in the database"""
        try:
            collections = utility.list_collections()
            logger.info(f"List of available collections: {collections}")
        except MilvusException as e:
            logger.error(f"Failed to list collections: {e}")
            raise Exception(f"Failed to list collections: {e}")

    def _create_collection(self):
        """checking and creating collection if not exist"""
        #self._listout_collections()
        #self._connect_database()
        try:
            if not self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=self.schema,
                    index_params=self.index_params,
                    using=self.db_name,
                    # highlight-next
                    consistency_level="Strong",
                )
                res = self.client.get_load_state(
                collection_name=self.collection_name)
                logger.info(f"{self.collection_name} is ready: {res}")
            self._listout_collections()
            # Insert data
            #self._insert_data()
        except MilvusException as e:
            raise Exception(f"[Error] Failed to create {self.collection_name}:{e}")


    def _insert_data(self):
        """Inserting data into collection"""
        logger.info("Inserting embedded data into Milvus server")
        try:
            data,_ = self.gen_embedding._make_embeddings()
            # Check if data is valid for insertion
            if data and any(item["id"] is not None for item in data):
                self.client.insert(
                    collection_name=self.collection_name,  
                    data=data
                )
                logger.info(f"Successfully inserted {len(data)} embedded data into {self.collection_name}")
            else:
                logger.error("[Error] No valid data to insert. Skipping insertion.")
        except Exception as ex:
            logger.error(f"[Error] Unable to insert data to Milvus:{ex}")
            raise Exception(f"[Error] Unable to insert data to Milvus: {ex}")
        
      
    def _search_and_output_query(self, question: str, json_indent:int, response_limit: int):
        """Generates Embeddings and returns retrieved data
        Args:
            question (str): User Query
            response_limit (int): Number of output responses from database
            json_indent (int): Indentation limit for Json output format
        Raises:
            Exception: If unable to retrieve data
        """
        try:
            logger.info(f"Creating embeddings for the User Query")
            _, query_embeddings = self.gen_embedding._make_embeddings(query=question)
            print(f"Embedding sample: {query_embeddings[0][:5]}")
            logger.info(f"User query embedding dim : {len(query_embeddings[0])}")
            logger.info(f"[Important] using: {self.metric_type} metric type")
            search_res = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[query_embeddings[0]],  
            limit=response_limit,  # Return top 3 results
            search_params={"metric_type": self.metric_type,  "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
            )
        except Exception as e:
            logger.error(f"[Error] unable to query search: {e}")
            raise Exception(f"[Error] unable to query search: {e}")
        output = self._get_retrieved_info(search_res=search_res,json_indent=json_indent)
        return output


    def _get_retrieved_info(self, json_indent: int, search_res: str):
        """Retrieving user query from Milvus DB"""
        try:
            retrieved_lines_with_distances = [
                (res["entity"]["text"], res["distance"]) for res in search_res[0]
                ]
            return json.dumps(retrieved_lines_with_distances, indent=json_indent)
        except MilvusException as e:
            logger.error(f"[Error] Unable to output query results: {e}")
            raise Exception(f"[Error] Unable to output query results: {e}")

    def _delete_database_and_collection(self):
        """Delete the entire database"""
        try:
            print(f"Deleting {self.collection_name}")
            self.client.drop_collection(collection_name=self.collection_name)
            print(f"Deleted {self.collection_name}")
            print(f"Deleting {self.db_name}")
            db.drop_database(self.db_name)
            print(f"Database {self.db_name} deleted")
            return db.list_database()
        except MilvusException as e:
            raise Exception(f"[Error] failed to delete {self.db_name}: {e}")
        

if __name__ == "__main__":
    try:
        vector_db = VectorDatabase() # Assuming GenerateEmbeddings() is a class you've defined
        question = "what is Glunet model and how it works ?"
        context_passages = vector_db._search_and_output_query(question=question,response_limit=3, json_indent=3)
        subprocess.run(['rm', '-rf', 'file_list.txt'], check=True)
        vector_db._delete_database_and_collection()
        print(context_passages)
    except Exception as e:
        print(f"An error occurred while initializing VectorDatabase: {e}")