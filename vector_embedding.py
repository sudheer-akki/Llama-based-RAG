import os
import json
from typing import List
import subprocess
from tqdm import tqdm
import torch
import torch.nn.functional as F
from data_validation import TextProcess
from utils import mean_pooling
from huggingface_hub import snapshot_download
from pymilvus import MilvusClient, MilvusException, utility, db, DataType
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")
from dotenv import load_dotenv
load_dotenv()

class GenerateEmbeddings:
    def __init__(
        self,
        device: str,
        folder_name: str,
        embed_model_name: str,
        embed_model_save_path: str ,
        upload_data: bool,
        chunk_length: int = 200, 
        overlap_length: int = 20, 
        stopwords_file: str = "stop_words.txt",
        remove_stop_words: bool = False,
        save_text: bool = False,
        ):
        self.save_text = save_text
        self.embed_model_name = embed_model_name
        self.embed_model_save_path = embed_model_save_path
        self.folder_name = folder_name
        self.upload_data = upload_data
        if upload_data:
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
        self.tokenizer, self.embed_model = self._load_tokenizer_and_embedmodel(
            embed_model_save_path=self.embed_model_save_path,
            device=device)
        #self.make_embeddings()

    def _download_snapshot(self):
        logger.info(f"Downloading {self.embed_model_name} into {self.embed_model_name} folder")  
        try:
            snapshot_download(repo_id=self.embed_model_name,local_dir=self.embed_model_name)
        except Exception as e:
            logger.error(f"Error while downloading {self.embed_model_name} into {self.embed_model_name}")
            raise Exception(f"Error while downloading {self.embed_model_name} into {self.embed_model_name}: {e}") 
    
    def _load_tokenizer_and_embedmodel(self,embed_model_save_path, device: str):
        if not os.path.exists(os.path.join(os.getcwd(),embed_model_save_path)):
            logger.error(f"Embedding model file {embed_model_save_path} not found")
            self._download_snapshot()
        try:
            logger.info(f"Loading '{self.embed_model_name}' from '{embed_model_save_path}' folder")
            # Load model from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(embed_model_save_path, revision="main",trust_remote_code=True)
            logger.info(f"Tokenizer loaded from {embed_model_save_path}")
            model = AutoModel.from_pretrained(embed_model_save_path, revision="main",trust_remote_code=True)
            logger.info(f"Loaded Embedding model from {embed_model_save_path}")
            logger.info(f"Loading model to {device}")
            model = model.to(device)
            return tokenizer, model
        except IOError as e:
            logger.error(f"[Error] Unable to load model or tokenizer from {embed_model_save_path}")
            raise IOError(f"Error loading model or tokenizer from {embed_model_save_path}: {e}")
                    
    
    def _make_embedding_using_torch(self, sentences: List[str]):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences,
                                       padding=True, 
                                       truncation=True, 
                                        return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()
    

    def _make_embeddings(self, query: str = None, embed_save_file: str ="embeddings.txt" ):
        """Making embeddings from chunked text"""
        logger.info("Generating embeddings from chunked data")
        try:
            if query:
                logger.info("Embedding user query")
                content_to_embed = query
            elif isinstance(self.chunked_content, list) and \
                all(isinstance(i, str) for i in self.chunked_content):
                logger.info(f"Embedding chunked content from '{self.folder_name}' folder")
                content_to_embed = self.chunked_content
            else:
                logger.error(f"[Error] No content available to generate embeddings")
                raise Exception(f"[Error] No content available to generate embeddings")
        except Exception as e:
            logger.error(f"[Error] Error in loading content for embeddings: {e}")
            raise Exception(f"[Error] Error in loading content for embeddings \
                           Provide a query or ensure chunked_content is set.: {e}")
        embeddings = None
        if content_to_embed:
            embeddings = self._make_embedding_using_torch(sentences=content_to_embed)
            if self.save_text:
                logger.info(f"Saving embedded content into {embed_save_file}")
                with open(embed_save_file, "w") as f:
                    for embed in embeddings:
                        f.write(str(embed))
                logger.info(f"Saved embedded content into {embed_save_file} successfully")
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
                Zilliz_CLUSTER_USER,
                Zilliz_CLUSTER_PWD,
                TOKEN,
                URI,
                response_limit,
                db_name: str ="rag_demo",
                collection_name: str="rag_collection",
                vector_field_dim: int= 768,
                metric_type: str = "COSINE"):
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_field_dim = vector_field_dim
        self.Zilliz_CLUSTER_USER = Zilliz_CLUSTER_USER
        self.Zilliz_CLUSTER_PWD = Zilliz_CLUSTER_PWD
        self.TOKEN = TOKEN
        self.URI = URI
        self.metric_type = metric_type
        self.response_limit = response_limit
        #self.gen_embedding = embed_model_loaded
        self._initial_connection_setup()

    def _initial_connection_setup(self):
        self._connect_client()
        logger.info(f"Connected to {self.URI}")
        self._create_collection()

    def _connect_client(self):
        try:
            # connecting to client
            logger.info(f"Connecting to {self.URI}")
            self.client = MilvusClient(
                uri=self.URI,
                token=f"{self.Zilliz_CLUSTER_USER}:{self.Zilliz_CLUSTER_PWD}",
            )
            #creating schema
            self.schema = self.client.create_schema(
                auto_id = False,
                enable_dynamic_field=True
            )
            self.schema.add_field(field_name="id",datatype=DataType.INT64, is_primary = True, description="primary id")
            self.schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_field_dim, description="vector")
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
                params={"nlist": 128 } #IVF_FLAT divides vector data into nlist cluster units; Range: [1, 65536]; default value: 128
            )
        except Exception as ex:
            logger.error(f"[Error] Unable to connect to Milvus Client: {ex}")
            raise Exception(f"[Error] Unable to connect to Milvus Client: {ex}")
    
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
                logger.info(f"Creating collection: {self.collection_name}; vector dimension: {self.vector_field_dim}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=self.schema,
                    consistency_level="Strong",
                )
                res = self.client.get_load_state(
                collection_name=self.collection_name)
                logger.info(f"{self.collection_name} is ready: {res}")
        except MilvusException as e:
            raise Exception(f"[Error] Failed to create {self.collection_name}:{e}")


    def _insert_data(self, data):
        """Inserting data into collection"""
        logger.info("Inserting embedded data into Milvus server")
        try:
            #data,_ = embed_model._make_embeddings()
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
        
      
    def _search_and_output_query(self, query_embeddings: List, json_indent:int):
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
            #_, query_embeddings = self.gen_embedding._make_embeddings(query=question)
            print(f"Embedding sample: {query_embeddings[0][:5]}")
            logger.info(f"User query embedding dim : {len(query_embeddings[0])}")
            logger.info(f"[Important] using: {self.metric_type} metric type")
            search_res = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[query_embeddings[0]],  
            limit=self.response_limit,  # Return top 5 results
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
                (res["entity"]["text"]) for res in search_res[0]
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