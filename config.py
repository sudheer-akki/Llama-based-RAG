import os
import torch
from box import Box
from dotenv import load_dotenv
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")
load_dotenv()

class Config:
    def __init__(self):
        # Load environment variables only once
        MODEL_NAME= os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
        MODEL_DIR= os.getenv("MODEL_DIR", "Llama-3.2-1B-Instruct")
        FOLDER = os.getenv("FOLDER","files")
        Zilliz_CLUSTER_USER = os.getenv("Zilliz_CLUSTER_USER")
        Zilliz_CLUSTER_PWD = os.getenv("Zilliz_CLUSTER_PWD")
        TOKEN = os.getenv("TOKEN")
        URI = os.getenv("URI") 
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration constants
        self.config = Box({
            "MODEL_NAME": MODEL_NAME,
            "MODEL_DIR": MODEL_DIR,
            "FOLDER": FOLDER,
            "Zilliz_CLUSTER_USER": Zilliz_CLUSTER_USER,
            "Zilliz_CLUSTER_PWD": Zilliz_CLUSTER_PWD,
            "TOKEN": TOKEN,
            "URI": URI,
            "DEVICE": DEVICE,

            # Embedding model configuration
            "EMBEDDING_CONFIG" : {
                "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "folder_name": FOLDER,
                "embed_model_save_path": "all-MiniLM-L6-v2",
                "stopwords_file": "stop_words.txt",
                "device": DEVICE,
                "remove_stop_words": False,
                "save_text": False,
                "chunk_length": 200,
                "overlap_length": 10,
                "upload_data": True
            },

            # Vector DB configuration
            "VECTOR_DB_CONFIG" : {
                "db_name": "rag_demo",
                "collection_name": "rag_collection",
                "vector_field_dim": 384,
                "Zilliz_CLUSTER_USER":Zilliz_CLUSTER_USER,
                "Zilliz_CLUSTER_PWD": Zilliz_CLUSTER_PWD,
                "TOKEN":TOKEN,
                "URI":URI,
                "response_limit": 5,
                "metric_type": "COSINE"
            },

            "TEXT_MODEL_CONFIG" : {
                "model_name": MODEL_NAME,
                "model_dir": MODEL_DIR,
                "device": DEVICE,
                "temperature": 0.5,
                "top_p": 0.6,
                "top_k": None,
                "num_return_seq": 1,
                "rep_penalty": 2.5,
                "max_token": 512,
                "do_sample": True,
                "bitsandbytes":True
            }
        })