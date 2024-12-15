from vector_embedding import VectorDatabase, GenerateEmbeddings
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    gen_embedding = GenerateEmbeddings(
          embedding_model = "sentence-transformers/all-mpnet-base-v2",
          folder_name = "files",
          model_save_path= "embedding_model",
          stopwords_file = "stop_words.txt",
          device=device,
          remove_stop_words = False,
          save_text= False,
          chunk_length= 250,
          overlap_length = 10)
except Exception as e:
        print(f"An error occurred while loading GenerateEmbeddings: {e}")
                                       
try:
    vector_db = VectorDatabase(
        embed_model_loaded = gen_embedding,
        db_name = "milvusdemo",
        collection_name = "rag_collection",
        embed_dim = 768,
        output_chunk_length = 150,
        host = "127.0.0.1",
        port = "19530",
        username = "root",
        password = "Milvus",
        token = "root:Milvus"
    )
except Exception as e:
        print(f"An error occurred while initializing VectorDatabase: {e}")


if __name__=="__main__":

    vector_db._insert_data()

    query = "what is llama 2 model ?"
    context_passages = vector_db._search_and_output_query(
    question=query, 
    response_limit=3, 
    json_indent=3)
    print(context_passages)
        