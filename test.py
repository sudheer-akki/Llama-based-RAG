from vector_embedding import VectorDatabase
import subprocess
from llm_model import TextModel


question = "what is Glunet ?"


if __name__=="__main__":
    try:
        vector_db = VectorDatabase()
        # Assuming GenerateEmbeddings() is a class you've defined
        context_passages = vector_db._search_and_output_query(question=question)
        print(context_passages)
        subprocess.run(['rm', '-rf', 'file_list.txt'], check=True)
        vector_db._delete_database_and_collection()
        MODEL_NAME = "tiiuae/falcon-rw-1b"     
        model = TextModel(model_name=MODEL_NAME, model_dir="text_model") 
        # Define the prompt template for generating AI responses
        PROMPT = f"""Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {' '.join(context_passages)}

        Question: {question}
        """
        response = model.model_response(message=PROMPT)
        print(response)
    except Exception as e:
        print(f"An error occurred while initializing VectorDatabase: {e}")