## RAG System using Zilliz Vector Database

### List of AI models:
* **all-MiniLM-L6-v2** - Embedding model (Best for Clustering or Semantic Search)
  * Maps sentences & paragraphs to **384** dimensional vector space
  * By default, input text longer than **256** word pieces is truncated. 
  * **Model Size:** 22.7 M ; **Tensor Type:** Float 32
* **Llama-3.2-1B-Instruct** - Text generation based on RAG search results
  * Input modalities: Multilingual Text
  * Output modalities: Multilingual Text and code
  * Model size: **1.24B** ;Tensor Type: **BF16**; Context length: **128k**; Token count: **up to 9T**
  * knowledge cutoff: December 2023 

### Instructions to follow:

1. Install dependencies

```
pip install requirements.txt
```

2. To download AI Models and data files using DVC

* Clone this repo and then run below command.

```
$ dvc pull 
```

**Note:** it will pull all the files into cloned directory

Incase of only one single file download 

```
$ dvc get "this repo URL" file or folder name 
```
It will download the corresponding files into local folder.


3. define configuration values inside config.py

4. Start the backend RAG database using FastAPI

```
python main.py
```

**Note:** 
  * open port 2000 in localhost to access backend service

  * Add files into files folder and backend will run automatically

*Hint:* It's almost automated just update files into files folder.