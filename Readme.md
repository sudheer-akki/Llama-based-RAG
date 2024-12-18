## RAG System using Milvus Vector Database

### List of AI models:
* **all-MiniLM-L6-v2** - Embedding model (Best for Clustering or Semantic Search)
  * Maps sentences & paragraphs to **384** dimensional vector space
  * By default, input text longer than **256** word pieces is truncated. 
  * **Model Size:** 22.7 M ; **Tensor Type:** Float 32
* **Llama-3.2-1B (text only)** - Text generation based on RAG search results
  * Input modalities: Multilingual Text
  * Output modalities: Multilingual Text and code
  * Model size: **1.24B** ;Tensor Type: **BF16**; Context length: **128k**; Token count: **up to 9T**
  * knowledge cutoff: December 2023 

### Prerequisites:

* Install Milvus docker standalone server 

```
$ wget https://github.com/milvus-io/milvus/releases/download/v2.5.0-beta/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Note: It will copy docker-compose file to the folder

* Install attu: The GUI for Milvus server

**Note:** Adding directly attu as a service to the docker compose file like below

```
  attu:
    container_name: attu
    image: zilliz/attu:latest
    ports:
      - 8000:3000
    environment:
      - HOST_URL= http://${HOST_IP}:8000
      - MILVUS_URL= ${HOST_IP}:19530
  ````

***Create .env file and replace the HOST_IP with your machine IP address***

### Instructions to follow:

1. add any .pdf or .txt files into new_files folder

2. start the Milvus and Attu docker containers using below command

```
sudo docker compose up -d
```
**Note:** Open http://${HOST_IP}:8000 on your local browser to access Attu GUI

3. To update Milvus DB with local files data

```
python update_db.py
```

4. To perform semantic search

* open update_db.py and update query

* For example,

  **query** = "what is your name ?

```
python update_db.py
``` 

4. To test **Llama-3.2-1B**

* open test_llama.py and update query

* For example,

  **query** = "what is your name ?

```
python test_llama.py
```

5. To run backend with FastAPI

**Note:** Make sure to run Milvus container

```
python main.py
```
* open port 2000 in localhost to access backend service


5. Stop the Milvus server

```
sudo docker compose down
```

6. Delete Milvus service data

```
$ sudo rm -rf volumes
```

**Note:** In the last, make sure to stop the running Milvus container

7. To download AI Models and data files using DVC

* clone this repo and then run below command.

```
$ dvc pull 
```
**Note:** it will pull all the files into cloned directory

Incase of only one single file download 

```
$ dvc get "this repo URL" file or folder name 
```
It will download the corresponding files into local folder.
