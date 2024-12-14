## RAG System using Milvus Vector Database

### prerequisites:

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
docker compose up -d
```
**Note:** Open http://${HOST_IP}:8000 on your local browser to access Attu GUI

3. Specify your question for RAG search inside test.py 

For example,

**question** = "what is your name ? 

4. Start the RAG system using below command

```
python test.py
```

