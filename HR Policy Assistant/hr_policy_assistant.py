import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import DocxReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from pymilvus import MilvusClient, DataType
import logging

logging.basicConfig(level=logging.ERROR)

#================================
#Config
#================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_api_client = Groq(api_key=GROQ_API_KEY)

model = SentenceTransformer("all-mpnet-base-v2")

milvus_db_client = (
    MilvusClient(uri="http://localhost:19530",
    token="root:Milvus"))

#================================
# Load Documents
#================================

def load_documents(data_dir: str):

    reader = SimpleDirectoryReader(input_dir=data_dir,
                                   recursive=False,
                                   required_exts=[".docx"],
                                   file_extractor=
                                   {".docx": DocxReader()})
    documents = reader.load_data()
    print("Document of Length -> "
          +str(len(documents))+" Loaded Successfully")

    return documents

#================================
# Chunking Documents
#================================

def chunk_documents(documents):

    splitter = (SentenceSplitter
                (chunk_size=512,
                 chunk_overlap=100))
    nodes = (splitter.get_nodes_from_documents
             (documents=documents))
    print("Total Chunk Nodes -> "+str(len(nodes)))

    return nodes

#================================
# Create Vector Database
#================================

def create_vector_database(vector_db_name: str,
                           vector_db_client):
    try:
        if (vector_db_name not in
                vector_db_client.list_databases()):
            vector_db_client.create_database(
                db_name=vector_db_name
            )
        (vector_db_client.use_database
         (db_name=vector_db_name))
    except Exception as e:
        print("Error while Create Vector "
              "Database -> "+str(e))

#================================
# Create Schema
#================================
def create_schema(schema_name: str,
                  vector_db_client):
    schema = vector_db_client.create_schema()

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary= True,
        auto_id=False
    )

    schema.add_field(
        field_name="vector_embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=768
    )

    schema.add_field(
        field_name="data",
        datatype=DataType.VARCHAR,
        max_length=1024
    )

    return schema

#================================
# Create Collection
#================================

def create_collection(collection_name: str,
                      schema,
                      vector_db_client):
    if (collection_name not in
            vector_db_client.list_collections()):
        vector_db_client.create_collection(
            collection_name=collection_name,
            schema=schema
        )

        index_params = vector_db_client.prepare_index_params()
        index_params.add_index(
            field_name="id",
            index_type="AUTOINDEX"
        )

        index_params.add_index(
            field_name="vector_embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        vector_db_client.create_index(
            collection_name=collection_name,
            index_params=index_params)

        print("Collection -> "+str(collection_name+
                            " Successfully Created"))
    else:
        print("Collection -> "+str(collection_name)+
              " Already Exists")

    vector_db_client.load_collection(
        collection_name=collection_name)

#================================
# Generate Embedding
#================================

def generate_embedding(text: str, model):

    return [model.encode(text).astype("float32").tolist()]

#================================
# Insert Vector Embedding
#================================

def insert_embedding(collection_name: str,
                     nodes,
                     model,
                     vector_db_client):

    id = 0
    vector_data = []
    for text_chunk in nodes[:5]:
        embedding = generate_embedding(str(text_chunk),
                                       model)

        vector_data.append({"id": int(id+1),
                            "vector_embedding": embedding,
                            "data":str(text_chunk)})
        id = id + 1

    insert_vector_data = (vector_db_client.
                          insert(collection_name=collection_name,
                                 data=vector_data))
    print("Vector Data Successfully Inserted -> "
          ""+str(insert_vector_data))

#================================
# Generate Query Vector Embedding
#================================

def generate_query_embedding(query: str, model):

    query_embedding = generate_embedding(query, model)

    return query_embedding

# ================================
# Generate Search Results
# ================================

def generate_search_results(collection_name: str,
                            query_embedding,
                            vector_db_client,
                            limit: int):

    query_response = []
    response = vector_db_client.search(
        collection_name=collection_name,
        data=query_embedding,
        limit=limit,
        output_fields=["data"]
    )

    for hits in response:
        print("TopK results:")
        for hit in hits:
            print(hit["entity"]["data"])
            query_response.append(hit["entity"]["data"])

    return query_response

# ================================
# Generate LLM Response
# ================================

def generate_llm_response(query_response: list, groq_client):

    message = [
        {
            "role": "system",
            "content": (
                "You are an HR Policy Assistant. "
                "Your role is to provide accurate, clear, "
                "and professional responses based strictly "
                "on the provided policy context."
            )
        },
        {
            "role": "system",
            "content": (
                "Context:\n"
                f"{query_response}"
            )
        },
        {
            "role": "user",
            "content": (
                "Please generate a well-structured, professional "
                "response based on the above context. "
                "Ensure the answer is concise, clearly "
                "formatted, and directly addresses the query."
            )
        }
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message)
    print(response.choices[0].message.content.strip())

# ================================
# Main Pipeline
# ================================

docs = load_documents("/Users/lordvoldemort/"
                      "PyCharmMiscProject/")

nodes = chunk_documents(docs)

create_vector_database(vector_db_name="hr_policy_assistant",
                       vector_db_client=milvus_db_client)

schema = create_schema(schema_name="hr_policy_schema",
                       vector_db_client=milvus_db_client)

create_collection(collection_name="hr_policy_assistant",
                  schema=schema, vector_db_client=milvus_db_client)

insert_embedding(collection_name="hr_policy_assistant", nodes=nodes, model=model, vector_db_client=milvus_db_client)

query = str(input("Enter HR Policy Query :- "))

query_embedding = generate_query_embedding(query=query,
                                           model=model)

query_response = (generate_search_results
                  (collection_name="hr_policy_assistant",
                   query_embedding=query_embedding,
                   vector_db_client=milvus_db_client, limit=5))

generate_llm_response(query_response=query_response,
                      groq_client=groq_api_client)
