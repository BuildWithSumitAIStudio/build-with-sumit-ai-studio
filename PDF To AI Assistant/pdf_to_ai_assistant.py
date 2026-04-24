import os
import warnings
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser)
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding)
from llama_index.core import (SimpleDirectoryReader,
                              StorageContext,
                              VectorStoreIndex)
from llama_index.vector_stores.milvus import (
    MilvusVectorStore)
from llama_index.llms.groq import Groq

warnings.filterwarnings("ignore",
                        category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =================================================
# CONFIG
# =================================================

GROQ_API_KEY = os.getenv("groq_api_key")
groq_api_client = (
    Groq(model="llama-3.3-70b-versatile",
                       api_key=GROQ_API_KEY))

embed_model = (
    HuggingFaceEmbedding
    (model_name=
     "sentence-transformers/all-MiniLM-L6-v2",
     embed_batch_size=32))

# =================================================
# STEP 1: Read document
# =================================================

text_pdf = (SimpleDirectoryReader
            (input_files=
             ["large_text_document.pdf"]).
            load_data())

# =================================================
# STEP 2: Split Document
# =================================================

splitter = SemanticSplitterNodeParser(buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model)
text_split = (splitter.get_nodes_from_documents
              (text_pdf))

# =================================================
# STEP 3: Create Vector DB
# =================================================

vector_store = (MilvusVectorStore
                (uri="http://localhost:19530",
                 dim=384,
                 collection_name="pdf_collection"))

# =================================================
# STEP 4: Create Storage Context
# =================================================

storage_context = (
    StorageContext.
    from_defaults(vector_store=vector_store))

# =================================================
# STEP 5: Build Vector Index
# =================================================

index = VectorStoreIndex(nodes=text_split,
                         storage_context=
                         storage_context,
                         embed_model=embed_model)

# =================================================
# STEP 6: Convert Index into Question-Answer Engine
# =================================================

query_engine = (index.as_query_engine
                (llm=groq_api_client,
                similarity_top_k=1))

# =================================================
# STEP 7: Ask Query
# =================================================

user_query = str(input("Enter your Query: "))
res = query_engine.query(user_query)

print("Response for RAG Pipeline ->")
print(res)
