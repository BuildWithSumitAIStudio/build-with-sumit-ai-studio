from VectorDBStore.create_index import CreateVectorDBIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from setembedding import SetVectorEmbedding
from setllm import SetLLM

class PDFDataQuery:

    @staticmethod
    def pdf_data_query(index_name: str, query: str):

        # Set Default LLM to Groq
        SetLLM.set_llm()

        # Set Default Embedding to HuggingFace
        SetVectorEmbedding.set_vector_embedding()

        # Return Vector DB Index if it is already created
        pinecone_index = CreateVectorDBIndex.create_pinecone_index(index_name=index_name)

        # Create a LlamaIndex Vector Store object backed by the Pinecone index
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create VectorStoreIndex from data already exists
        loaded_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        # Query Data
        query_engine = loaded_index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(query)

        """retriever = loaded_index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)"""

        return response