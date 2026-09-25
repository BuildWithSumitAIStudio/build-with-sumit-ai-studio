from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from VectorDBStore.create_index import CreateVectorDBIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from setembedding import SetVectorEmbedding
from llama_index.readers.file import PDFReader


class PDFDataIngestion:

    @staticmethod
    def pdf_data_ingestion(index_name: str, pdf_folder_path: str):

        # Create Vector DB Index
        pinecone_index = CreateVectorDBIndex.create_pinecone_index(index_name=index_name)

        # Set Default Embedding to HuggingFace
        SetVectorEmbedding.set_vector_embedding()

        # Read PDF file from Local Directory
        documents = SimpleDirectoryReader(
            pdf_folder_path,
            file_extractor={
                ".pdf": PDFReader()
            }
        ).load_data()

        # Create a LlamaIndex Vector Store object backed by the Pinecone index
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create a StorageContext and configure it to use the Pinecone vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build the VectorStoreIndex from the documents and store their vectors in Pinecone
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        print("Data Successfully Ingested -> "+str(index))