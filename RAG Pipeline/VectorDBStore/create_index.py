import os

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

load_dotenv(verbose=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

class CreateVectorDBIndex:

    @staticmethod
    def create_pinecone_index(index_name: str):

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={
                    "environment": "development"
                }
            )
        return pc.Index(index_name)

