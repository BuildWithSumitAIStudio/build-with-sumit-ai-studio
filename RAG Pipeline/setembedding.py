from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

class SetVectorEmbedding:
    @staticmethod
    def set_vector_embedding():
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = embed_model