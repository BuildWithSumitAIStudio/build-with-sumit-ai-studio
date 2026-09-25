from llama_index.core import Settings
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os


load_dotenv()
class SetLLM:

    @staticmethod
    def set_llm():
        llm = Groq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

        Settings.llm = llm