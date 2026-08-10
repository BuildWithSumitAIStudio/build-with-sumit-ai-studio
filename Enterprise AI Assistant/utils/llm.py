from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

class EnterpriseAssistant:

    def __init__(self):
        self.messages = []

        self.system_prompt = self._load_system_prompt()

        self.messages.append(
            {
                "role": "system",
                "content": self.system_prompt
            }
        )


    @staticmethod
    def _load_system_prompt():
        with open(
                "/Users/lordvoldemort/PycharmProjects/EnterpriseAIAssistant/system_prompts/prompt.txt",
                "r",
                encoding="utf-8"
        ) as file:
            return file.read()

    def ask(self, user_message: str) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.messages
        )

        answer = str(response.choices[0].message.content)

        self.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return answer

    


    
