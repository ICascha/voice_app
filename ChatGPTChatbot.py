from Chatbot import Chatbot
import openai


class ChatGPTBot(Chatbot):
    def __init__(self, open_ai_key):
        global openai
        openai.api_key = open_ai_key
        self.messages = [
            {
                "role": "user",
                "content": "You are Assistant, an American language model. The first message after this will be from me, John.",
            },
        ]

    def chat(self, text) -> str:
        if text:
            self.messages.append({"role": "user", "content": text})
            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
            reply = chat.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        return ""