from openai import OpenAI 
from dotenv import load_dotenv
import os

load_dotenv()

client=OpenAI( api_key=os.getenv("OPENCODE_API_KEY"),
    base_url="https://opencode.ai/zen/v1")

print("Chatbot started")
print("Type 'exit' to quite.\n")

while True: 
    user_input=input("You: ")

    if user_input.lower()=='exit':
        print("Chatbot closed.")
        break
    response=client.chat.completions.create(
        model='minimax-m2.5-free',
        messages=[
            {"role":"system","content":"You are a helpful assistant"},
            {"role":"user","content":user_input}
        ]
    )

    ai_reply=response.choices[0].message.content
    print("\nBot:",ai_reply)
    print()