from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import dotenv
import os
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=os.getenv("OPENCODE_API_KEY"),
    base_url="https://opencode.ai/zen/v1",
    model="minimax-m2.5-free"
)

# Store memory for multiple users
store = {}

# Function to get/create chat history
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create chain with memory
chain = RunnableWithMessageHistory(
    llm,
    get_session_history
)

# --- NEW: Loop to provide information ---
informations = [
    "My favorite color is Blue.",
    "I live in Kolkata.",
    "I am a software engineer.",
    "I love eating Biryani for dinner."
]

session_config = {"configurable": {"session_id": "user1"}}

# print("--- Feeding information to memory ---")
# for info in informations:
#     print(f"Providing: {info}")
#     # We invoke the chain for each piece of info
#     chain.invoke(f"Remember this: {info}", config=session_config)
#     print(f"Store in memory: {store}")

print("\n--- Testing memory ---")

# The test question
test_query = "Based on what I told you, what is my favorite food and where do I live?"

response = chain.invoke(test_query, config=session_config)

print(f"Question: {test_query}")
print(f"AI Answer:\n{response.content}")
