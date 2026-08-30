from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    ("human", "{question}")
])

chain = prompt | model

print("Basic LLM Chat Application")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = chain.invoke({"question": question})
    print("AI:", response.content)