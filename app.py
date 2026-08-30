from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="company-chroma-db",
    embedding_function=emb
)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """Answer only from the context.
If the answer is not found, say "I could not find the answer."

Context: {context}
Question: {question}"""
)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

print("Company RAG Chatbot | Type 'exit' to stop.")

while True:
    q = input("\nYou: ")
    if q.lower() == "exit":
        break

    docs = retriever.invoke(q)
    context = "\n".join(d.page_content for d in docs)

    answer = llm.invoke(
        prompt.invoke({"context": context, "question": q})
    )

    print("Bot:", answer.content)