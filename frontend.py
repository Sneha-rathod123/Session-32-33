import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load .env from this folder
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

key = os.getenv("GROQ_API_KEY")

if not key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# Embeddings
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma DB
db = Chroma(
    persist_directory="company-chroma-db",
    embedding_function=emb
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# RAG Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer only from the context.
If the answer is not found, say "I could not find the answer."

Context: {context}
Question: {question}"""
)

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=key
)

# Frontend
st.title("🤖 Company RAG Chatbot")

question = st.text_input("Ask a question about the company:")

if question:
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)

    answer = llm.invoke(
        prompt.invoke({
            "context": context,
            "question": question
        })
    )

    st.write("### Answer")
    st.write(answer.content)