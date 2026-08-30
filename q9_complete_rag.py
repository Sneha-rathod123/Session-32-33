import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load existing Chroma DB
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma-db",
    embedding_function=emb
)

# Retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# RAG Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer only from the context.
If the answer is not found, say "I could not find the answer."

Context:
{context}

Question:
{question}"""
)

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Test 3 questions
questions = [
    "What is the main topic of this document?",
    "What is mentioned in the document about AI?",
    "What is the capital of Australia?"
]

for q in questions:
    docs = retriever.invoke(q)
    context = "\n".join(d.page_content for d in docs)

    response = llm.invoke(
        prompt.invoke({
            "context": context,
            "question": q
        })
    )

    print("\nQuestion:", q)
    print("Answer:", response.content)