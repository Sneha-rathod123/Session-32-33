from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing Chroma database
vector_store = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings)

# Create MMR retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3})

# Ask a question
question = input("Enter your question: ")
# Retrieve relevant documents
docs = retriever.invoke(question)
# Print retrieved documents
print("\nRetrieved Documents:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)