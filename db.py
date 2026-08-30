from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load company information
docs = TextLoader("data/company_info.txt").load()

# Split into chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
).split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma database
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="company-chroma-db"
)

print("Company RAG database created successfully!")
print("Total chunks:", len(chunks))