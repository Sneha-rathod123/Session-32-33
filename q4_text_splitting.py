from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("assignment.pdf")
documents = loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# Print total chunks
print("Total number of chunks created:", len(chunks))

# Display first 3 chunks
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk.page_content)