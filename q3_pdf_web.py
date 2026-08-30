from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# Load PDF
pdf_loader = PyPDFLoader("assignment.pdf")
pdf_documents = pdf_loader.load()

# Load Wikipedia page
web_loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
)
web_documents = web_loader.load()

# Print number of pages/documents
print("Number of PDF pages loaded:", len(pdf_documents))
print("Number of Wikipedia documents loaded:", len(web_documents))