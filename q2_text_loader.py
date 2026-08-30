from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/company_info.txt")

documents = loader.load()

print("Number of documents loaded:", len(documents))

print("\nContent of the first document:")
print(documents[0].page_content)