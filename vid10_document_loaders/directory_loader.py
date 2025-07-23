#  didn't made a books folder so don't run it

from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path = 'books',
    glob= "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.load()

print(docs[0].metadata)

