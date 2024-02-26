# open ai
# ---
# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import PyMuPDFLoader, PDFMinerLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.storage import LocalFileStore
# from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
# from dotenv import load_dotenv
# load_dotenv()

# underlying_embeddings = OpenAIEmbeddings(request_timeout=15,
#                                          show_progress_bar=True)
# fs = LocalFileStore("./cache/")

# cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#     underlying_embeddings, fs, namespace=underlying_embeddings.model
# )

# loader = PyMuPDFLoader('../DOCS/IRC/IRC_CODES.pdf')

# documents = loader.load()
# print(len(documents))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=100)

# docs = text_splitter.split_documents(documents)

# db = FAISS.from_documents(docs, cached_embedder)
# db.save_local("TAXGPT_IRC_FAISS")


# query = "How to qualify for section 179?"

# docs = db.similarity_search(query)

# print(docs[0].page_content)

# hugging face 
# ---
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = PyMuPDFLoader("whitepaper.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

db = FAISS.from_documents(docs, embeddings)
db.save_local("test_faiss")

query = "what is safety in pretraining?"
docs = db.similarity_search(query)

print(docs[0].page_content)
