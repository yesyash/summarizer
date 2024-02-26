from langchain.embeddings.openai import OpenAIEmbeddings
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

# loader = PyMuPDFLoader("whitepaper.pdf")
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)

db = FAISS.load_local("test_faiss", embeddings)
# db.save_local("test_faiss")

query = "what is safety in pretraining?"
# query = "what is red teaming?"
docs = db.similarity_search(query)

for doc in docs:
    print(doc.page_content)

# print(docs[0].page_content)
