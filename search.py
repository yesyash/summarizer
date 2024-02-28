import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings

import subprocess

load_dotenv()

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = FAISS.load_local("test_faiss", embeddings)

model = input("enter model name (optional): ")
query = input('Enter your query \n')
filename = input('enter the output file: ')

if filename == '':
    filename = 'search-output.txt'

if model == '':
    model = "llama2"

# do a similartiy search from the databse
docs = db.similarity_search(query)

# write the contents to a file
file = open(filename, 'w+')

for doc in docs:
    file.write(doc.page_content)

file.close()

# Execute olama command
time.sleep(10)
command = f"ollama run {model} \"$(cat {filename})\" \"{query}\""
print(command)

subprocess.run(command, shell=True)
