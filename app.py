# app.py
# PDF-based Question Answering System

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

# Load PDF (replace 'sample.pdf' with your file)
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Build embeddings + FAISS store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

retriever = db.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What is the document about?"
answer = qa.run(query)

print("❓ Query:", query)
print("📖 Answer:", answer)
