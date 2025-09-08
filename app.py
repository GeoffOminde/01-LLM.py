# app.py
# PDF-based Question Answering System with .env support

import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------------------------
# Step 0: Load environment variables
# ------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key not found. Add it to your .env file.")

print("✅ OpenAI API Key loaded successfully!")

# ------------------------------------------------
# Step 1: Ensure PDF exists with content
# ------------------------------------------------
pdf_dir = "pdfs"
pdf_path = os.path.join(pdf_dir, "sample.pdf")

if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
    print("📝 Creating sample PDF...")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Sample PDF Document")
    c.drawString(100, 730, "This document explains AI, Machine Learning, and Large Language Models (LLMs).")
    c.drawString(100, 710, "LLMs like GPT help in chatbots, summarization, and information retrieval.")
    c.drawString(100, 690, "They are transforming industries such as healthcare, education, and finance.")
    c.save()
else:
    print("📂 Found existing sample PDF.")

# ------------------------------------------------
# Step 2: Load PDF into LangChain
# ------------------------------------------------
print(f"📂 Loading PDF: {pdf_path}")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ------------------------------------------------
# Step 3: Initialize embeddings + FAISS DB
# ------------------------------------------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------------------------------------
# Step 4: Run a query
# ------------------------------------------------
query = "What does the document say about Large Language Models?"
answer = qa.run(query)

print("\n❓ Query:", query)
print("📖 Answer:", answer)
