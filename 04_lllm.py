# 04_lllm.py
"""
RAG (Retrieval-Augmented Generation) system using OpenAI + FAISS.
Loads data from 04LLM.xlsx, builds embeddings, and answers questions.
"""

import os
import pandas as pd
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 🔑 Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 📊 Load dataset from Excel
df = pd.read_excel("04LLM.xlsx")

# Convert rows into LangChain Documents
documents = [
    Document(page_content=row["Content"], metadata={"Title": row["Title"], "Tags": row["Tags"], "Source": row["Source"]})
    for _, row in df.iterrows()
]

# Split long texts into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 🔎 Build FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = FAISS.from_documents(docs, embeddings)

# 📌 RAG function
def rag_query(query: str):
    # Search top 2 relevant chunks
    results = db.similarity_search(query, k=2)
    
    # Prepare context
    context = "\n\n".join([f"{r.metadata['Title']}: {r.page_content}" for r in results])
    
    # Ask LLM
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that answers based on retrieved context."},
            {"role": "user", "content": f"Answer the question using the context:\n\nContext:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

# 🚀 Run Example
if __name__ == "__main__":
    question = "What is RAG and how does it improve LLMs?"
    answer = rag_query(question)
    print("Q:", question)
    print("A:", answer)
