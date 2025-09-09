### 📄 PDF-based Question Answering System with LangChain

A Python-based project that allows you to query PDF documents using retrieval-augmented generation (RAG) powered by LangChain, FAISS, and OpenAI GPT models.

### 🚀 Features

✅ Automatically creates a sample PDF if none exists.

📂 Loads and parses PDF documents.

🔍 Splits documents into chunks for better retrieval.

🧠 Generates embeddings using OpenAI for semantic search.

⚡ Uses FAISS for fast vector search.

🤖 Supports RAG-based question answering with LangChain LLM.

🔄 Can query the document for relevant information.

🔧 Optional .env support for storing your OpenAI API key securely.

### 🛠️ Prerequisites

Python 3.10+

pip (Python package manager)

An OpenAI API Key with sufficient quota.

### 📦 Installation

Clone the repository

```
git clone <your-repo-url>
cd 01-LLM.py
```

## Create a virtual environment (optional but recommended)
```
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

### 3.Install dependencies
```
pip install -r requirements.txt

```
Example requirements.txt includes:
```
langchain
langchain-openai
langchain-community
reportlab
python-dotenv
faiss-cpu
pandas
openpyxl
```

### 5.Set your OpenAI API Key

Create a .env file in the project root:
```
OPENAI_API_KEY=sk-<your_actual_openai_key_here>
```
📝 Usage

### 1.Run the main app
```
python app.py

```
### 2.Expected output

✅ OpenAI API Key loaded successfully!
📂 Found existing sample PDF.
📂 Loading PDF: pdfs/sample.pdf

❓ Query: What does the document say about Large Language Models?
📖 Answer: [GPT-generated answer based on PDF content]


### 3.Query custom questions

Edit app.py:
```
query = "Explain how LLMs are used in education."
answer = qa.run(query)
print(answer)
```
### ⚙️ Code Structure
```
01-LLM.py/
│
├─ app.py                  # Main Python script
├─ pdfs/
│   └─ sample.pdf          # Auto-generated sample PDF
├─ .env                    # Store OPENAI_API_KEY
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation
```
### 🧠 How It Works

PDF Creation / Loading

Checks if sample.pdf exists.

Creates a default PDF if missing.

Document Parsing & Chunking

Loads the PDF using PyPDFLoader.

Splits content into chunks using RecursiveCharacterTextSplitter.

Embeddings & FAISS Indexing

Generates semantic embeddings for each chunk via OpenAI.

Builds a FAISS vector database for fast similarity search.

RAG-based Querying

Retrieves the top-k relevant chunks from FAISS.

Feeds the retrieved context to a GPT model for answers.

Optional Quota Fallback

If OpenAI quota is exceeded, can return mock answers.

### ⚡ Notes & Tips

OpenAI API Quota: Ensure your API key has sufficient quota to generate embeddings and run LLM queries.

Custom PDFs: You can replace sample.pdf with your own documents in pdfs/.

Chunk Size: Adjust chunk_size and chunk_overlap for better context coverage.

LangChain LLM Models: You can switch to gpt-4o-mini, gpt-3.5-turbo, or any other model supported by your API plan.

### 🛡️ Security

Never commit your OpenAI API key to GitHub.

Use .env to store credentials securely.

For deployment, consider environment variables instead of .env.

### 📚 References

LangChain Documentation

OpenAI API Documentation

FAISS Vector Search

ReportLab PDF Generation

### 👩‍💻 Author
```
Your Name
Email: youremail@example.com

GitHub: github.com/yourusername
```