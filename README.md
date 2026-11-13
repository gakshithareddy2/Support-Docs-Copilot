# Support Docs Copilot
### AI-Powered Documentation Assistant using RAG

## About the Project

Support Docs Copilot is an AI-powered application that helps users find information from support documents without manually reading lengthy files.

It uses **Retrieval-Augmented Generation (RAG)** to search uploaded documents, retrieve relevant information, and generate answers using Google Gemini.

Users can upload PDF or TXT documents and ask questions in natural language. The application provides answers based on the available document content, along with source references.

## Key Features

- **Document Upload:** Upload PDF and TXT support documents.
- **Document Processing:** Extract text and divide it into smaller chunks.
- **Semantic Search:** Retrieve relevant information based on user questions.
- **AI-Powered Answers:** Generate responses using Google Gemini.
- **Vector Storage:** Store document embeddings using ChromaDB.
- **Source References:** Identify the documents used to generate answers.
- **Grounded Responses:** Answer using retrieved document content instead of inventing information.
- **Interactive Interface:** Simple chat interface built with Streamlit.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| Google Gemini | AI-powered answer generation |
| Gemini Embeddings | Text embedding generation |
| ChromaDB | Vector database |
| LangChain | RAG pipeline and document processing |

## How It Works

1. **Upload Documents** – Users upload support documents in PDF or TXT format.
2. **Document Loading** – Extract text from uploaded documents.
3. **Text Chunking** – Divide the text into smaller, manageable sections.
4. **Embedding Generation** – Convert text chunks into vector embeddings.
5. **Vector Storage** – Store embeddings in ChromaDB.
6. **Question Processing** – Convert the user's question into an embedding.
7. **Information Retrieval** – Search for the most relevant document chunks.
8. **Answer Generation** – Send retrieved information to Gemini to generate an answer.
9. **Source Display** – Display the answer with its supporting document references.

## Project Structure

```text
Support-Docs-Copilot/
│
├── documents/
│   ├── database_errors.txt
│   ├── installation.txt
│   └── password_reset.txt
│
├── app.py
├── ingestion_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Application Preview

Support Docs Copilot provides an interactive interface where users can upload documents, select specific files, ask questions, and receive AI-generated answers grounded in the selected documents.

<img width="1848" height="810" alt="Screenshot (3644)" src="https://github.com/user-attachments/assets/c85b06a3-c9f6-47c0-bfcd-404f333b4ad1" />



### Key Interface Features
- **Document Upload:** Upload PDF and TXT files.
- **Document Selection:** Choose which documents to use for each session.
- **Semantic Retrieval:** Retrieve relevant information from selected documents using ChromaDB.
- **AI-Powered Answers:** Generate context-aware responses using Google Gemini.
- **Source References:** View the document sources supporting each answer.
## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gakshithareddy2/Support-Docs-Copilot.git
cd Support-Docs-Copilot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Example Questions

- How do I install the application?
- Why am I getting a database connection error?
- How can I reset my forgotten password?
- What should I check if the database server is not running?

## Future Enhancements

- Conversation memory for follow-up questions.
- More detailed source citations and page references.
- Support for additional document formats.
- Advanced retrieval and reranking techniques.
- Retrieval and answer quality evaluation.

## Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG).
- Working with document loading and text chunking.
- Understanding vector embeddings and semantic search.
- Integrating Large Language Models with external knowledge sources.
- Building an interactive AI application using Python.


---

*Built to make support documentation easier to search, understand, and use.*
