# RAG_contextual_retrieval

![image](https://github.com/OmVT/AI_RAG/blob/fbdbc6c0e19bea1b0b0e48fccf0dcbd088450558/screenshots/upload.png)

<ul>
  <li>Used SentenceTransformers to generate document embeddings and ChromaDB for
efficient vector-based document retrieval.</li>
  <li>Integrated Google's Gemini API for natural language generation to deliver accurate and
relevant responses.</li>
  <li>Used Streamlit to create GUI for better user experience.</li>
</ul>

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that grounds an LLM's answers in your own documents instead of relying only on what the model memorized during training. Instead of asking the model a question directly, the system first **retrieves** the most relevant pieces of your document, then **augments** the model's prompt with that retrieved text before it **generates** an answer.

This app works in two phases:

1. **Indexing** — When you upload a file, its text is extracted, split into overlapping chunks (~2000 characters each), and converted into vector embeddings (numerical representations of meaning) using a local Sentence-Transformers model. These embeddings are stored in a ChromaDB vector database.
2. **Retrieval + Generation** — When you ask a question, the question itself is embedded and compared against the stored chunk embeddings using cosine similarity, pulling back the most semantically relevant chunks. Those chunks are inserted into a prompt along with your question and sent to Google's Gemini API, which generates the final answer grounded in your document.

This approach lets the model answer questions about content it has never seen before (private notes, PDFs, books) and reduces hallucination, since answers are based on retrieved text rather than the model's memory alone.

## How to Run

### Prerequisites

- Python 3.9+
- A free [Gemini API key](https://aistudio.google.com/apikey) from Google AI Studio

### 1. Set your Gemini API key

The app reads the key from an environment variable named `gemini_api_key`.

**Windows (PowerShell):**
```powershell
setx gemini_api_key "your-api-key-here"
```
Restart your terminal after running this so the variable takes effect.

**macOS/Linux:**
```bash
export gemini_api_key="your-api-key-here"
```

### 2. Install dependencies

```bash
pip install streamlit PyPDF2 chromadb google-generativeai langchain-text-splitters
```

### 3. Run the app

From the project directory:

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

### 4. Use it

1. Upload a `.txt` or `.pdf` file.
2. Wait for it to be chunked and indexed into ChromaDB (a `chroma_db` folder will be created in the project directory to persist the data).
3. Type a question about the document and click **Get Answers**.
