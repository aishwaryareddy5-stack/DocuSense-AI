# DocuSense AI

DocuSense AI is a general-purpose PDF question-answering assistant built with Streamlit, LangChain, HuggingFace embeddings, FAISS, and a HuggingFace-hosted Llama 3.1 model.

## How it works

1. Upload a PDF in the Streamlit sidebar.
2. The PDF is loaded and split into small overlapping chunks.
3. `sentence-transformers/all-MiniLM-L6-v2` converts chunks into vector embeddings.
4. FAISS stores the vectors and retrieves the most relevant passages for each question.
5. Llama 3.1 8B Instruct generates **Answer from PDF** using the retrieved context.
6. A second LLM call generates **Extension from AI**, clearly separated from the PDF-grounded answer.

## Project structure

```text
RAG/
├── FinalBot.py
├── LLM_memory.py
├── connect_llm.py
├── Pipfile
├── Pipfile.lock
├── .env.example
├── data/                  # local PDFs; ignored by Git
└── vectorstore/           # local FAISS index; ignored by Git
```

> The folder name can be changed later to match the final project name. The application itself is now general-purpose and is not limited to any documents.

## Setup

Create a `.env` file from `.env.example` and add your Hugging Face access token:

```text
HF_TOKEN=your_token_here
```

Install dependencies with Pipenv:

```bash
python -m pipenv install
python -m pipenv shell
streamlit run FinalBot.py
```

Then open the local Streamlit URL shown in the terminal and upload a PDF.

## Security

- Never commit `.env` or API/access tokens.
- For deployment, configure `HF_TOKEN` as a platform secret instead of putting it in source code.

## Notes

The original fixed-PDF FAISS workflow is preserved in `LLM_memory.py` as an optional offline index builder. The main Streamlit application now processes the PDF selected by the user directly.
