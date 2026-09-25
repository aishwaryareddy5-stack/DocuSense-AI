"""Simple command-line test for the HuggingFace LLM + an existing FAISS index."""

import os

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
DB_FAISS_PATH = "vectorstore/db_faiss"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing from the environment.")


def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=endpoint)


PROMPT = """
Use only the information in the context to answer the question.
If the answer is not present, say that the PDF does not contain enough information.

Context: {context}
Question: {question}

Answer clearly and directly.
"""

prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("\nRESULT:\n")
print(response["result"])
