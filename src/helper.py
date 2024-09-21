# src/helper.py

import os
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

load_dotenv()
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_documents(file_path):
    """
    Function to load and process a PDF file using LangChain's PyPDFLoader.
    This will also validate the file path to ensure it's a PDF.
    """
    # Step 1: Check if the file exists (file_path should be a string)
    if not isinstance(file_path, str):
        raise TypeError(f"Expected a string for file path, but got {type(file_path)} instead.")
    
    if not os.path.exists(file_path):
        raise ValueError(f"File path {file_path} does not exist.")
    
    # Step 2: Ensure the file is a PDF
    if not file_path.endswith(".pdf"):
        raise ValueError(f"The file {file_path} is not a valid PDF document.")
    
    # Step 3: Load the PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(file_path)
    document_pages = loader.load()

    # Step 4: Combine all pages into a single text
    document_text = ""
    for page in document_pages:
        document_text += page.page_content  # Collect content from each page
    
    # Step 5: Create Document objects from the text
    documents = [Document(page_content=document_text)]
    
    print(f"Successfully loaded {len(document_pages)} pages from {file_path}.")
    
    return documents


def create_faiss_index(documents):
    # Split documents into chunks for better embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Initialize embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use GeminiEmbeddings if applicable
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def get_retrieval_chain(vectorstore, model_name="gemini-1.5-pro"):
    
    llm = ChatGoogleGenerativeAI(
    model = 'gemini-1.5-pro',
    temperature = 0.3
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain
