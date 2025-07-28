# RAG Agent with Google Gemini, PDF/DOCX/TXT Support, Memory, and Recursive Splitter

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 1: Load document (PDF, DOCX, TXT)
def load_documents_from_file(file_path):
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext == "docx":
        return Docx2txtLoader(file_path).load()
    elif ext == "txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Step 2: Split document into chunks
def prepare_documents(file_path):
    documents = load_documents_from_file(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Step 3: Create FAISS vectorstore with Gemini embeddings
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embeddings)

# Step 4: Create RAG Chain with conversational memory
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    model = ChatGoogleGenerativeAI(
        model="gemini‑1.5‑flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

# Step 5: Run the chatbot
if __name__ == "__main__":
    file_path = input("Enter path to your document (.pdf, .docx, .txt): ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        exit()

    documents = prepare_documents(file_path)
    vectorstore = create_vectorstore(documents)
    rag_chain = create_rag_chain(vectorstore)

    print("\n🤖 Chatbot is ready! Type 'exit' to stop.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        response = rag_chain.invoke({"question": query})
        print("\nGemini:", response["answer"])
