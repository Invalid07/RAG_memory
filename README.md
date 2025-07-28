# RAG_memory
Retrieval-Augmented Generation (RAG) framework with integrated memory support. It combines vector-based document retrieval with a memory module to maintain conversational context across multiple interactions. Built with modular components for embedding, retrieval, and generation

🔍💬 Conversational RAG with Memory using LangChain + Google Gemini
This project integrates Retrieval-Augmented Generation (RAG) with memory to create smart, context-aware chat experiences. It uses:

🧠 ChatGoogleGenerativeAI for powerful LLM responses
🧲 GoogleGenerativeAIEmbeddings + FAISS for fast, semantic vector search
🧩 RecursiveCharacterTextSplitter for clean chunking of large texts
🗂️ ConversationBufferMemory to retain past chat context
🔄 ConversationalRetrievalChain to tie everything together seamlessly

