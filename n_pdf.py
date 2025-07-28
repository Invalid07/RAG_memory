from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import docx  # Library for reading .docx files

# Load environment variables
load_dotenv()

# Load the HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=1.2,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Function to extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to take document input from user
def get_document():
    file_path = input("Enter the full path of your document (.txt or .docx): ").strip()

    # Check if file exists
    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path and try again.")
        return None, None

    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == ".txt":
        loader = TextLoader(file_path)
        doc = loader.load()
    elif file_extension == ".docx":
        text = extract_text_from_docx(file_path)
        doc = [{"page_content": text}]
    else:
        print("Error: Unsupported file format. Please upload a .txt or .docx file.")
        return None, None

    return doc, file_extension

# Get document from user
doc, file_extension = get_document()
if not doc:
    exit()  # Exit if file is not found or unsupported format

# Split the document into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
doc_split = splitter.split_documents(doc)

# Convert the text into embeddings and store in FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(doc_split, embeddings)

# Create retrieval
retrieval = vector_store.as_retriever()

# Get user query
query = input("Enter your query: ").strip()
retrieved_docs = retrieval.get_relevant_documents(query)

# Combine retrieval results into a single prompt
prompt_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Create prompt template
template = PromptTemplate(
    input_variables=["query"],
    template="Write an answer in simple language so everyone can understand easily: or give the dtail what user asked {query}"
)

# Output parser
parser = StrOutputParser()

# Create and run the chain
chain = LLMChain(prompt=template, llm=llm, output_parser=parser)
result = chain.run(prompt_text)

print("\nAI Response:\n")
print(result)
