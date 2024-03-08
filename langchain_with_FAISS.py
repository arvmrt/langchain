import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Define embedding Model
embeddings_google = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
embeddings_openai = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)
embeddings_model_to_use = embeddings_google   #Select which embedding model to use

# Split the document
raw_documents = TextLoader('state_of_union.txt', encoding="utf8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
    )
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, embeddings_model_to_use)

query_texts = "What did the president say about Ketanji Brown Jackson"
results = db.similarity_search(query_texts)
print(results[0].page_content)
