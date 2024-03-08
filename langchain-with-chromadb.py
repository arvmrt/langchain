import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Setup embedding Model
embeddings_google = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
embeddings_openai = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)
embeddings_model_to_use = embeddings_google  #Select which embedding model ot use

# Split the document
raw_documents = TextLoader('state_of_union.txt', encoding="utf8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=70,
    length_function=len,
    is_separator_regex=False,
    )
documents = text_splitter.split_documents(raw_documents)

# Setup Chroma DB and Save DB to disk
db = Chroma.from_documents(
    documents, 
    embeddings_model_to_use,
    persist_directory="chroma_db",
    )

db.persist()

# Perform Similarity search
query_texts = "What did the president say about Ketanji Brown Jackson"
print("\n" + query_texts)
results = db.similarity_search(query_texts)
print(results[0].page_content)


# Load DB from disk and perform Similarity search
db2 = Chroma(persist_directory="chroma_db", embedding_function=embeddings_model_to_use)
query_texts2="What did the president say about Ukraine"
print("\n" + query_texts2)
results2 = db2.similarity_search(query_texts)
print(results2[0].page_content)


# Similarity search with score and metadata
query_texts3="What did the president say about Ukraine"
print("\n" + query_texts3)
results3 = db2.similarity_search_with_score(query_texts)
print(results3[0])

db = None
db2 = None
