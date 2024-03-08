import os
from dotenv import load_dotenv
import uuid
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Define embedding Model
embeddings_google = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
embeddings_openai = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
embeddings_model_to_use = embeddings_google         #Select which embedding model to use


persist_directory = "chroma_native_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="my_collection3", embedding_function=embeddings_model_to_use)

input_file = "state_of_union.txt"
raw_documents = TextLoader(input_file, encoding="utf8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
documents = text_splitter.split_documents(raw_documents)

print(collection.count())

for doc in documents:
  collection.add(
    ids=[str(uuid.uuid1())], 
    metadatas=doc.metadata, 
    documents=doc.page_content
  )

results = collection.query(
    query_texts=["What did the president say about Ketanji Brown Jackson"],
    n_results=2
)

print(collection.count())
#print(collection)
print(results['documents'])
