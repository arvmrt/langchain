# Import Liabraries
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load OpenAI API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat Model
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model='gpt-3.5-turbo'
)

# Define your followup question
prompt = "and who was 16th?"

# Chat History for sample Context
messages = [
    HumanMessage(content="Who is the first president of USA?"),
    SystemMessage(content="George Washington"),
    HumanMessage(content=prompt),
]

# Get response from OpenAI Chat Model
response = chat.invoke(messages)

# Display Response from OpenAI Chat Model
print(str(response.content))
