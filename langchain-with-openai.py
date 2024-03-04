# Import Liabraries
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

'''
Setup a environment variable on Linux Server with name OPENAI_API_KEY and save the API key. Make sure to run source command to enable the variable.
# vi .bashrc
export OPENAI_API_KEY="<Input your API Key Here>"  #Add this line at end of file and save file.
# source .bashrc
'''

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
