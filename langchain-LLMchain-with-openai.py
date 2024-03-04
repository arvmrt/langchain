import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load OpenAI API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat Model
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model='gpt-3.5-turbo'
)

# Create prompt template
template = "Who is the {sequence} President of {country}?"
prompt = PromptTemplate(template=template,input_variables=['sequence','country'])

# Create LLM Chain
chain = LLMChain(llm=llm,prompt=prompt)

# Define values to input variables
input = {'sequence':2 ,'country':'USA'}

# Execute the Chain
response = chain.invoke(input)

# Display Response from OpenAI Chat Model
print(str(response))
