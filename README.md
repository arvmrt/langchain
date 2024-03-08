# Langchain with OpenAI GPT


## Getting Started

#### Prerequisites
```
- RHEL8/CENTOS8
- Python 3.12.1
```

#### Create a virtual environment
```
mkdir myenv
pip -m venv myenv
source myenv/bin/activate
```

#### Install Packages 
```
pip install langchain==0.1.11
pip install langchain-openai==0.0.8
pip install langchain-google-genai==0.0.9
pip install pillow==10.2.0
pip install python-dotenv==1.0.1
```

#### Create a ".env" file under the project directory and add the following API KEYS:
```
OPENAI_API_KEY = '<Paste your OpenAI API Key>'
GOOGLE_API_KEY = '<Paste your Google Gemini API Key>'
```
