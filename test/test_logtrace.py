#!pip install ragmetrics-client
#!pip install openai litellm langchain_groq

import ragmetrics
from openai import OpenAI
import litellm
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv(".env")

# os.environ['RAGMETRICS_API_KEY'] = 'your_ragmetrics_key'
# os.environ['GROQ_API_KEY'] = 'your_groq_key'
# os.environ['OPENAI_API_KEY'] = 'your_openai_key'

# Login with the API key from environment
ragmetrics.login()

def create_messages(client_name, country):
    return [
        {"role": "system", "content": f"You are a helpful assistant based on {client_name}."},
        {"role": "user", "content": f"What is the capital of {country}?"}
    ]

# Define a callback that takes raw input and output and returns processed fields.
def my_callback(raw_input, raw_output):
    # Your custom post-processing logic here. For example:
    processed = {
         "input": raw_input,
         "output": raw_output
    }
    return processed

# Test OpenAI client (chat-based)
openai_client = OpenAI()
ragmetrics.monitor(openai_client, metadata={"client": "openai"})
messages = create_messages("OpenAI", "France")
resp = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    metadata={"client": "OpenAI Native", "step": 1},
    contexts=["sample context 1", "sample context 2"]
)
print(resp)

# Test LiteLLM client (module-level function)
ragmetrics.monitor(litellm, metadata={"client": "litellm"})
messages = create_messages("LiteLLM", "Germany")
resp = litellm.completion(
    model="gpt-3.5-turbo",
    messages=messages,
    metadata={"task": "test", "step": "litellm"}
)
print(resp)

# Test LangChain-style client
ragmetrics.monitor(ChatGroq, metadata={"client": "langchain"}, callback=my_callback)
langchain_model = ChatGroq(model="llama3-8b-8192")
messages = create_messages("LangChain", "Italy")
resp = langchain_model.invoke(
    model="llama3-8b-8192",
    input=messages,
    metadata={"task": "test", "step": "langchain"}
)
print(resp)
