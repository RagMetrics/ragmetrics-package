from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(".env")



llm = ChatOpenAI()
response = llm.invoke("Hello, world!")
print(response)