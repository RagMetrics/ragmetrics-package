import ragmetrics
from openai import OpenAI
import litellm

from dotenv import load_dotenv
import os
load_dotenv(".env")

ragmetrics.login(os.environ["RAGMETRICS_KEY"])

# #Use for OpenAI
openai_client = OpenAI()
ragmetrics.monitor(openai_client)
response = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print("OpenAI Response:", response.choices[0].message.content)


#Use for LiteLLM
ragmetrics.monitor(litellm)
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Germany?"}
    ]
)
print("LiteLLM Response:", response['choices'][0]['message']['content'])
