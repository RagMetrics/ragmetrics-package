from dotenv import load_dotenv
import ragmetrics

# Import LLM libraries as needed
from openai import OpenAI
import litellm
from langchain_groq import ChatGroq  

load_dotenv(".env")
#Login with os.environ['RAGMETRICS_API_KEY']
# or ragmetrics.login(key="your_ragmetrics_api_key")
# Get the key at https://ragmetrics.ai/
ragmetrics.login(base_url="http://localhost:8000")

def print_response(label, response):
    # Try chat-completion style first.
    try:
        content = response.choices[0].message.content
    except AttributeError:
        # If that fails, maybe it's a dict.
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', None)
        except AttributeError:
            content = None
    # As a final fallback, if response is an AIMessage or similar:
    if content is None and hasattr(response, "content"):
        content = response.content
    if content is None:
        content = "No content"
    print(f"{label} Response: {content}")


def create_messages(client_name, country):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"What is the capital of {country}? (tested with {client_name})"}
    ]

# Test OpenAI client (chat-based)
openai_client = OpenAI()
ragmetrics.monitor(openai_client, context={"client": "openai"})
messages = create_messages("OpenAI", "France")
resp = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    metadata={"task": "test", "step": "openai"}
)
print_response("OpenAI", resp)

# Test LiteLLM client (module-level function)
ragmetrics.monitor(litellm, context={"client": "litellm"})
messages = create_messages("LiteLLM", "Germany")
resp = litellm.completion(
    model="gpt-3.5-turbo",
    messages=messages,
    metadata={"task": "test", "step": "litellm"}
)
print_response("LiteLLM", resp)

# Test LangChain-style client
ragmetrics.monitor(ChatGroq, context={"client": "langchain"})
langchain_model = ChatGroq(model="llama3-8b-8192")
messages = create_messages("LangChain", "Italy")
resp = langchain_model.invoke(
    model="llama3-8b-8192",
    messages=messages, 
    metadata={"task": "test", "step": "langchain"}
)
print_response("LangChain", resp)
