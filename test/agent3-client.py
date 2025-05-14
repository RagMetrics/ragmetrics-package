import os
import ragmetrics
from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, function_tool
# Import monitor_agents directly from ragmetrics
from ragmetrics import monitor_agents

#load .env
from dotenv import load_dotenv
load_dotenv('.env')

# 1. Configure RagMetrics and monitor the OpenAI client
ragmetrics.login()

# Create both sync and async clients - the agents SDK needs the async one
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)
async_openai_client = AsyncOpenAI(api_key=openai_api_key)

# Monitor the sync client for regular API calls
ragmetrics.monitor(openai_client)

# Use the monitor_agents function with the async client
monitor_agents(async_openai_client)

# 2. Define your tools and agents as usual
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

agent = Agent(
    name="WeatherAgent",
    instructions="Answer weather queries",
    tools=[get_weather]
)

# 3. Run the agent; RagMetrics will capture the trace
result = Runner.run_sync(agent, "What's the weather in Berlin?")
print(result.final_output)
