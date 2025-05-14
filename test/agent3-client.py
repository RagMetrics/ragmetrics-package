import os
import ragmetrics
from openai import OpenAI
from agents import Agent, Runner, function_tool

#load .env
from dotenv import load_dotenv
load_dotenv('.env')

# 1. Configure RagMetrics and monitor the agents Runner
ragmetrics.login()

# Monitor the Runner class - this will internally configure the agents SDK
ragmetrics.monitor(Runner)

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
