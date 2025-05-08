from agents import Agent, Runner, RunConfig
# No longer need add_trace_processor or specific span data imports here for RagMetrics
# from agents import add_trace_processor
# from agents.tracing.span_data import GenerationSpanData, FunctionSpanData, MessageData, ToolDefinitionData

# Import for RagMetrics integration
from ragmetrics.api import ragmetrics_client, default_callback, monitor as ragmetrics_monitor # Import monitor directly
# import json # No longer needed here for the simplified script
# import uuid # No longer needed here

# Initialize and login to RagMetrics
# !!! IMPORTANT: Replace "YOUR_RAGMETRICS_API_KEY" with your actual key !!!
RAGMETRICS_API_KEY = "YOUR_RAGMETRICS_API_KEY" 
try:
    if RAGMETRICS_API_KEY == "YOUR_RAGMETRICS_API_KEY" or not RAGMETRICS_API_KEY:
        print("Warning: RagMetrics API key is a placeholder or empty. Logging to RagMetrics will likely fail.")
        print("Please set your RAGMETRICS_API_KEY in test/openai_agent.py or as an environment variable.")
        # ragmetrics_client.login(key=RAGMETRICS_API_KEY, off=True) # Optionally disable if key is placeholder
        # For testing without a key, ensure ragmetrics_client.login can handle off=True if key is None
        ragmetrics_client.login(key=None, off=True) # Turn off logging if no key
        print("RagMetrics logging is OFF.")
    else:
        ragmetrics_client.login(key=RAGMETRICS_API_KEY)
        print("Successfully logged into RagMetrics.")
except ValueError as e:
    print(f"Failed to login to RagMetrics: {e}")
    print("Please ensure your RAGMETRICS_API_KEY is correct and RagMetrics service is accessible.")
    ragmetrics_client.login(key=None, off=True) # Default to off if login fails
    print("RagMetrics logging is OFF due to login failure.")
except Exception as e:
    print(f"An unexpected error occurred during RagMetrics login: {e}")
    ragmetrics_client.login(key=None, off=True) # Default to off on other errors
    print("RagMetrics logging is OFF due to an unexpected error.")


# Initialize an agent for arithmetic
agent = Agent(
    name="MathAgent",
    instructions="Solve arithmetic expressions.",
    # id="math_agent_001" # Example if agent had an ID, RagMetrics will try to pick it up
)

# Monitor the OpenAI Agent Runner class
# This will wrap Runner.run_sync (and potentially Runner.run if implemented in RagMetrics)
# All calls to Runner.run_sync() will now be logged.
ragmetrics_monitor(
    Runner, 
    metadata={"app_version": "1.0", "environment": "testing"}, # Overall metadata for this monitored client
    # callback=my_custom_callback # Optionally, provide a custom callback
)
print("OpenAI Agent Runner is now being monitored by RagMetrics.")


# Run the agent with a query
# This call will now be automatically logged by RagMetrics
print("\nRunning agent normally (should be traced by RagMetrics)...")
result = Runner.run_sync(agent, "Calculate 10 * 2")
print(f"Agent's Final Output: {result.final_output}")

# Example: Run with RunConfig metadata
print("\nRunning agent with RunConfig metadata (should also be traced by RagMetrics)...")
run_config_with_metadata = RunConfig(
    metadata={"user_id": "user_123", "request_source": "test_script"}
)
result_with_config = Runner.run_sync(agent, "Calculate 5 + 7", config=run_config_with_metadata)
print(f"Agent's Final Output (with config): {result_with_config.final_output}")


print("\n--- RagMetrics Integration Summary (via direct Runner monitoring) ---")
print("OpenAI Agent SDK calls to Runner.run_sync() are now automatically logged by RagMetrics.")
print("The previous custom trace processor is no longer needed for this.")
print("Ensure your RAGMETRICS_API_KEY is correctly set for logs to appear in your dashboard.")
print("---------------------------------------------------------------------")
