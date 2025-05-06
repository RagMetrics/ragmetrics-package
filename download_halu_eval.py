from datasets import load_dataset
import ragmetrics
import os
import time

# Load the 'qa' split of the HaluEval dataset
# The 'name' parameter corresponds to the 'subset' in Hugging Face
halu_eval_qa = load_dataset("pminervini/HaluEval", name="qa")

# The dataset is now loaded.
# You can inspect its structure, for example by printing it:
print("Dataset structure:")
print(halu_eval_qa)

# Initialize RagMetrics
# Ensure RAGMETRICS_API_KEY environment variable is set
api_key = os.environ.get("RAGMETRICS_API_KEY")
if not api_key:
    print("Error: RAGMETRICS_API_KEY environment variable not set.")
    print("Please set it before running the script.")
    # You might want to exit here if the API key is mandatory for proceeding
    # import sys
    # sys.exit(1)
else:
    try:
        ragmetrics.login(key=api_key)
        print("Successfully logged into RagMetrics.")
    except ValueError as e:
        print(f"Error logging into RagMetrics: {e}")
        # Optionally exit or handle error
        # import sys
        # sys.exit(1)

# According to the Hugging Face page, the 'qa' subset has a 'data' split.
# Let's try to access and print the first example from the 'data' split:
if 'data' in halu_eval_qa:
    print("\nFirst example from the 'data' split:")
    print(halu_eval_qa['data'][0])

    # Check if login was successful before attempting to log traces
    if api_key and ragmetrics.ragmetrics_client.access_token:
        print("\nLooping through the dataset to log traces...")
        for i, example in enumerate(halu_eval_qa['data']):
            question = example['question']
            hallucinated_answer = example['hallucinated_answer']
            right_answer = example['right_answer'] # Available, can be used as 'expected'

            # Prepare data for _log_trace
            # input_messages should be in the format expected by RagMetrics (e.g., OpenAI message format)
            input_messages_for_log = [{"role": "user", "content": question}]
            
            # response should also be in a structured format if possible,
            # mimicking an LLM response.
            # For simplicity, we'll create a basic structure.
            # The actual 'output' for logging will come from callback_result.
            response_for_log = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": hallucinated_answer 
                    }
                }]
            }

            callback_results = {
                "input": question,
                "output": hallucinated_answer,
                "expected": right_answer # Logging the right_answer as 'expected'
            }
            
            start_time = time.time()
            # Simulate some processing time if necessary, or just use a small constant
            duration = 0.1 # Dummy duration in seconds
            
            try:
                ragmetrics.ragmetrics_client._log_trace(
                    input_messages=input_messages_for_log,
                    response=response_for_log, # This is raw response
                    metadata_llm=None, # No specific LLM call metadata here
                    contexts=None, # No external contexts provided to an LLM
                    expected=right_answer, # This is the specific field for expected
                    duration=duration,
                    tools=None,
                    callback_result=callback_results # This provides structured input/output
                )
                if (i + 1) % 10 == 0: # Print progress every 10 traces
                    print(f"Logged trace {i+1}...")
            except Exception as e:
                print(f"Error logging trace for example {i+1}: {e}")
        print("Finished logging traces.")
    else:
        print("\nSkipping trace logging as RagMetrics login was not successful or API key not provided.")

else:
    print("\nThe 'data' split was not found. Available splits are:")
    print(halu_eval_qa.keys())

# If you want to save it locally, you can iterate through the splits and save them.
# For example, to save the 'data' split to a CSV file:
# if 'data' in halu_eval_qa:
#     halu_eval_qa['data'].to_csv("halu_eval_qa_data.csv", index=False)
#     print("\nSaved 'data' split to halu_eval_qa_data.csv") 