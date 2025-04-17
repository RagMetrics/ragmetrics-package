import sys
import os
import time
from dotenv import load_dotenv

import ragmetrics
from ragmetrics import Task, Example, Dataset, Experiment, Cohort, Criteria

# Load environment variables
load_dotenv(".env")

# Get environment variables
api_key = os.environ.get('RAGMETRICS_API_KEY')
base_url = os.environ.get('RAGMETRICS_BASE_URL')

# Login to ragmetrics
ragmetrics.login(key=api_key, base_url=base_url)

# Define test function
def test_callable_function(example):
    """Test function that will be used as a callable"""
    question = example.question
    answer = f"Answer from callable: {question}"
    contexts = [
        {
            "metadata": {
                "source": "Source 1",
                "title": "Title for Source 1",
                "description": "Description for Source 1",
                "language": "en-US"
            },
            "page_content": f"Content for source 1: {question}"
        }
    ]
    
    output_json = {
        "generated_answer": answer,
        "contexts": contexts,
    }
    
    return output_json

# Create unique names for this test run
timestamp = int(time.time())
dataset_name = f"FunctionHandling_Dataset_{timestamp}"
task_callable_name = f"FunctionHandling_TaskCallable_{timestamp}"
task_string_name = f"FunctionHandling_TaskString_{timestamp}"
model_name = "gpt-4o-mini"

# Create test examples
e1 = Example(
    question="What is the capital of New York State?", 
    ground_truth_answer="Albany", 
    ground_truth_context=["Albany is the capital of NYS"]
)
e2 = Example(
    question="What is the capital of Denmark?", 
    ground_truth_answer="Copenhagen", 
    ground_truth_context=["Copenhagen is the capital of Denmark"]
)

# Create dataset
dataset1 = Dataset(examples=[e1, e2], name=dataset_name)

# Create criteria
criteria1 = Criteria(name="Accuracy")

# Test 1: Create task with callable function
print("\n=== Testing Task with callable function ===")
task_callable = Task(
    name=task_callable_name,
    function=test_callable_function
)

# Verify that the function was properly set
print(f"Task function type: {type(task_callable.function)}")
print(f"Task function name: {task_callable.function.__name__ if callable(task_callable.function) else task_callable.function}")

# Test 2: Create task with function string (module.function format)
print("\n=== Testing Task with function string ===")
try:
    # Note: This will fail to import (the module doesn't exist), but should handle the error gracefully
    task_string = Task(
        name=task_string_name,
        function="test_module.test_function"
    )
    print("Successfully created task with string function")
except Exception as e:
    print(f"Error creating task with string function: {str(e)}")

# Test 3: Create cohort with callable function
print("\n=== Testing Cohort with callable function ===")
cohort_callable = Cohort(
    name="Callable Cohort", 
    function=test_callable_function
)

print(f"Cohort function type: {type(cohort_callable.function)}")
print(f"Cohort function name: {cohort_callable.function_name}")
print(f"Cohort to_dict(): {cohort_callable.to_dict()}")

# Test 4: Create cohort with function string
print("\n=== Testing Cohort with function string ===")
cohort_string = Cohort(
    name="String Cohort", 
    function="test_function_name"
)

print(f"Cohort function_name: {cohort_string.function_name}")
print(f"Cohort to_dict(): {cohort_string.to_dict()}")

# Test 5: Create an experiment with default cohort from callable task function
print("\n=== Testing Experiment with default cohort from callable task function ===")
exp_callable = Experiment(
    name=f"FunctionHandling_Exp_Callable_{timestamp}",
    dataset=dataset1,
    task=task_callable,
    criteria=[criteria1],
    judge_model=model_name
)

# Get the cohorts after processing to see what was created
processed_cohorts = exp_callable._process_cohorts()
print(f"Created cohorts for callable function task: {processed_cohorts}")

print("\nAll tests completed!") 