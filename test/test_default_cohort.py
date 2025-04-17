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

def test_function(example):
    """Test function for function-based cohort"""
    question = example.question
    answer = f"Answer to the question: {question}"
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
dataset_name = f"DefaultCohort_Dataset_{timestamp}"
task_function_name = f"DefaultCohort_TaskFunction_{timestamp}"
task_model_name = f"DefaultCohort_TaskModel_{timestamp}"
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

# Test 1: Create experiment with task that has a function
print("\n=== Testing default cohort creation with function-based task ===")
task_function = Task(
    name=task_function_name,
    function=test_function
)

exp_function = Experiment(
    name=f"DefaultCohort_Function_Exp_{timestamp}",
    dataset=dataset1,
    task=task_function,
    criteria=[criteria1],
    judge_model=model_name
)

# Access the cohorts after processing to see what was created
processed_cohorts = exp_function._process_cohorts()
print(f"Created cohorts for function-based task: {processed_cohorts}")

# Test 2: Create experiment with task that has a generator_model
print("\n=== Testing default cohort creation with model-based task ===")
task_model = Task(
    name=task_model_name,
    generator_model=model_name,
    system_prompt="Answer questions accurately."
)

exp_model = Experiment(
    name=f"DefaultCohort_Model_Exp_{timestamp}",
    dataset=dataset1,
    task=task_model,
    criteria=[criteria1],
    judge_model=model_name
)

# Access the cohorts after processing to see what was created
processed_cohorts = exp_model._process_cohorts()
print(f"Created cohorts for model-based task: {processed_cohorts}")

print("\nTests completed successfully!") 