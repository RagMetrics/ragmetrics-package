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
    """Test function for function-based cohort"""
    question = example.question
    answer = f"Auto-generated answer for: {question}"
    contexts = [
        {
            "metadata": {
                "source": "Auto-generated source",
                "title": "Auto-generated title",
                "description": "Auto-generated description",
                "language": "en-US"
            },
            "page_content": f"Auto-generated content for: {question}"
        }
    ]
    
    output_json = {
        "generated_answer": answer,
        "contexts": contexts,
    }
    
    return output_json

# Create unique names for this test run
timestamp = int(time.time())
dataset_name = f"OptionalCohorts_Dataset_{timestamp}"
task_function_name = f"OptionalCohorts_TaskFunction_{timestamp}"
task_model_name = f"OptionalCohorts_TaskModel_{timestamp}"
model_name = "gpt-4o-mini"

# Create test examples
e1 = Example(
    question="What is the capital of France?", 
    ground_truth_answer="Paris", 
    ground_truth_context=["Paris is the capital of France"]
)
e2 = Example(
    question="What is the largest planet in our solar system?", 
    ground_truth_answer="Jupiter", 
    ground_truth_context=["Jupiter is the largest planet in our solar system"]
)

# Create dataset
dataset1 = Dataset(examples=[e1, e2], name=dataset_name)

# Create criteria
criteria1 = Criteria(name="Accuracy")

print("\n=== Testing Experiment with Optional Cohorts Parameter ===")

# 1. Create experiment with explicit cohorts
print("\n1. Creating experiment WITH explicit cohorts:")
cohort1 = Cohort(name="Test Cohort", generator_model=model_name)
exp_with_cohorts = Experiment(
    name=f"OptionalCohorts_WithCohorts_{timestamp}",
    dataset=dataset1,
    task=Task(name=task_model_name, generator_model=model_name),
    criteria=[criteria1],
    judge_model=model_name,
    cohorts=[cohort1]  # Explicitly provide cohorts
)

processed_cohorts = exp_with_cohorts._process_cohorts()
print(f"Explicit cohorts: {processed_cohorts}")

# 2. Create experiment with NO cohorts (model-based task)
print("\n2. Creating experiment with NO cohorts (model-based task):")
task_model = Task(
    name=f"{task_model_name}_auto", 
    generator_model=model_name
)
exp_without_cohorts_model = Experiment(
    name=f"OptionalCohorts_NoCohorts_Model_{timestamp}",
    dataset=dataset1,
    task=task_model,
    criteria=[criteria1],
    judge_model=model_name
    # cohorts parameter omitted - should create a default cohort from task.generator_model
)

processed_cohorts = exp_without_cohorts_model._process_cohorts()
print(f"Auto-generated cohorts from model task: {processed_cohorts}")

# 3. Create experiment with NO cohorts (function-based task)
print("\n3. Creating experiment with NO cohorts (function-based task):")
task_function = Task(
    name=f"{task_function_name}_auto", 
    function=test_callable_function
)
exp_without_cohorts_function = Experiment(
    name=f"OptionalCohorts_NoCohorts_Function_{timestamp}",
    dataset=dataset1,
    task=task_function,
    criteria=[criteria1],
    judge_model=model_name
    # cohorts parameter omitted - should create a default cohort from task.function
)

processed_cohorts = exp_without_cohorts_function._process_cohorts()
print(f"Auto-generated cohorts from function task: {processed_cohorts}")

print("\nAll tests completed successfully!") 