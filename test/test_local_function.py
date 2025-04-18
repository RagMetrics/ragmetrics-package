import os
import time
from dotenv import load_dotenv

import ragmetrics
from ragmetrics import Task, Example, Dataset, Experiment, Criteria

# Load environment variables
load_dotenv(".env")

# Get environment variables
api_key = os.environ.get('RAGMETRICS_API_KEY')
base_url = os.environ.get('RAGMETRICS_BASE_URL')

# Login to ragmetrics
ragmetrics.login(key=api_key, base_url=base_url)

def local_llm_pipeline(example, cohort):
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
            },
            {
                "metadata": {
                    "source": "Source 2",
                    "title": "Title for Source 2",
                    "description": "Description for Source 2",
                    "language": "en-US"
                },
                "page_content": f"Content for source 2: {question}"
            }
        ]
    
    output_json = {
        "generated_answer": answer,
        "contexts": contexts,
    }
    
    return output_json

dataset_name = f"QA_Capitals_{int(time.time())}"
task_name = f"QA_Capitals_{int(time.time())}"
experiment_name = f"QA_Local_function_{int(time.time())}"

e1 = Example(
    question="What is the capital of New York State?", 
    ground_truth_answer="Albany", 
    ground_truth_context=[
        "Albany is the capital of NYS",
        "NYS is short for New York State"
    ])
e2 = Example(
    question="What is the capital of Denmark?", 
    ground_truth_answer="Copenhagen", 
    ground_truth_context=[
        "Copenhagen is the capital of Denmark",
        "Denmark is a Scandinavian country"
    ])
dataset1 = Dataset(examples = [e1, e2], name=dataset_name)

task1 = Task(name=task_name, function=local_llm_pipeline)

criteria1 = Criteria(name = "Accuracy")

exp1 = Experiment(
            name=experiment_name,
            dataset=dataset1,
            task=task1,
            criteria=[criteria1],                
            judge_model="gpt-4o-mini"
        )
final_progress_data = exp1.run()

assert final_progress_data.get("state") == "SUCCESS", \
    f"Expected state 'SUCCESS', got: {final_progress_data.get('state')}"