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

def local_function(input, cohort = None):
    answer = f"Reflect input: {input}"
    contexts = [
        {"page_content": f"Source 1: {input}"},
        {"page_content": f"Source 2: {input}"}
    ]    
    output_json = {
        "generated_answer": answer,
        "contexts": contexts,
    }
    
    return output_json

e1 = Example(question="Alice", ground_truth_answer="Reflect input: Alice")
e2 = Example(question="Bob", ground_truth_answer="Reflect input: Bob")
dataset1 = Dataset(examples = [e1, e2], name="Names")
task1 = Task(name="Reflect", function=local_function)
criteria1 = Criteria(name = "Accuracy")
criteria2 = Criteria(name = "Context Relevance")

exp1 = Experiment(
            name="Generation and Retrieval",
            dataset=dataset1,
            task=task1,
            criteria=[criteria1, criteria2],                
            #criteria=[criteria2],
            judge_model="gpt-4o-mini"
        )
status = exp1.run()

assert status.get("state") == "SUCCESS", \
    f"Expected state 'SUCCESS', got: {status.get('state')}"