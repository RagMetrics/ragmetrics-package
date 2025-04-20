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

def say_hi(input, cohort = None):
    return "Hi " + input

e1 = Example(question="Alice", ground_truth_answer="Hi Alice")
e2 = Example(question="Bob", ground_truth_answer="Hi Bob")
dataset1 = Dataset(examples = [e1, e2], name="Names")
task1 = Task(name="Greet", function=say_hi)
criteria1 = Criteria(name = "Accuracy")

exp1 = Experiment(
            name="Naming Experiment",
            dataset=dataset1,
            task=task1,
            criteria=[criteria1],                
            judge_model="gpt-4o-mini"
        )
status = exp1.run()

assert status.get("state") == "SUCCESS", \
    f"Expected state 'SUCCESS', got: {status.get('state')}"