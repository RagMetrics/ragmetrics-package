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
ragmetrics.login(off=True)

def say_hi(input, cohort = None):
    return "Hi " + input

# Commenting out module-level object creation/saving for test collection
"""
e1 = Example(question="Alice", ground_truth_answer="Hi Alice")
e2 = Example(question="Bob", ground_truth_answer="Hi Bob")
dataset1 = Dataset(examples = [e1, e2], name="Names")
# dataset1.save() # Requires login
task1 = Task(name="Greet", function=say_hi)
# task1.save() # Requires login
criteria1 = Criteria(name = "Accuracy")
# criteria1.save() # Requires login

exp1 = Experiment(
            name="Naming Experiment",
            dataset=dataset1,
            task=task1,
            criteria=[criteria1],                
            judge_model="gpt-4o-mini"
        )
# status = exp1.run()

# assert status.get("state") == "SUCCESS", \
#     f"Expected state 'SUCCESS', got: {status.get('state')}"
"""

# TODO: Add actual tests using pytest fixtures