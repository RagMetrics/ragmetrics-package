import os
import time
from dotenv import load_dotenv

import ragmetrics
from ragmetrics import Task, Example, Dataset, Experiment, Criteria

# Load environment variables
load_dotenv(".env")

# Attempt login using environment variables - Changed to off=True for collection
# api_key = os.getenv("RAGMETRICS_API_KEY")
# base_url = os.getenv("RAGMETRICS_BASE_URL")
ragmetrics.login(off=True)

def local_function(input, cohort = None):
    # Simulate RAG retrieval + generation
    context = f"Context about {input}"
    generation = f"Reflect input: {input}"
    return {
        "contexts": [context], 
        "output": generation 
    }

# Commenting out module-level object creation/saving for test collection
"""
e1 = Example(question="Alice", ground_truth_answer="Reflect input: Alice")
e2 = Example(question="Bob", ground_truth_answer="Reflect input: Bob")
dataset1 = Dataset(examples = [e1, e2], name="Names")
# dataset1.save() # Requires login
task1 = Task(name="Reflect", function=local_function)
# task1.save() # Requires login
criteria1 = Criteria(name = "Accuracy")
# criteria1.save() # Requires login
criteria2 = Criteria(name = "Context Relevance")
# criteria2.save() # Requires login

exp1 = Experiment(
            name="Generation and Retrieval",
            dataset=dataset1,
            task=task1,
            criteria=[criteria1, criteria2],                
            judge_model="gpt-4o-mini"
        )
# status = exp1.run() # Requires login

# assert status.get("state") == "SUCCESS", \
#     f"Expected state 'SUCCESS', got: {status.get('state')}"
"""

# TODO: Add actual tests using pytest fixtures