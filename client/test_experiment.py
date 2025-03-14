#!pip install ragmetrics-client

import ragmetrics
from dotenv import load_dotenv
load_dotenv(".env")

# os.environ['RAGMETRICS_API_KEY'] = 'your_ragmetrics_key'
# Login with the API key from environment
ragmetrics.login()

from ragmetrics import Cohort, Experiment, Task, Dataset, Example

task1 = Task(
    name="Test Task API",
    generator_model="gpt-4o-mini",
    system_prompt="Answer in English."
)

e1 = Example(
    question="What is the biggest city in the US?",
    ground_truth_context=["NYC is the biggest city in the US."],
    ground_truth_answer="NYC"
)
e2 = Example(
    question="Is it beautiful?",
    ground_truth_context=["NYC is known for its beauty."],
    ground_truth_answer="Yes"
)
dataset1 = Dataset(examples = [e1, e2], name="API Dataset")

cohort1 = Cohort(name="gpt-4o-mini", generator_model="gpt-4o-mini")
cohort2 = Cohort(name="API Demo: Stub", rag_pipeline="API Demo: Stub")

exp_models = Experiment(
    name="Model Experiment",
    dataset=dataset1,   
    task=task1,         
    cohorts=[cohort1, cohort2],
    criteria=["Accuracy", "QA_Context mentions a phone"],
    judge_model="gpt-4o-mini"
)

final_progress_data = exp_models.run()
