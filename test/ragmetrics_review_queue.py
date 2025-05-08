import ragmetrics
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv(".env")

task_name = "halu_eval"

#Set up the RAGMetrics client
class StubLLMClient:
    """A mock LLM client. Takes input and output. Returns the output."""
    def invoke(self, *args, **kwargs):
        return kwargs.get('output')

ragmetrics.login()
stub_client = StubLLMClient()
ragmetrics.monitor(stub_client, metadata={"task": task_name})


#Load the Halu-eval dataset (https://github.com/RUCAIBox/HaluEval)
halu_eval_qa = load_dataset("pminervini/HaluEval", name="qa")
dataset = halu_eval_qa['data']

#Log traces, top 2 only
topX = 2
dataset_topX = list(dataset)[:topX]

for i, example in enumerate(dataset_topX):
    print(f"Logging example {i+1} of {topX}")
    input = f"Question: {example['question']}\n\n"
    input += f"Knowledge: {example['knowledge']}"
    output = example["hallucinated_answer"]
    expected = example["right_answer"]

    resp = stub_client.invoke(
        input=input,
        output=output,
        expected=expected
    )

# Create a review retroactive review queue
from ragmetrics.reviews import ReviewQueue
rq = ReviewQueue(
    name=task_name,
    condition=task_name,
    criteria=["Accuracy"],
    judge_model="o3-mini",
    retroactive=True # Apply to existing traces
)
rq.save()