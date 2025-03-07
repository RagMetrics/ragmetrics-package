from .api import RagMetricsObject

class Example:
    def __init__(self, question, ground_truth_context, ground_truth_answer):
        self.question = question
        self.ground_truth_context = ground_truth_context
        self.ground_truth_answer = ground_truth_answer

    def to_dict(self):
        """Convert the Example instance into a dictionary for API requests."""
        return {
            "question": self.question,
            "ground_truth_context": self.ground_truth_context,
            "ground_truth_answer": self.ground_truth_answer
        }

class Dataset(RagMetricsObject):
    object_type = "dataset"

    def __init__(self,name, examples = [],  source_type="", source_file="", questions_qty=0):
        self.name = name
        self.examples = examples 
        self.source_type = source_type
        self.source_file = source_file
        self.questions_qty = questions_qty
        self.id = None

    def to_dict(self):
        return {
            "datasetName": self.name,
            "datasetSource": "DA",
            "examples": [ex.to_dict() for ex in self.examples],
            "datasetQty": len(self.examples)
        }

    @classmethod
    def from_dict(cls, data: dict):
        examples = [
            Example(**{k: v for k, v in ex.items() if k in ['question', 'ground_truth_context', 'ground_truth_answer']})
            for ex in data.get("examples", [])
        ]
        ds = cls(
            name=data.get("name", ""),
            examples=examples,
            source_type=data.get("source_type", ""),
            source_file=data.get("source_file", ""),
            questions_qty=data.get("questions_qty", 0)
        )
        ds.id = data.get("id")
        return ds
    
    def __iter__(self):
        return iter(self.examples)