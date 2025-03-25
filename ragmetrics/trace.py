from .api import RagMetricsObject

class Trace(RagMetricsObject):
    object_type = "trace"  

    def __init__(self, id=None, created_at=None, input=None, output=None, raw_input=None, raw_output=None, contexts=None, metadata=None):
        self.id = id
        self.created_at = created_at
        self.input = input
        self.output = output
        self.raw_input = raw_input
        self.raw_output = raw_output
        self.contexts = contexts
        self.metadata = metadata

    def to_dict(self):
        """Convert the Trace object to a dict for API payload or serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "input": self.input,
            "output": self.output,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "contexts": self.contexts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate a Trace object from a dictionary."""
        return cls(
            id=data.get("id"),
            created_at=data.get("created_at"),
            input=data.get("input"),
            output=data.get("output"),
            raw_input=data.get("raw_input"),
            raw_output=data.get("raw_output"),
            contexts=data.get("contexts"),
            metadata=data.get("metadata")
        )