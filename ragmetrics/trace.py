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
        self.edit_mode = False

    def __setattr__(self, key, value):
        # Automatically enable edit mode when any attribute except 'edit_mode' is changed,
        # and if id is already set (i.e. an existing trace is being modified).
        if key not in {"edit_mode"} and hasattr(self, "id") and self.id is not None:
            object.__setattr__(self, "edit_mode", True)
        object.__setattr__(self, key, value)

    def to_dict(self):
        """Convert the Trace object to a dict for API payload or serialization."""
        return {
            "id": self.id if self.edit_mode else None,
            "created_at": self.created_at,
            "input": self.input,
            "output": self.output,
            "raw_input": self.raw_input,
            "raw_output": self.raw_output,
            "contexts": self.contexts,
            "metadata": self.metadata,
            "edit": self.edit_mode,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Instantiate a Trace object from a dictionary."""
        trace = cls(
            id=data.get("id"),
            created_at=data.get("created_at"),
            input=data.get("input"),
            output=data.get("output"),
            raw_input=data.get("raw_input"),
            raw_output=data.get("raw_output"),
            contexts=data.get("contexts"),
            metadata=data.get("metadata")
        )
        trace.edit_mode = False
        return trace
