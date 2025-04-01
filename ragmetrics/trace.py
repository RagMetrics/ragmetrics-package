from .api import RagMetricsObject

class Trace(RagMetricsObject):
    """
    Represents a logged interaction between an application and an LLM.
    
    A Trace captures the complete details of an LLM interaction, including
    raw inputs and outputs, processed data, metadata, and contextual information.
    Traces can be retrieved, modified, and saved back to the RagMetrics platform.
    """
    
    object_type = "trace"  

    def __init__(self, id=None, created_at=None, input=None, output=None, raw_input=None, raw_output=None, contexts=None, metadata=None):
        """
        Initialize a new Trace instance.

    
    Args:
            id (str, optional): Unique identifier of the trace.
            created_at (str, optional): Timestamp when the trace was created.
            input (str, optional): The processed/formatted input to the LLM.
            output (str, optional): The processed/formatted output from the LLM.
            raw_input (dict, optional): The raw input data sent to the LLM.
            raw_output (dict, optional): The raw output data received from the LLM.
            contexts (list, optional): List of context information provided during the interaction.
            metadata (dict, optional): Additional metadata about the interaction.
        """
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
        """
        Override attribute setting to enable edit mode when modifying an existing trace.
        
        This automatically sets edit_mode to True when any attribute (except edit_mode itself)
        is changed on a trace with an existing ID.
        
    
    Args:
            key (str): The attribute name.
            value: The value to set.
        """
        # Automatically enable edit mode when any attribute except 'edit_mode' is changed,
        # and if id is already set (i.e. an existing trace is being modified).
        if key not in {"edit_mode"} and hasattr(self, "id") and self.id is not None:
            object.__setattr__(self, "edit_mode", True)
        object.__setattr__(self, key, value)

    def to_dict(self):
        """
        Convert the Trace object to a dictionary for API communication.
        
    
    Returns:
            dict: A dictionary representation of the trace, with edit_mode flag
                 to indicate whether this is an update to an existing trace.
        """
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
        """
        Create a Trace instance from a dictionary.
        
    
    Args:
            data (dict): Dictionary containing trace information.
            
    
    Returns:
            Trace: A new Trace instance initialized with the provided data.
        """
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
