"""
``ragmetrics-client`` is a python package that offers a comprehensive toolkit for developing and monitoring LLM applications:

* Monitoring LLM applications in production
* Evaluating LLM responses with custom criteria
* Running experiments to compare different models or RAG implementations
* Managing datasets for systematic evaluation
* Creating and executing review queues for human-in-the-loop evaluation

Main Components
==============

* **login**: Authenticate with the RagMetrics API
* **monitor**: Wrap LLM clients to automatically log interactions
* **trace_function_call**: Decorator to trace function execution for tracking retrieval in RAG pipelines
* **import_function**: Utility to import functions from string paths for execution

Core API Functions:
* **Cohort**: Run controlled experiments to group and compare different LLM or RAG implementations
* **Criteria**: Define custom evaluation criteria for assessing model responses
* **Dataset**: Classes for managing evaluation datasets with questions, contexts, and ground truth answers
* **Example**: Define individual test cases with questions, contexts, and ground truth answers
* **Experiment**: Run controlled experiments to compare different LLM or RAG implementations
* **ReviewQueue**: Manage human reviews of LLM interactions with configurable workflows
* **Task**: Define evaluation tasks with specific criteria and parameters
* **Trace**: Access and manipulate logged interactions, including inputs, outputs, and metadata
"""

from .api import (
    login, 
    monitor,
    RagMetricsError,
    RagMetricsConfigError,
    RagMetricsAuthError,
    RagMetricsAPIError
)
from .client import RagMetricsClient, ragmetrics_client
from .base_object import RagMetricsObject
from .decorators import trace_function_call
from .dataset import Dataset, Example
from .tasks import Task
from .utils import default_input, default_output, default_callback
from .experiments import Experiment, Cohort
from .criteria import Criteria
from .reviews import ReviewQueue
from .trace import Trace
from .client import ragmetrics_client, login, monitor
from .default_callback import default_callback, default_input, default_output
from .openai_chat_wrapper import patch_openai_client

__all__ = [
    "RagMetricsClient",
    "ragmetrics_client",
    "login",
    "monitor",
    "trace_function_call",
    "RagMetricsObject",
    "Task",
    "Example",
    "Dataset",
    "default_input",
    "default_output",
    "default_callback",
    "Experiment",
    "Cohort",
    "Criteria",
    "ReviewQueue",
    "Trace",
    "RagMetricsError",
    "RagMetricsConfigError",
    "RagMetricsAuthError",
    "RagMetricsAPIError"
]

# Create a trace_function_call decorator
def trace_function_call(func):
    """
    Decorator to trace function execution and log structured input/output.
    
    Wrap a function with this decorator to automatically log its execution
    details to RagMetrics, including inputs, outputs, and timing information.
    This is particularly useful for tracking retrieval functions in RAG applications.

    Args:
        func: The function to be traced.

    Returns:
        Callable: A wrapped version of the function that logs execution details.
    """
    def wrapper(*args, **kwargs):
        import time
        import uuid
        
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Format parameters in the requested format
        params = []
        # Add positional arguments
        for i, arg in enumerate(args):
            # Try to get the parameter name from the function signature
            try:
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if i < len(param_names):
                    params.append(f"{param_names[i]}={repr(arg)}")
                else:
                    params.append(f"{repr(arg)}")
            except:
                params.append(f"{repr(arg)}")
        
        # Add keyword arguments
        for k, v in kwargs.items():
            params.append(f"{k}={repr(v)}")
        
        # Create the formatted function call string
        formatted_call = f"={func.__name__}({', '.join(params)})"

        # Prepare structured input format
        function_input = [
            {
                "role": "user",
                "content": formatted_call,
                "tool_call": True
            }
        ]

        function_output = {
            "result": result
        }

        # Log the function execution
        ragmetrics_client._log_trace(
            input_messages=function_input,
            response=function_output,
            metadata_llm=None,
            contexts=None,
            duration=duration,
            tools=None,  
            callback_result={
                "input": formatted_call,
                "output": result
            }
        )
        return result

    return wrapper

# Version information
__version__ = '0.2.0'