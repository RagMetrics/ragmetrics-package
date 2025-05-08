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

from .api import RagMetricsClient, ragmetrics_client, login, monitor
from .decorators import trace_function_call
from .models import RagMetricsObject
from .dataset import Dataset, Example
from .tasks import Task
from .utils import default_input, default_output, default_callback
from .experiments import Experiment, Cohort
from .criteria import Criteria

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
]