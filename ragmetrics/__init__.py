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

from ragmetrics.api import login, monitor, trace_function_call
from ragmetrics.dataset import Example, Dataset
from ragmetrics.tasks import Task
from ragmetrics.experiments import Experiment, Cohort
from ragmetrics.criteria import Criteria
from ragmetrics.reviews import ReviewQueue
from ragmetrics.trace import Trace
from ragmetrics.utils import import_function