"""
RagMetrics Python Client

A comprehensive toolkit for monitoring, evaluating, and improving LLM applications.

This package provides tools for:
- Monitoring LLM applications in production
- Evaluating LLM responses with custom criteria
- Running experiments to compare different models or RAG implementations
- Managing datasets for systematic evaluation
- Creating and executing review queues for human-in-the-loop evaluation

Main components:
- login: Authenticate with the RagMetrics API
- monitor: Wrap LLM clients to automatically log interactions
- trace_function_call: Decorator to trace function execution
- Dataset/Example: Classes for managing evaluation datasets
- Task: Define evaluation tasks
- Experiment/Cohort: Run controlled experiments
- Criteria: Define evaluation criteria
- ReviewQueue: Manage human reviews of LLM interactions
- Trace: Access and manipulate logged interactions
"""

from ragmetrics.api import login, monitor, trace_function_call
from ragmetrics.dataset import Example, Dataset
from ragmetrics.tasks import Task
from ragmetrics.experiments import Experiment, Cohort
from ragmetrics.criteria import Criteria
from ragmetrics.reviews import ReviewQueue
from ragmetrics.trace import Trace