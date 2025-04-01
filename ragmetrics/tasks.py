from .api import RagMetricsObject  # This is your HTTP client wrapper for RagMetrics

class Task(RagMetricsObject):
    """
    A class representing a specific task configuration for LLM evaluations.
    
    Tasks define how models should generate responses for each example in a dataset.
    This includes specifying the prompt format, system message, and any other
    parameters needed for generation.
    """

    object_type = "task" 

    def __init__(self, name, generator_model="", system_prompt=""):
        """
        Initialize a new Task instance.
        
        Example - Creating a simple QA task:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Task
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Create a basic QA task
                qa_task = Task(
                    name="Question Answering",
                    generator_model="gpt-4",
                    system_prompt="You are a helpful assistant that answers questions accurately and concisely."
                )
                
                # Save the task for future use
                qa_task.save()
        
        Example - Creating a RAG evaluation task:
        
            .. code-block:: python
            
                # RAG evaluation task
                rag_task = Task(
                    name="RAG Evaluation",
                    generator_model="gpt-4",
                    system_prompt="Answer the question using only the provided context. If the context doesn't contain the answer, say 'I don't know'."
                )
                
                # Save the task
                rag_task.save()

    
        Args:
            name (str): The name of the task.
            generator_model (str, optional): Default model for generation if not specified in cohort.
            system_prompt (str, optional): System prompt to use when generating responses.
        """
        self.name = name
        self.generator_model = generator_model
        self.system_prompt = system_prompt
        self.id = None

    def to_dict(self):
        """
        Convert the Task instance to a dictionary for API communication.

    
        Returns:
            dict: Dictionary containing the task configuration.
        """
        return {
            "taskName": self.name,
            "taskPrompt": self.system_prompt,
            "taskModel": self.generator_model
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Task instance from a dictionary.
        
        Used internally when downloading tasks from the RagMetrics API.

    
        Args:
            data (dict): Dictionary containing task information.

    
        Returns:
            Task: A new Task instance with the specified data.
        """
        task = cls(
            name=data.get("taskName", ""),
            system_prompt=data.get("taskPrompt", ""),
            generator_model=data.get("taskModel", "")
        )
        task.id = data.get("id")
        return task
