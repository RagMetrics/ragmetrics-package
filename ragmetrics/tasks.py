from .api import RagMetricsObject  # This is your HTTP client wrapper for RagMetrics

class Task(RagMetricsObject):
    """
    A class representing a specific task configuration for LLM evaluations.
    
    Tasks define how models should generate responses for each example in a dataset.
    This includes specifying the prompt format, system message, and any other
    parameters needed for generation.
    """

    object_type = "task" 

    def __init__(self,
                 name,
                 generator_model=None,
                 system_prompt=None,
                 prompt_template=None,
                 max_tokens=None,
                 temperature=None,
                 stop=None):
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
                    system_prompt="You are a helpful assistant that answers questions accurately and concisely.",
                    prompt_template="Question: {question}\nAnswer:"
                )
                
                # Save the task for future use
                qa_task.save()
        
        Example - Creating a task with advanced parameters:
        
            .. code-block:: python
            
                # Task with temperature and token settings for creative writing
                creative_task = Task(
                    name="Creative Writing Task",
                    generator_model="claude-3-opus-20240229",
                    system_prompt="You are an award-winning novelist with a lyrical style.",
                    prompt_template="Write a short story about {question} in the style of magical realism.",
                    max_tokens=1000,
                    temperature=0.8,
                    stop=["THE END", "###"]
                )
                
                # Save the task
                creative_task.save()
            
        Example - Creating a task for RAG evaluation:
        
            .. code-block:: python
            
                # RAG evaluation task that includes context
                rag_task = Task(
                    name="RAG Evaluation",
                    generator_model="gpt-4",
                    system_prompt="Answer the question using only the provided context. If the context doesn't contain the answer, say 'I don't know'.",
                    prompt_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                )
                
                # Save the task
                rag_task.save()

    
    Args:
            name (str): The name of the task.
            generator_model (str, optional): Default model for generation if not specified in cohort.
            system_prompt (str, optional): System prompt to use when generating responses.
            prompt_template (str, optional): Template for formatting the prompt with example fields.
            max_tokens (int, optional): Maximum tokens to generate in responses.
            temperature (float, optional): Sampling temperature for generation.
            stop (list or str, optional): Sequences where the model should stop generating.
        """
        self.name = name
        self.generator_model = generator_model
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop
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
            "taskModel": self.generator_model,
            "promptTemplate": self.prompt_template,
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": self.stop
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
            generator_model=data.get("taskModel", ""),
            prompt_template=data.get("promptTemplate", ""),
            max_tokens=data.get("maxTokens", None),
            temperature=data.get("temperature", None),
            stop=data.get("stop", None)
        )
        task.id = data.get("id")
        return task

    @classmethod
    def download(cls, name=None, task_id=None):
        """
        Download a Task from the RagMetrics server.
        
        You must provide either a name or a task_id to identify the task to download.
        
        Example - Download by name:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Task
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Download a task by name
                qa_task = Task.download(name="Question Answering")
                
                # Use the downloaded task in an experiment
                experiment = Experiment(
                    name="Evaluation",
                    dataset=dataset,
                    task=qa_task,
                    # other parameters...
                )
            
        Example - Download by ID:
        
            .. code-block:: python
            
                # Download a task by its ID
                task = Task.download(task_id="task_1234567890")

    
    Args:
            name (str, optional): The name of the task to download.
            task_id (str, optional): The ID of the task to download.

    
    Returns:
            Task: The downloaded Task instance.

    
    Raises:
            ValueError: If neither name nor task_id is provided, or if the task
                is not found on the server.

        """
        from .api import api_client
        
        if not name and not task_id:
            raise ValueError("Either name or task_id must be provided")
        
        if task_id:
            response = api_client.get(f"/tasks/{task_id}")
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve task: {response.text}")
            return cls.from_dict(response.json())
        
        # Search by name
        response = api_client.get("/tasks")
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve tasks: {response.text}")
        
        tasks = response.json()
        for task in tasks:
            if task.get("taskName") == name:
                return cls.from_dict(task)
        
        raise ValueError(f"Task with name '{name}' not found")

    def save(self):
        """
        Save the Task to the RagMetrics server.
        
        If a task with the same name already exists, it will be updated.
        
        Example:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Task
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Create a task
                task = Task(
                    name="Custom QA Task",
                    generator_model="gpt-4",
                    system_prompt="You are a helpful assistant.",
                    prompt_template="Question: {question}\nAnswer:"
                )
                
                # Save the task to the server
                task.save()

    
    Returns:
            str: The ID of the saved task.

    
    Raises:
            Exception: If saving fails due to network issues or invalid data.
        """
        from .api import api_client
        
        payload = self.to_dict()
        
        if self.id:
            # Update existing task
            response = api_client.put(f"/tasks/{self.id}", json=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to update task: {response.text}")
            self.id = response.json().get("id")
        else:
            # Create new task
            response = api_client.post("/tasks", json=payload)
            if response.status_code != 201:
                raise Exception(f"Failed to create task: {response.text}")
            self.id = response.json().get("id")
            
        return self.id
