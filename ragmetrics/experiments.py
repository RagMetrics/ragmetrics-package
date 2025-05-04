import concurrent.futures
import requests
import time
import json
from tqdm import tqdm
from ragmetrics.api import ragmetrics_client  
from ragmetrics.tasks import Task
from ragmetrics.dataset import Dataset 
from ragmetrics.criteria import Criteria
from ragmetrics.utils import import_function

# --- Cohort Object ---
class Cohort:
    """
    Represents a cohort for an experiment.

    A cohort defines a specific configuration to test in an experiment. It can
    represent either a single model or a RAG pipeline configuration. Cohorts
    allow comparing different setups against the same dataset and criteria.
    """

    def __init__(
            self, 
            name, 
            generator_model=None, 
            rag_pipeline=None, 
            system_prompt=None, 
            function=None, 
            **kwargs
        ):
        """
        Initialize a Cohort instance.

        A cohort defines a specific configuration to test in an experiment. It can
        represent either a single model or a RAG pipeline configuration. Cohorts
        allow comparing different setups against the same dataset and criteria.

        Note: A cohort must include exactly one of: generator_model, rag_pipeline, or function_name.

        Example - Creating model cohorts:
        
            .. code-block:: python
            
                # For comparing different models:
                cohorts = [
                    Cohort(name="GPT-4", generator_model="gpt-4"),
                    Cohort(name="Claude 3 Sonnet", generator_model="claude-3-sonnet-20240229"),
                    Cohort(name="Llama 3", generator_model="llama3-8b-8192")
                ]
                
                # For comparing different models with custom system prompts:
                cohorts = [
                    Cohort(
                        name="GPT-4 with QA Prompt", 
                        generator_model="gpt-4", 
                        system_prompt="You are a helpful assistant that answers questions accurately."
                    ),
                    Cohort(
                        name="GPT-4 with Concise Prompt", 
                        generator_model="gpt-4", 
                        system_prompt="Provide extremely concise answers with minimal explanation."
                    )
                ]
            
        Example - Creating RAG pipeline cohorts:
        
            .. code-block:: python
            
                # For comparing different RAG approaches:
                cohorts = [
                    Cohort(name="Basic RAG", rag_pipeline="basic-rag-pipeline"),
                    Cohort(name="Query Rewriting RAG", rag_pipeline="query-rewriting-rag"),
                    Cohort(name="Hypothetical Document Embeddings", rag_pipeline="hyde-rag")
                ]
                
        Example - Creating function cohorts:
        
            .. code-block:: python
            
                # For using a local function with string reference:
                cohorts = [
                    Cohort(name="My Function", function="my_module.my_function")
                ]
                
                # For using a local function with callable:
                def my_processor(input, cohort):
                    # Process input based on cohort config
                    return {"generated_answer": "Some response"}
                    
                cohorts = [
                    Cohort(name="My Processor", function=my_processor)
                ]

        Args:
            name (str): Name of the cohort.
            generator_model (str, optional): Name/ID of the LLM model to use for generation.
            rag_pipeline (str, optional): Name/ID of the RAG pipeline to use.
            system_prompt (str, optional): System prompt to use with the generator model.
            function (callable or str, optional): Function to use for local execution instead of an API call.
                Can be a callable function or a string in the format "module.submodule.function_name".
                The function should take an input (e.g., question from dataset) and the cohort object,
                and return a dictionary with generated outputs.
            **kwargs: Additional keyword arguments to set as cohort attributes.
        """
        self.name = name
        # Set built-in arguments first
        self.function = import_function(function) if function else None
        self.generator_model = generator_model
        self.rag_pipeline = rag_pipeline 
        self.system_prompt = system_prompt
        
        # Set additional kwargs
        for key, value in kwargs.items():
            if key not in ['generator_model', 'rag_pipeline', 'system_prompt', 'function']:
                setattr(self, key, value)

    def to_dict(self):
        """
        Convert the Cohort instance to a dictionary for API communication.
        
        Note: For function-based cohorts, if function is a string it will be included in the output.
        If function is a callable object, the function name will be included.

        Returns:
            dict: Dictionary containing the cohort's configuration for API use.
        """
        data = {"name": self.name}
        if self.generator_model:
            data["generator_model"] = self.generator_model
        if self.rag_pipeline:
            data["rag_pipeline"] = self.rag_pipeline
        if self.system_prompt:
            data["system_prompt"] = self.system_prompt
        if self.function:
            if isinstance(self.function, str):
                data["function"] = self.function
            elif callable(self.function):
                data["function"] = self.function.__name__
        return data

    def __str__(self):
        """Return a human-readable string representation of the cohort."""
        if isinstance(self.function, str):
            return f"Cohort('{self.name}', function='{self.function}')"
        elif callable(self.function):
            return f"Cohort('{self.name}', function='{self.function.__name__}')"
        elif self.generator_model:
            return f"Cohort('{self.name}', model='{self.generator_model}')"
        elif self.rag_pipeline:
            return f"Cohort('{self.name}', pipeline='{self.rag_pipeline}')"
        else:
            return f"Cohort('{self.name}')"
    
    def __repr__(self):
        """Return a string representation of the cohort for debugging."""
        return self.__str__()

# --- Experiment Object ---
class Experiment:
    """
    A class representing an evaluation experiment.
    
    An Experiment orchestrates the evaluation of one or more cohorts (model configurations)
    against a dataset using specified criteria. It handles all the complexity of 
    coordinating the API calls, tracking progress, and retrieving results.
    
    Experiments are the core way to systematically evaluate and compare LLM configurations
    in RagMetrics.
    """

    def __init__(self, name, dataset, task, criteria, judge_model, cohorts=None):
        """
        Initialize a new Experiment instance.
        
        Example - Basic experiment with existing components:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Experiment, Cohort, Dataset, Task, Criteria
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Download existing components by name
                dataset = Dataset.download(name="Geography QA")
                task = Task.download(name="Question Answering")
                
                # Create cohorts to compare
                cohorts = [
                    Cohort(name="GPT-4", generator_model="gpt-4"),
                    Cohort(name="Claude 3", generator_model="claude-3-sonnet-20240229")
                ]
                
                # Use existing criteria (by name)
                criteria = ["Accuracy", "Relevance", "Conciseness"]
                
                # Create and run experiment
                experiment = Experiment(
                    name="Model Comparison - Geography",
                    dataset=dataset,
                    task=task,
                    criteria=criteria,
                    judge_model="gpt-4",
                    cohorts=cohorts  # Optional - will be auto-created from task if not provided
                )
                
                # Run the experiment and wait for results
                results = experiment.run()
        
        Example - Complete experiment creation flow:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Experiment, Cohort, Dataset, Task, Criteria, Example
                
                # Login
                ragmetrics.login("your-api-key")
                
                # 1. Create a dataset
                examples = [
                    Example(
                        question="What is the capital of France?",
                        ground_truth_context="France is a country in Western Europe. Its capital is Paris.",
                        ground_truth_answer="Paris"
                    ),
                    Example(
                        question="What is the largest planet in our solar system?",
                        ground_truth_context="Jupiter is the largest planet in our solar system.",
                        ground_truth_answer="Jupiter"
                    )
                ]
                dataset = Dataset(name="General Knowledge QA", examples=examples)
                dataset.save()
                
                # 2. Create a task
                task = Task(
                    name="General QA Task",
                    generator_model="gpt-4",
                    system_prompt="You are a helpful assistant that answers questions accurately."
                )
                task.save()
                
                # 3. Create criteria
                relevance = Criteria(
                    name="Relevance",
                    phase="generation",
                    output_type="5-point",
                    criteria_type="llm_judge",
                    header="How relevant is the response to the question?",
                    likert_score_1="Not relevant at all",
                    likert_score_2="Slightly relevant",
                    likert_score_3="Moderately relevant",
                    likert_score_4="Very relevant",
                    likert_score_5="Completely relevant"
                )
                relevance.save()
                
                factual = Criteria(
                    name="Factual Accuracy",
                    phase="generation",
                    output_type="bool", 
                    criteria_type="llm_judge",
                    header="Is the answer factually correct?",
                    bool_true="Yes, the answer is factually correct.",
                    bool_false="No, the answer contains factual errors."
                )
                factual.save()
                
                # 4. Define cohorts (optional)
                cohorts = [
                    Cohort(name="GPT-4", generator_model="gpt-4"),
                    Cohort(name="Claude 3", generator_model="claude-3-sonnet-20240229"),
                    Cohort(name="GPT-3.5", generator_model="gpt-3.5-turbo")
                ]
                
                # 5. Create experiment (with optional cohorts)
                experiment = Experiment(
                    name="Model Comparison - General Knowledge",
                    dataset=dataset,
                    task=task,
                    criteria=[relevance, factual],
                    judge_model="gpt-4",
                    cohorts=cohorts  # Optional - will be auto-created from task if not provided
                )
                
                # 6. Run the experiment
                results = experiment.run()
                
        Example - Experiment with default cohort from task:
        
            .. code-block:: python
            
                # Create experiment without specifying cohorts
                # A default cohort will be created based on the task's function or generator_model
                experiment = Experiment(
                    name="Auto Cohort Example",
                    dataset=dataset,
                    task=task,
                    criteria=criteria,
                    judge_model="gpt-4"
                    # cohorts parameter omitted - will create default cohort from task
                )
                
                # Print cohorts that were automatically created
                print(experiment.cohorts)
                
                results = experiment.run()

    
    Args:
            name (str): The name of the experiment.
            dataset (Dataset or str): The dataset to use for evaluation.
            task (Task or str): The task definition to evaluate.
            criteria (list or str): List of evaluation criteria.
            judge_model (str): The model to use for judging responses.
            cohorts (list or str, optional): List of cohorts to evaluate, or JSON string.
                If not provided, a default cohort will be created based on the task configuration.
        """
        self.name = name
        self.judge_model = judge_model
        self._downloaded_dataset = None
        self._downloaded_task = None
        
        # Process in order of dependency
        # 1. Process dataset - returns the processed Dataset object
        self.dataset = self._process_dataset(dataset)
        
        # 2. Process task - returns the processed Task object
        self.task = self._process_task(task)
        
        # 3. Process criteria - returns a list of processed Criteria objects
        self.criteria = self._process_criteria(criteria)
        
        # 4. Process cohorts - returns a list of processed Cohort objects
        # If cohorts is None, default cohorts will be created based on task
        self.cohorts = self._process_cohorts(cohorts)
    
    def _process_dataset(self, dataset):
        """
        Process and validate the dataset parameter.
        
        Handles different ways of specifying a dataset (object, name, ID) and ensures
        it exists on the server.

        Args:
            dataset (Dataset or str): The dataset to process.

        Returns:
            Dataset: The processed dataset object with ID populated.

        Raises:
            ValueError: If the dataset is invalid or missing required attributes.
            Exception: If the dataset cannot be found on the server.
        """
        if isinstance(dataset, Dataset):
            # Check if full attributes are present.
            if dataset.name and getattr(dataset, "examples", None) and len(dataset.examples) > 0:
                # Full dataset provided: save it to get a new id.
                dataset.save()
                self._downloaded_dataset = dataset
                return dataset
            else:
                # If only id or name is provided.
                if getattr(dataset, "id", None):
                    downloaded = Dataset.download(id=dataset.id)
                    if downloaded and getattr(downloaded, "id", None):
                        self._downloaded_dataset = downloaded
                        return downloaded
                elif getattr(dataset, "name", None):
                    downloaded = Dataset.download(name=dataset.name)
                    if downloaded and getattr(downloaded, "id", None):
                        self._downloaded_dataset = downloaded
                        return downloaded
                    else:
                        raise Exception(f"Dataset with name '{dataset.name}' not found on server.")
                else:
                    raise Exception("Dataset object missing required attributes.")
        elif isinstance(dataset, str):
            downloaded = Dataset.download(name=dataset)
            if downloaded and getattr(downloaded, "id", None):
                self._downloaded_dataset = downloaded
                return downloaded
            else:
                raise Exception(f"Dataset not found on server with name: {dataset}")
        else:
            raise ValueError("Dataset must be a Dataset object or a string.")

    def _process_task(self, task):
        """
        Process and validate the task parameter.
        
        Handles different ways of specifying a task (object, name, ID) and ensures
        it exists on the server.

        Args:
            task (Task or str): The task to process.

        Returns:
            Task: The processed task object with ID populated.

        Raises:
            ValueError: If the task is invalid or missing required attributes.
            Exception: If the task cannot be found on the server.
        """
        if isinstance(task, Task):
            # Check for full attributes: name, system_prompt, and generator_model
            if task.name \
                and (getattr(task, "generator_model", None) or 
                     getattr(task, "function", None)):
                task.save()
                self._downloaded_task = task
                return task
            else:
                if getattr(task, "id", None):
                    downloaded = Task.download(id=task.id)
                    if downloaded and getattr(downloaded, "id", None):
                        self._downloaded_task = downloaded
                        return downloaded
                elif getattr(task, "name", None):
                    downloaded = Task.download(name=task.name)
                    if downloaded and getattr(downloaded, "id", None):
                        self._downloaded_task = downloaded
                        return downloaded
                    else:
                        raise Exception(f"Task with name '{task.name}' not found on server.")
                else:
                    raise Exception("Task object missing required attributes.")
        elif isinstance(task, str):
            downloaded = Task.download(name=task)
            if downloaded and getattr(downloaded, "id", None):
                self._downloaded_task = downloaded
                return downloaded
            else:
                raise Exception(f"Task not found on server with name: {task}")
        else:
            raise ValueError("Task must be a Task object or a string.")

    def _process_cohorts(self, cohorts):
        """
        Process and validate the cohorts parameter.
        
        Converts the cohorts parameter (list of Cohort objects, dictionaries, or JSON string)
        to a properly formatted list of Cohort objects.
        
        If no cohorts are provided, creates a default cohort:
        - If task has a function, creates a cohort with name=function_name and function=function
        - If task has a generator_model, creates a cohort with name=generator_model and generator_model=generator_model

        Args:
            cohorts (list, str, or None): Cohorts to process, or None to create default cohorts
            
        Returns:
            list: List of processed Cohort objects
            
        Raises:
            ValueError: If cohorts are invalid or improperly configured.
        """
        
        if not cohorts:
            # Use the _cohorts_default method to create default cohorts
            return self._cohorts_default()      
        
        if isinstance(cohorts, str):
            try:
                cohorts_list_dicts = json.loads(cohorts)
                # Convert dictionaries to Cohort objects
                cohorts_list = [Cohort(**c) for c in cohorts_list_dicts]
            except Exception as e:
                raise ValueError("Invalid JSON for cohorts: " + str(e))
        elif isinstance(cohorts, list):
            cohorts_list = []
            
            for c in cohorts:
                if isinstance(c, Cohort):
                    cohorts_list.append(c)
                elif isinstance(c, dict):
                    cohort_obj = Cohort(**c)
                    cohorts_list.append(cohort_obj)
                else:
                    raise ValueError("Each cohort must be a Cohort object or a dict.")
        else:
            raise ValueError("Cohorts must be provided as a JSON string or a list.")
        
        # Validate: Each cohort must have exactly one generator field
        for cohort in cohorts_list:
            cohort_dict = cohort.to_dict()
            main_keys = ["generator_model", "rag_pipeline", "function"]
            generator_count = sum(key in cohort_dict for key in main_keys)
            if generator_count == 0:
                #If missing, fill it in from the task
                if self.task.function:
                    cohort.function = self.task.function
                elif self.task.generator_model:
                    cohort.generator_model = self.task.generator_model
                elif self.task.rag_pipeline:
                    cohort.rag_pipeline = self.task.rag_pipeline
                else:
                    raise ValueError("Each cohort (or the task) must include either 'generator_model', 'rag_pipeline', or 'function'.")            
            elif generator_count > 1:
                raise ValueError("Each cohort should include only one of ['generator_model', 'rag_pipeline', 'function']")
            
        return cohorts_list

    def _process_criteria(self, criteria):
        """
        Process and validate the criteria parameter.
        
        Handles different ways of specifying criteria (objects, names, IDs) and ensures
        they exist on the server.

        Args:
            criteria (list or str): The criteria to process.

        Returns:
            list: List of processed Criteria objects.

        Raises:
            ValueError: If the criteria are invalid.
            Exception: If criteria cannot be found on the server.
        """
        processed_criteria = []
        criteria_ids = []  # We still need IDs for the API payload
        
        if isinstance(criteria, list):
            for crit in criteria:
                if isinstance(crit, Criteria):
                    if getattr(crit, "id", None):
                        criteria_ids.append(crit.id)
                        processed_criteria.append(crit)
                    else:
                        # Check that required fields are nonempty
                        if (crit.name and crit.name.strip() and
                            crit.phase and crit.phase.strip() and
                            crit.output_type and crit.output_type.strip() and
                            crit.criteria_type and crit.criteria_type.strip()):
                            crit.save()
                            criteria_ids.append(crit.id)
                            processed_criteria.append(crit)
                        else:
                            # Otherwise, try to download by name as a reference.
                            try:
                                downloaded = Criteria.download(name=crit.name)
                                if downloaded and getattr(downloaded, "id", None):
                                    criteria_ids.append(downloaded.id)
                                    processed_criteria.append(downloaded)
                                else:
                                    raise Exception(f"Criteria with name '{crit.name}' not found on server.")
                            except Exception as e:
                                raise Exception(
                                    f"Criteria '{crit.name}' is missing required attributes (phase, output type, or criteria type) and lookup failed: {str(e)}"
                                )
                elif isinstance(crit, str):
                    try:
                        downloaded = Criteria.download(name=crit)
                        if downloaded and getattr(downloaded, "id", None):
                            criteria_ids.append(downloaded.id)
                            processed_criteria.append(downloaded)
                        else:
                            raise Exception(f"Criteria with name '{crit}' not found on server.")
                    except Exception as e:
                        raise Exception(f"Criteria lookup failed for '{crit}': {str(e)}")
                else:
                    raise ValueError("Each Criteria must be a Criteria object or a string.")
            
            # Store the criteria IDs for the API payload
            self._criteria_ids = criteria_ids
            return processed_criteria
            
        elif isinstance(criteria, str):
            downloaded = Criteria.download(name=criteria)
            if downloaded and getattr(downloaded, "id", None):
                self._criteria_ids = [downloaded.id]
                return [downloaded]
            else:
                raise Exception(f"Criteria not found on server with name: {criteria}")
        else:
            raise ValueError("Criteria must be provided as a list of Criteria objects or a string.")

    def _process_function(self):
        """
        Process and execute the function parameter from the task.
        
        If a callable is provided, run it on each row of the already downloaded dataset,
        and return a dict of {cohort_name: function_output_list}.        
        If a string is provided with module path, attempt to import it first, then run it if successful.
        
        Returns:
            dict: {cohort_name: function_output_list}
            
        Raises:
            Exception: If dataset is not available or function execution fails
        """
        cross_cohort_outputs = {}
        for cohort in self.cohorts:
            if not cohort.function:
                continue
            
            if callable(cohort.function):
                local_function = cohort.function
            elif isinstance(cohort.function, str):
                local_function = import_function(cohort.function)
            else:
                raise ValueError(f"Expected function to be callable or a string, got {local_function} of type {type(local_function)}.")            

            # Execute the local function on each example
            cohort_outputs = []            
            for example in self._downloaded_dataset.examples:
                input = example.question
                output = local_function(input, cohort)
                cohort_outputs.append(output)
                
            cross_cohort_outputs[cohort.name] = cohort_outputs

        return cross_cohort_outputs


    def _build_payload(self):
        """
        Build the payload for the API request.
        
        Uses the processed components of the experiment to construct the complete
        payload to send to the server.

        Returns:
            dict: The payload to send to the server.
        """
        # Make sure all components are processed
        if not hasattr(self, '_downloaded_dataset') or self._downloaded_dataset is None:
            self.dataset = self._process_dataset(self.dataset)
            
        if not hasattr(self, '_downloaded_task') or self._downloaded_task is None:
            self.task = self._process_task(self.task)
            
        if not hasattr(self, '_criteria_ids'):
            self.criteria = self._process_criteria(self.criteria)
            
        # Make sure cohorts are processed
        if not hasattr(self, 'cohorts') or self.cohorts is None:
            self.cohorts = self._process_cohorts(self.cohorts)
                        
        payload = {
            "experiment_name": self.name,
            "dataset": self._downloaded_dataset.id,
            "task": self._downloaded_task.id,
            "exp_type": "advanced",  
            "criteria": self._criteria_ids,
            "judge_model": self.judge_model,
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
            "function_output": self._process_function()
        }
            
        return payload

    def _call_api(self, payload):
        """
        Make the API call to run the experiment.
        
        Sends the experiment configuration to the server and handles the response.

    
    Args:
            payload (dict): The payload to send to the API.

    
    Returns:
            dict: The API response.

    
    Raises:
            Exception: If the API call fails.
        """
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        response = ragmetrics_client._make_request(
            method="post",
            endpoint="/api/client/experiment/run/",
            json=payload,
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to run experiment: " + response.text)

    def run_async(self):
        """
        Submit the experiment asynchronously.
        
        Starts the experiment on the server without waiting for it to complete.
        Use this when you want to start an experiment and check its status later.

    
    Returns:
            concurrent.futures.Future: A Future object that will contain the API response.
        """
        payload = self._build_payload()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._call_api, payload)
        return future

    def run(self, poll_interval=2):
        """
        Run the experiment and display real-time progress.
        
        This method submits the experiment to the server and then polls for progress
        updates, displaying a progress bar. It blocks until the experiment completes
        or fails.
        
        Example:
        
            .. code-block:: python
            
                # Create the experiment
                experiment = Experiment(
                    name="Model Comparison",
                    dataset="My Dataset",
                    task="QA Task",
                    cohorts=cohorts,
                    criteria=criteria,
                    judge_model="gpt-4"
                )
                
                # Run with default polling interval (2 seconds)
                results = experiment.run()
                
                # Or run with custom polling interval
                results = experiment.run(poll_interval=5)  # Check every 5 seconds

    
    Args:
            poll_interval (int): Time between progress checks in seconds (default: 2).

    
    Returns:
            dict: The experiment results once completed.

    
    Raises:
            Exception: If the experiment fails to start or encounters an error.
        """
        future_result = self.run_async()
        initial_result = future_result.result()
        
        if initial_result.get('status') != 'running':
            raise Exception(f"Experiment failed to start: {initial_result.get('message', 'Unknown error')}")
        
        experiment_run_id = initial_result["experiment_run_id"]
        results_url = initial_result["results_url"]
        base_url = ragmetrics_client.base_url.rstrip('/')
        
        # Print a single status message.
        print(f'Experiment "{self.name}" is running. Check progress at: {base_url}{results_url}')
        
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        progress_url = f"{base_url}/api/experiment/progress/{experiment_run_id}/"
        
        with tqdm(total=100, desc="Progress", bar_format="{l_bar}{bar}| {n_fmt}%[{elapsed}<{remaining}]") as pbar:
            last_progress = 0
            retry_count = 0
            
            while True:
                try:
                    response = requests.get(progress_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    progress_data = response.json()
                    
                    if progress_data.get('state') == 'FAILED':
                        raise Exception(f"Experiment failed: {progress_data.get('error', 'Unknown error')}")
                    
                    current_progress = progress_data.get('progress', 0)
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                    
                    if progress_data.get('state') in ['COMPLETED', 'SUCCESS']:
                        pbar.update(100 - last_progress)  
                        pbar.set_postfix({'Status': 'Finished!'})
                        pbar.close()  
                        tqdm.write(f"Finished!")
                        return progress_data
                    
                    retry_count = 0
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retry_count += 1
                    if retry_count > 3:
                        raise Exception("Failed to connect to progress endpoint after 3 retries")
                    pbar.set_postfix({'Status': f"Connection error, retrying ({retry_count}/3)..."})
                    time.sleep(poll_interval * 2)
                
                time.sleep(poll_interval)

    def __str__(self):
        """
        Return a string representation of the experiment for easier debugging.
        """
        return f"Experiment(name='{self.name}', cohorts={self.cohorts})"

    def _cohorts_default(self):
        """
        Create default cohorts based on the task configuration.
        
        Returns:
            list: List of default Cohort objects based on the task's function or generator_model.
            
        Raises:
            ValueError: If the task doesn't have a function or generator_model.
        """
        # Make sure task is processed
        if self._downloaded_task is None:
            self._downloaded_task = self._process_task(self.task)
            
        # Create cohort based on task properties
        if self._downloaded_task.function:
            # Task has a function - create cohort with function name
            if callable(self._downloaded_task.function):
                name = self._downloaded_task.function.__name__
            elif isinstance(self._downloaded_task.function, str):
                # Function is a string
                name = self._downloaded_task.function
            else:
                raise ValueError(f"Expected function to be callable or a string, got {self._downloaded_task.function} of type {type(self._downloaded_task.function)}.")
                
            cohort = Cohort(name=name, function=self._downloaded_task.function)
                
        elif self._downloaded_task.generator_model:
            name = self._downloaded_task.generator_model            
            cohort = Cohort(name=name, generator_model=name)

        else:
            raise ValueError("Cannot create default cohort: task has neither generator_model nor function.")
        
        return [cohort]