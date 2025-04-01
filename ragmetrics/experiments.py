import concurrent.futures
import requests
import time
import json
from tqdm import tqdm
from .api import ragmetrics_client  
from .tasks import Task 
from .dataset import Dataset 
from .criteria import Criteria

# --- Cohort Object ---
class Cohort:
    """
    A class representing a group of models or pipelines to be evaluated.
    
    A cohort defines a specific configuration to test in an experiment. It can 
    represent either a single model or a RAG pipeline configuration. Cohorts 
    allow comparing different setups against the same dataset and criteria.
    """

    def __init__(self, name, generator_model=None, rag_pipeline=None, system_prompt=None):
        """
        Initialize a new Cohort instance.
        
        Note: A cohort must include either generator_model OR rag_pipeline, not both.
        
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

    
    Args:
            name (str): The name of the cohort (e.g., "GPT-4", "RAG-v1").
            generator_model (str, optional): The model identifier to use for generation.
            rag_pipeline (str, optional): The RAG pipeline configuration identifier.
            system_prompt (str, optional): Override system prompt to use with this cohort.
        """
        self.name = name
        self.generator_model = generator_model
        self.rag_pipeline = rag_pipeline
        self.system_prompt = system_prompt

    def to_dict(self):
        """
        Convert the Cohort instance to a dictionary for API communication.

    
    Returns:
            dict: Dictionary containing the cohort's configuration.
        """
        data = {"name": self.name}
        if self.generator_model:
            data["generator_model"] = self.generator_model
        if self.rag_pipeline:
            data["rag_pipeline"] = self.rag_pipeline
        if self.system_prompt:
            data["system_prompt"] = self.system_prompt
        return data

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

    def __init__(self, name, dataset, task, cohorts, criteria, judge_model):
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
                    cohorts=cohorts,
                    criteria=criteria,
                    judge_model="gpt-4"
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
                
                # 4. Define cohorts
                cohorts = [
                    Cohort(name="GPT-4", generator_model="gpt-4"),
                    Cohort(name="Claude 3", generator_model="claude-3-sonnet-20240229"),
                    Cohort(name="GPT-3.5", generator_model="gpt-3.5-turbo")
                ]
                
                # 5. Create experiment
                experiment = Experiment(
                    name="Model Comparison - General Knowledge",
                    dataset=dataset,
                    task=task,
                    cohorts=cohorts,
                    criteria=[relevance, factual],
                    judge_model="gpt-4"
                )
                
                # 6. Run the experiment
                results = experiment.run()

    
    Args:
            name (str): The name of the experiment.
            dataset (Dataset or str): The dataset to use for evaluation.
            task (Task or str): The task definition to evaluate.
            cohorts (list or str): List of cohorts to evaluate, or JSON string.
            criteria (list or str): List of evaluation criteria.
            judge_model (str): The model to use for judging responses.
        """
        self.name = name
        self.dataset = dataset
        self.task = task
        self.cohorts = cohorts   
        self.criteria = criteria
        self.judge_model = judge_model

    def _process_dataset(self, dataset):
        """
        Process and validate the dataset parameter.
        
        Handles different ways of specifying a dataset (object, name, ID) and ensures
        it exists on the server.

    
    Args:
            dataset (Dataset or str): The dataset to process.

    
    Returns:
            str: The ID of the processed dataset.

    
    Raises:
            ValueError: If the dataset is invalid or missing required attributes.
            Exception: If the dataset cannot be found on the server.
        """
        if isinstance(dataset, Dataset):
            # Check if full attributes are present.
            if dataset.name and getattr(dataset, "examples", None) and len(dataset.examples) > 0:
                # Full dataset provided: save it to get a new id.
                dataset.save()
                return dataset.id
            else:
                # If only id or name is provided.
                if getattr(dataset, "id", None):
                    downloaded = Dataset.download(id=dataset.id)
                    if downloaded and getattr(downloaded, "id", None):
                        dataset.id = downloaded.id
                        return dataset.id
                elif getattr(dataset, "name", None):
                    downloaded = Dataset.download(name=dataset.name)
                    if downloaded and getattr(downloaded, "id", None):
                        dataset.id = downloaded.id
                        return dataset.id
                    else:
                        raise Exception(f"Dataset with name '{dataset.name}' not found on server.")
                else:
                    raise Exception("Dataset object missing required attributes.")
        elif isinstance(dataset, str):
            downloaded = Dataset.download(name=dataset)
            if downloaded and getattr(downloaded, "id", None):
                return downloaded.id
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
            str: The ID of the processed task.

    
    Raises:
            ValueError: If the task is invalid or missing required attributes.
            Exception: If the task cannot be found on the server.
        """
        if isinstance(task, Task):
            # Check for full attributes: name, system_prompt, and generator_model
            if task.name  and getattr(task, "generator_model", None):
                task.save()
                return task.id
            else:
                if getattr(task, "id", None):
                    downloaded = Task.download(id=task.id)
                    if downloaded and getattr(downloaded, "id", None):
                        task.id = downloaded.id
                        return task.id
                elif getattr(task, "name", None):
                    downloaded = Task.download(name=task.name)
                    if downloaded and getattr(downloaded, "id", None):
                        task.id = downloaded.id
                        return task.id
                    else:
                        raise Exception(f"Task with name '{task.name}' not found on server.")
                else:
                    raise Exception("Task object missing required attributes.")
        elif isinstance(task, str):
            downloaded = Task.download(name=task)
            if downloaded and getattr(downloaded, "id", None):
                return downloaded.id
            else:
                raise Exception(f"Task not found on server with name: {task}")
        else:
            raise ValueError("Task must be a Task object or a string.")

    def _process_cohorts(self):
        """
        Process and validate the cohorts parameter.
        
        Converts the cohorts parameter (list of Cohort objects or JSON string) to
        a JSON string for the API. Validates that each cohort is properly configured.

    
    Returns:
            str: JSON string containing the processed cohorts.

    
    Raises:
            ValueError: If cohorts are invalid or improperly configured.
        """
        if isinstance(self.cohorts, str):
            try:
                cohorts_list = json.loads(self.cohorts)
            except Exception as e:
                raise ValueError("Invalid JSON for cohorts: " + str(e))
        elif isinstance(self.cohorts, list):
            cohorts_list = []
            for c in self.cohorts:
                if hasattr(c, "to_dict"):
                    cohorts_list.append(c.to_dict())
                elif isinstance(c, dict):
                    cohorts_list.append(c)
                else:
                    raise ValueError("Each cohort must be a dict or have a to_dict() method.")
        else:
            raise ValueError("cohorts must be provided as a JSON string or a list.")
        
        for cohort in cohorts_list:
            if not ("generator_model" in cohort or "rag_pipeline" in cohort):
                raise ValueError("Each cohort must include either 'generator_model' or 'rag_pipeline'.")
            if "generator_model" in cohort and "rag_pipeline" in cohort:
                raise ValueError("Each cohort must include either 'generator_model' or 'rag_pipeline', not both.")
        return json.dumps(cohorts_list, indent=4)

    def _process_criteria(self, criteria):
        """
        Process and validate the criteria parameter.
        
        Handles different ways of specifying criteria (objects, names, IDs) and ensures
        they exist on the server.

    
    Args:
            criteria (list or str): The criteria to process.

    
    Returns:
            list: List of criteria IDs.

    
    Raises:
            ValueError: If the criteria are invalid.
            Exception: If criteria cannot be found on the server.
        """
        criteria_ids = []
        if isinstance(criteria, list):
            for crit in criteria:
                if isinstance(crit, Criteria):
                    if getattr(crit, "id", None):
                        criteria_ids.append(crit.id)
                    else:
                        # Check that required fields are nonempty
                        if (crit.name and crit.name.strip() and
                            crit.phase and crit.phase.strip() and
                            crit.output_type and crit.output_type.strip() and
                            crit.criteria_type and crit.criteria_type.strip()):
                            crit.save()
                            criteria_ids.append(crit.id)
                        else:
                            # Otherwise, try to download by name as a reference.
                            try:
                                downloaded = Criteria.download(name=crit.name)
                                if downloaded and getattr(downloaded, "id", None):
                                    crit.id = downloaded.id
                                    criteria_ids.append(crit.id)
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
                        else:
                            raise Exception(f"Criteria with name '{crit}' not found on server.")
                    except Exception as e:
                        raise Exception(f"Criteria lookup failed for '{crit}': {str(e)}")
                else:
                    raise ValueError("Each Criteria must be a Criteriaobject or a string.")
            return criteria_ids
        elif isinstance(criteria, str):
            downloaded = Criteria.download(name=criteria)
            if downloaded and getattr(downloaded, "id", None):
                return [downloaded.id]
            else:
                raise Exception(f"Criteria not found on server with name: {criteria}")
        else:
            raise ValueError("Criteria must be provided as a list of Criteria objects or a string.")

    def _build_payload(self):
        """
        Build the payload for the API request.
        
        Processes all components of the experiment and constructs the complete
        payload to send to the server.

    
    Returns:
            dict: The payload to send to the server.
        """
        payload = {
            "experiment_name": self.name,
            "dataset": self._process_dataset(self.dataset),
            "task": self._process_task(self.task),
            "exp_type": "advanced",  
            "criteria": self._process_criteria(self.criteria),
            "judge_model": self.judge_model,
            "cohorts": self._process_cohorts(),
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
