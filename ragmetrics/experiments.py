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
    def __init__(self, name, generator_model=None, rag_pipeline=None, system_prompt=None):

        self.name = name
        self.generator_model = generator_model
        self.rag_pipeline = rag_pipeline
        self.system_prompt = system_prompt

    def to_dict(self):
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
    def __init__(self, name, dataset, task, cohorts, criteria, judge_model):
        self.name = name
        self.dataset = dataset
        self.task = task
        self.cohorts = cohorts   
        self.criteria = criteria
        self.judge_model = judge_model

    def _process_dataset(self, dataset):

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
        Processes the cohorts parameter:
          - If a string, assumes it's a JSON string.
          - If a list, converts each element to a dict (using to_dict() if available).
        Returns a JSON string.
        Validates that each cohort contains either "generator_model" or "rag_pipeline" (but not both).
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
        Processes the criteria parameter.
        Accepts a list of Criteria objects or strings.
        Returns a list of Criteria IDs.
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
        Builds the payload to send to the server.
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
        Submits the experiment asynchronously.
        Returns a Future that will be fulfilled with the JSON response from the API.
        """
        payload = self._build_payload()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._call_api, payload)
        return future

    def run(self, poll_interval=2):
        """
        Runs the experiment and displays real-time progress with enhanced error handling.
        This method wraps the asynchronous run (run_async) with a polling progress bar.
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
