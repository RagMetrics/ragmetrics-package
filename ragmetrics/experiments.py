import concurrent.futures
import requests
import time
from tqdm import tqdm
from .api import ragmetrics_client  

class Experiment:
    ALLOWED_TYPES = {"Compare models", "Compare prompts", "Advanced"}
    
    def __init__(self, name, dataset, task, type, description, criteria, judge_model):
        self.name = name
        self.dataset = dataset
        self.task = task
        self.type = type  # Must be one of the ALLOWED_TYPES
        self.description = description  # Varies by type
        self.criteria = criteria  # List of criteria objects (or names)
        self.judge_model = judge_model

    def _build_payload(self):
        # Validate experiment type.
        if self.type not in self.ALLOWED_TYPES:
            raise ValueError(f"Type must be one of {self.ALLOWED_TYPES}")

        if self.type == "Compare models":
            if not isinstance(self.description, list) or not all(isinstance(item, str) for item in self.description):
                raise ValueError("For 'Compare models' type, description must be a list of model names (strings).")
            cohorts = [{"name": model, "model": model} for model in self.description]

        elif self.type == "Compare prompts":
            if not isinstance(self.description, list) or not all(
                isinstance(item, dict) and "name" in item and ("prompt" in item or "system_prompt" in item)
                for item in self.description
            ):
                raise ValueError("For 'Compare prompts' type, description must be a list of dicts with keys 'Name' and 'system_prompt'.")
            cohorts = []
            for item in self.description:
                new_item = item.copy()
                new_item.setdefault("generator_model", "gpt-4o-mini")
                cohorts.append(new_item)

        elif self.type == "Advanced":
            if not isinstance(self.description, list) or not all(isinstance(item, dict) for item in self.description):
                raise ValueError("For 'Advanced' type, description must be a list of dictionaries.")
            for item in self.description:
                if not ("generator_model" in item or "rag_pipeline" in item):
                    raise ValueError("For 'Advanced' type, each item must include either 'generator_model' or 'rag_pipeline'.")
            cohorts = self.description

        payload = {
            "experiment_name": self.name,
            "dataset": self.dataset,
            "task": self.task,
            "exp_type": self.type,
            "criteria": self.criteria,
            "judge_model": self.judge_model,
            "cohorts": cohorts,
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
        Returns a Future that will be fulfilled with the JSON response
        from the API.
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
        # Submit the experiment run asynchronously.
        future_result = self.run_async()
        initial_result = future_result.result()
        
        if initial_result.get('status') != 'running':
            raise Exception(f"Experiment failed to start: {initial_result.get('message', 'Unknown error')}")
        
        experiment_run_id = initial_result["experiment_run_id"]
        results_url = initial_result["results_url"]
        
        # Set up progress tracking.
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        base_url = ragmetrics_client.base_url.rstrip('/')
        progress_url = f"{base_url}/api/experiment/progress/{experiment_run_id}/"
        
        # Initialize the progress bar using tqdm.
        with tqdm(total=100, desc="Experiment Progress", 
                  bar_format="{l_bar}{bar}| {n_fmt}% [{elapsed}<{remaining}]") as pbar:
            last_progress = 0
            retry_count = 0
            
            while True:
                try:
                    # Request progress update.
                    response = requests.get(progress_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    progress_data = response.json()
                    
                    # Handle error states from the API.
                    if progress_data.get('state') == 'FAILED':
                        raise Exception(f"Experiment failed: {progress_data.get('error', 'Unknown error')}")
                    
                    # Update the progress bar.
                    current_progress = progress_data.get('progress', 0)
                    pbar.update(current_progress - last_progress)
                    last_progress = current_progress
                    
                    # Display ETA and status if available.
                    if progress_data.get('eta_lower') is not None:
                        pbar.set_postfix({
                            'ETA': f"{progress_data['eta_lower']}-{progress_data['eta_upper']}min",
                            'Status': progress_data.get('description', '')
                        })
                    
                    # Exit if experiment is complete.
                    if progress_data.get('state') in ['COMPLETED', 'SUCCESS']:
                        pbar.set_postfix({'Status': 'Completed!'})
                        print(f"\nResults available at: {base_url}{results_url}")
                        return progress_data
                    
                    # Reset retry counter on success.
                    retry_count = 0
                    
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retry_count += 1
                    if retry_count > 3:
                        raise Exception("Failed to connect to progress endpoint after 3 retries")
                    pbar.set_postfix({'Status': f"Connection error, retrying ({retry_count}/3)..."})
                    time.sleep(poll_interval * 2)
                
                time.sleep(poll_interval)
