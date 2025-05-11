from ragmetrics.base_object import RagMetricsObject
from ragmetrics.dataset import Dataset
from ragmetrics.criteria import Criteria
from ragmetrics.trace import Trace

class ReviewQueue(RagMetricsObject):
    """
    Manages a queue of traces for manual review and evaluation.
    
    A ReviewQueue allows for structured human evaluation of LLM interactions
    by collecting traces that match specific conditions and applying evaluation
    criteria. It supports both automated and human-in-the-loop evaluation workflows.
    """
    
    object_type = "reviews"

    def __init__(self, name, condition="", criteria=None, judge_model=None, dataset=None, retroactive=False):
        """
        Initialize a new ReviewQueue instance.

    
    Args:
            name (str): The name of the review queue.
            condition (str, optional): SQL-like condition to filter traces (default: "").
            criteria (list or str, optional): Evaluation criteria to apply.
            judge_model (str, optional): LLM model to use for automated evaluation.
            dataset (Dataset or str, optional): Dataset to use for evaluation.
            retroactive (bool, optional): Whether the review queue should apply to existing traces (default: False).
        """
        from .api import logger
        self.name = name
        self.condition = condition
        self.id = None  
        self._traces = None
        self.edit_mode = False
        self.retroactive = retroactive
        self.judge_model = judge_model

        # Process criteria with proper error handling
        try:
            self.criteria = self._process_criteria(criteria)
        except Exception as e:
            logger.error(f"Error processing criteria during initialization: {e}")
            self.criteria = []

        # Process dataset with proper error handling
        try:
            # Special handling for dataset - keep original value for processing later
            self.dataset = dataset  # Store original value
        except Exception as e:
            logger.error(f"Error setting dataset during initialization: {e}")
            self.dataset = None

    def __setattr__(self, key, value):
        """
        Override attribute setting to enable edit mode when modifying an existing queue.
        
        This automatically sets edit_mode to True when any attribute (except edit_mode itself)
        is changed on a queue with an existing ID.
        
    
    Args:
            key (str): The attribute name.
            value: The value to set.
        """
        if key not in {"edit_mode"} and hasattr(self, "id") and self.id:
            object.__setattr__(self, "edit_mode", True)
        object.__setattr__(self, key, value)

    @property
    def traces(self):
        """
        Get the traces associated with this review queue.
        
        Lazily loads traces from the server if they haven't been loaded yet.
        
    
    Returns:
            list: List of Trace objects in this review queue.
        """
        if self.id is None:
            raise ValueError("Please save this review queue before retrieving traces.")
        
        if self._traces in [None, []]:
            self._traces = self._download_traces(review_queue_id=self.id) if self.id else []
        return self._traces

    @traces.setter
    def traces(self, value):
        """
        Set the traces associated with this review queue.
        
    
    Args:
            value (list): List of Trace objects to associate with this queue.
        """
        self._traces = value

    def _process_dataset(self, dataset):
        """
        Process and validate the dataset parameter.
        
        Converts various dataset representations (object, ID, name) to a dataset ID
        that can be used in API requests.
        
    
    Args:
            dataset (Dataset, int, str): The dataset to process.
            
    
    Returns:
            int: The ID of the processed dataset.
            
    
    Raises:
            ValueError: If the dataset is invalid or not found.
            Exception: If the dataset cannot be found on the server.
        """
        if dataset is None:
            return None
        
        elif isinstance(dataset, Dataset):
            # Check if full attributes are present or ID is already set
            if dataset.id is not None:
                return dataset.id
            elif dataset.name and getattr(dataset, "examples", None) and len(dataset.examples) > 0:
                # Full dataset provided: save it to get a new id.
                dataset.save()
                return dataset.id
            else:
                # If only name is provided in the Dataset object
                if getattr(dataset, "name", None):
                    try:
                        downloaded = Dataset.download(name=dataset.name)
                        if downloaded and getattr(downloaded, "id", None):
                            return downloaded.id
                        else:
                            raise Exception(f"Dataset with name '{dataset.name}' not found on server.")
                    except Exception as e:
                        from .api import logger
                        logger.error(f"Error downloading dataset '{dataset.name}': {e}")
                        raise
                else:
                    raise Exception("Dataset object missing required attributes.")
                
        elif isinstance(dataset, int):
            # If it's already an ID, just return it
            return dataset
            
        elif isinstance(dataset, str):
            try:
                downloaded = Dataset.download(name=dataset)
                if downloaded and getattr(downloaded, "id", None):
                    return downloaded.id
                else:
                    raise Exception(f"Dataset with name '{dataset}' not found on server.")
            except Exception as e:
                from .api import logger
                logger.error(f"Error downloading dataset '{dataset}': {e}")
                raise
        else:
            raise ValueError("Dataset must be a Dataset object, integer ID, or a string name.")

    def _process_criteria(self, criteria):
        """
        Process and validate the criteria parameter.
        Converts various criteria representations (object, dict, ID, name) to a list
        of dicts with id and name for use in API requests.
        Args:
            criteria (list, Criteria, str, int): The criteria to process.
        Returns:
            list: List of dicts with 'id' and 'name' for each criteria.
        Raises:
            ValueError: If the criteria are invalid.
            Exception: If criteria cannot be found on the server.
        """
        criteria_list = []
        if isinstance(criteria, list):
            for crit in criteria:
                if isinstance(crit, Criteria):
                    if crit.id:
                        criteria_list.append({'id': crit.id, 'name': crit.name})
                    else:
                        # Check that required fields are nonempty
                        if (crit.name and crit.name.strip() and
                            crit.phase and crit.phase.strip() and
                            crit.output_type and crit.output_type.strip() and
                            crit.criteria_type and crit.criteria_type.strip()):
                            crit.save()
                            criteria_list.append({'id': crit.id, 'name': crit.name})
                        else:
                            try:
                                downloaded = Criteria.download(name=crit.name)
                                if downloaded and downloaded.id:
                                    crit.id = downloaded.id
                                    criteria_list.append({'id': crit.id, 'name': crit.name})
                                else:
                                    raise Exception(f"Criteria with name '{crit.name}' not found on server.")
                            except Exception as e:
                                raise Exception(
                                    f"Criteria '{crit.name}' is missing required attributes and lookup failed: {str(e)}"
                                )
                elif isinstance(crit, dict):
                    # Assume the dict represents a Criteria object
                    try:
                        c_obj = Criteria.from_dict(crit)
                        if c_obj.id:
                            criteria_list.append({'id': c_obj.id, 'name': c_obj.name})
                        else:
                            c_obj.save()
                            criteria_list.append({'id': c_obj.id, 'name': c_obj.name})
                    except Exception as e:
                        raise Exception(f"Failed to process criteria dict: {str(e)}")
                elif isinstance(crit, int):
                    # If an integer is provided, assume it's an ID.
                    downloaded = Criteria.download(id=crit)
                    if downloaded and downloaded.id:
                        criteria_list.append({'id': downloaded.id, 'name': downloaded.name})
                    else:
                        raise Exception(f"Criteria not found on server with id: {crit}")
                elif isinstance(crit, str):
                    try:
                        downloaded = Criteria.download(name=crit)
                        if downloaded and downloaded.id:
                            criteria_list.append({'id': downloaded.id, 'name': downloaded.name})
                        else:
                            raise Exception(f"Criteria with name '{crit}' not found on server.")
                    except Exception as e:
                        raise Exception(f"Criteria lookup failed for '{crit}': {str(e)}")
                else:
                    raise ValueError("Each Criteria must be a Criteria object, dict, integer, or a string.")
            return criteria_list
        elif isinstance(criteria, str):
            downloaded = Criteria.download(name=criteria)
            if downloaded and downloaded.id:
                return [{'id': downloaded.id, 'name': downloaded.name}]
            else:
                raise Exception(f"Criteria not found on server with name: {criteria}")
        else:
            raise ValueError("Criteria must be provided as a list or a string.")

    def to_dict(self):
        """
        Convert the ReviewQueue to a dictionary for API communication.
        
    
    Returns:
            dict: Dictionary representation of the review queue with all necessary
                 fields for API communication.
        """
        from .api import logger
        
        try:
            # Process criteria with error handling
            processed_criteria = None
            try:
                processed_criteria = self._process_criteria(self.criteria)
            except Exception as e:
                logger.error(f"Error processing criteria: {e}")
                # Default to empty list if processing fails
                processed_criteria = []
                
            # Process dataset with error handling
            processed_dataset = None
            try:
                processed_dataset = self._process_dataset(self.dataset)
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                # Keep as None if processing fails
                
            return {
                "name": self.name,
                "condition": self.condition,
                "criteria": processed_criteria,
                "judge_model": self.judge_model,
                "dataset": processed_dataset,
                "retroactive": self.retroactive,
                "edit": self.edit_mode,
                "id": self.id if self.edit_mode else None
            }
        except Exception as e:
            logger.error(f"Error in ReviewQueue.to_dict(): {e}")
            # Provide minimal valid dict if conversion fails
            return {
                "name": self.name,
                "condition": self.condition or "",
                "criteria": [],
                "retroactive": False
            }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a ReviewQueue instance from a dictionary.
        
    
    Args:
            data (dict): Dictionary containing review queue information.
            
    
    Returns:
            ReviewQueue: A new ReviewQueue instance with the specified data.
        """
        rq = cls(
            name=data.get("name", ""),
            condition=data.get("condition", ""),
            criteria=data.get("criteria", []),
            judge_model=data.get("judge_model", None),
            dataset=data.get("dataset", None),
            retroactive=data.get("retroactive", False)
        )
        rq.id = data.get("id")
        traces_data = data.get("traces", [])
        rq.traces = [Trace.from_dict(td) for td in traces_data] if traces_data else []
        return rq

    def __iter__(self):
        """
        Make the ReviewQueue iterable over its traces.
        
    
    Returns:
            iterator: An iterator over the review queue's traces.
        """
        return iter(self.traces)
    
    @classmethod
    def _download_traces(cls, review_queue_id=None):
        """
        Retrieve a list of traces associated with a review queue.
        
        Args:
            review_queue_id (int, optional): The ID of the review queue to get traces for.
            search_term (str, optional): Additional search terms to filter traces.
            
        Returns:
            list: A list of Trace objects matching the criteria.
            
        Raises:
            RagMetricsAPIError: If the API request fails.
            RagMetricsAuthError: If authentication fails.
        """
        from .api import ragmetrics_client, logger
        from .trace import Trace

        if not ragmetrics_client.access_token:
            logger.warning("RagMetrics: Not logged in. Cannot retrieve traces. Call login() first.")
            return []

        if review_queue_id is None:
            # No filtering criteria provided
            logger.warning("No review_queue_id or search_term provided for filtering traces")
            return []            
            
        try:
            # Call the new queue_traces endpoint that wraps get_queue_traces
            endpoint = "/api/client/trace/queue_traces/"
            payload = {"review_queue_id": review_queue_id}
            
            response = ragmetrics_client._make_request(
                endpoint=endpoint,
                method="post",
                json=payload
            )
            
            if not response or "data" not in response:
                logger.warning("No traces found or unexpected response format")
                return []
            
            return [Trace.from_dict(td) for td in response["data"]]
                
        except Exception as e:
            logger.error(f"Error retrieving traces: {e}")
            return []