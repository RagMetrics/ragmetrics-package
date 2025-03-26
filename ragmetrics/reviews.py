from .api import RagMetricsObject
from .dataset import Dataset 
from .criteria import Criteria
from .trace import Trace

class ReviewQueue(RagMetricsObject):
    object_type = "reviews"

    def __init__(self, name, condition="", criteria=None, judge_model=None, dataset=None):
        self.name = name
        self.condition = condition
        self.criteria = criteria
        self.judge_model = judge_model
        self.dataset = dataset
        self.id = None  
        self.traces = []
        self.edit_mode = False  

    def __setattr__(self, key, value):
        """
        Automatically enable edit mode if modifying an attribute after id is set.
        """
        if key not in {"edit_mode"} and hasattr(self, "id") and self.id:
            object.__setattr__(self, "edit_mode", True)
        object.__setattr__(self, key, value)

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
        elif isinstance(dataset, int):
            downloaded = Dataset.download(id=dataset)
            if downloaded and getattr(downloaded, "id", None):
                return downloaded.id
            else:
                raise Exception(f"Dataset not found on server with id: {dataset}")
        elif isinstance(dataset, str):
            downloaded = Dataset.download(name=dataset)
            if downloaded and getattr(downloaded, "id", None):
                return downloaded.id
            else:
                raise Exception(f"Dataset not found on server with name: {dataset}")
        else:
            raise ValueError("Dataset must be a Dataset object or a string.")

    def _process_criteria(self, criteria):
        """
        Processes the criteria parameter.
        Accepts a list of Criteria objects, dictionaries, integers, or strings.
        Returns a list of Criteria IDs.
        """
        criteria_ids = []
        if isinstance(criteria, list):
            for crit in criteria:
                if isinstance(crit, Criteria):
                    if crit.id:
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
                            try:
                                downloaded = Criteria.download(name=crit.name)
                                if downloaded and downloaded.id:
                                    crit.id = downloaded.id
                                    criteria_ids.append(crit.id)
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
                            criteria_ids.append(c_obj.id)
                        else:
                            c_obj.save()
                            criteria_ids.append(c_obj.id)
                    except Exception as e:
                        raise Exception(f"Failed to process criteria dict: {str(e)}")
                elif isinstance(crit, int):
                    # If an integer is provided, assume it's an ID.
                    criteria_ids.append(crit)
                elif isinstance(crit, str):
                    try:
                        downloaded = Criteria.download(name=crit)
                        if downloaded and downloaded.id:
                            criteria_ids.append(downloaded.id)
                        else:
                            raise Exception(f"Criteria with name '{crit}' not found on server.")
                    except Exception as e:
                        raise Exception(f"Criteria lookup failed for '{crit}': {str(e)}")
                else:
                    raise ValueError("Each Criteria must be a Criteria object, dict, integer, or a string.")
            return criteria_ids
        elif isinstance(criteria, str):
            downloaded = Criteria.download(name=criteria)
            if downloaded and downloaded.id:
                return [downloaded.id]
            else:
                raise Exception(f"Criteria not found on server with name: {criteria}")
        else:
            raise ValueError("Criteria must be provided as a list or a string.")


    def to_dict(self):
        """Builds the payload to send to the server."""
        return {
            "name": self.name,
            "condition": self.condition,
            "criteria": self._process_criteria(self.criteria),
            "judge_model": self.judge_model,
            "dataset": self._process_dataset(self.dataset),
            "edit": self.edit_mode,
            "id": self.id if self.edit_mode else None
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Creates an instance from a dictionary."""
        rq = cls(
            name=data.get("name", ""),
            condition=data.get("condition", ""),
            criteria=data.get("criteria", []),
            judge_model=data.get("judge_model", None),
            dataset=data.get("dataset", None)
        )
        rq.id = data.get("id")
        traces_data = data.get("traces", [])
        rq.traces = [Trace.from_dict(td) for td in traces_data] if traces_data else []
        return rq

    def __iter__(self):
        return iter(self.traces)