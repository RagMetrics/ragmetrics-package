from .api import RagMetricsObject
from .dataset import Dataset 
from .criteria import Criteria
from .trace import Trace

class Review(RagMetricsObject):
    object_type = "reviews"

    def __init__(self, name, condition="", criteria=None, judge_model=None, dataset=None, edit_mode=False, edit_id=None):
        self.name = name
        self.condition = condition
        self.criteria = criteria
        self.judge_model = judge_model
        self.dataset = dataset
        self.edit_mode = edit_mode  # If True, we are editing an existing review
        self.edit_id = edit_id if edit_mode else None  # ID is required when editing
        self.traces = []
    
    def edit(self, id):
        """Marks the review for editing and assigns an existing ID."""
        self.edit_mode = True
        self.edit_id = id
        return self

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

    def to_dict(self):
        """Builds the payload to send to the server."""
        return {
            "name": self.name,
            "condition": self.condition,
            "criteria": self._process_criteria(self.criteria),
            "judge_model": self.judge_model,
            "dataset": self._process_dataset(self.dataset),
            "edit": self.edit_mode,
            "edit_id": self.edit_id if self.edit_mode else None
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