from .api import ragmetrics_client

class Example:
    def __init__(self, question, ground_truth_context, ground_truth_answer):
        self.question = question
        self.ground_truth_context = ground_truth_context
        self.ground_truth_answer = ground_truth_answer

    def to_dict(self):
        """Convert the Example instance into a dictionary for API requests."""
        return {
            "question": self.question,
            "ground_truth_context": self.ground_truth_context,
            "ground_truth_answer": self.ground_truth_answer
        }

class Dataset:
    def __init__(self, examples, name):
        """
        :param examples: A list of Example instances.
        :param name: Name of the dataset.
        """
        self.examples = examples
        self.name = name
        self.id = None  

    def save(self):
        """
        Saves the dataset to RagMetrics by calling the API endpoint.
        """
        payload = {
            "datasetName": self.name,
            "datasetSource":"DM" ,
            "examples": [ex.to_dict() for ex in self.examples],
            "datasetQty": len(self.examples) 
        }
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        response = ragmetrics_client._make_request(
            method="post",
            endpoint="/api/client/dataset/save/",
            json=payload,
            headers=headers
        )
        if response.status_code == 200:
            json_resp = response.json()
            self.id = json_resp.get("dataset", {}).get("id")
        else:
            raise Exception("Failed to save dataset: " + response.text)

    @classmethod
    def download(cls, name):
        """
        Downloads a dataset from RagMetrics.
        """
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}

        # Determine if the identifier is numeric.
        try:
            # If this conversion succeeds, treat identifier as an ID.
            int(name)
            endpoint = f"/api/client/dataset/download/?id={name}"
        except (ValueError, TypeError):
            # Otherwise, treat it as a name.
            endpoint = f"/api/client/dataset/download/?name={name}"

        response = ragmetrics_client._make_request(
            method="GET",
            endpoint=endpoint,
            headers=headers
        )

        if response.status_code == 200:
            json_resp = response.json()
            ds_data = json_resp.get("dataset", {})
            examples = [
                Example(**{k: v for k, v in ex.items() if k in ['question', 'ground_truth_context', 'ground_truth_answer']})
                for ex in ds_data.get("examples", [])
            ]
            ds = cls(examples, ds_data.get("name", ""))
            ds.id = ds_data.get("id")
            return ds
        else:
            raise Exception("Failed to download dataset: " + response.text)

    def __iter__(self):
        """Allow iteration over the examples in the dataset."""
        return iter(self.examples)