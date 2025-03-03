
from .api import ragmetrics_client  # This is your HTTP client wrapper for RagMetrics

class Task:
    def __init__(self, name,model, prompt=""):
        self.name = name
        self.prompt = prompt
        self.model = model  
        self.id = None

    def to_dict(self):
        """Convert the Task instance into a dictionary for API requests."""
        return {
            "taskName": self.name,
            "taskPrompt": self.prompt,
            "taskModel": self.model  
        }

    def save(self):
        """
        Saves the task to RagMetrics by calling the API endpoint.
        """
        payload = self.to_dict()
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        response = ragmetrics_client._make_request(
            method="post",
            endpoint="/api/client/task/save/",
            json=payload,
            headers=headers
        )
        if response.status_code == 200:
            json_resp = response.json()
            self.id = json_resp.get("task", {}).get("id")
        else:
            raise Exception("Failed to save task: " + response.text)

    @classmethod
    def download(cls, id):
        """
        Downloads a task from RagMetrics.
        """
        headers = {"Authorization": f"Token {ragmetrics_client.access_token}"}
        endpoint = f"/api/client/task/download/?id={id}"
        response = ragmetrics_client._make_request(
            method="get",
            endpoint=endpoint,
            headers=headers
        )
        if response.status_code == 200:
            json_resp = response.json()
            task_data = json_resp.get("task", {})
            task = cls(
                name=task_data.get("taskName", ""),
                prompt=task_data.get("taskPrompt", ""),
                model=task_data.get("taskModel", "")
            )
            task.id = task_data.get("id")
            return task
        else:
            raise Exception("Failed to download task: " + response.text)
