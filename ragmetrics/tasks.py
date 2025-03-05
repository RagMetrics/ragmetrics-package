from .api import RagMetricsObject  # This is your HTTP client wrapper for RagMetrics

class Task(RagMetricsObject):
    object_type = "task" 

    def __init__(self, name, model, prompt=""):
        self.name = name
        self.model = model
        self.prompt = prompt
        self.id = None

    def to_dict(self):
        return {
            "taskName": self.name,
            "taskPrompt": self.prompt,
            "taskModel": self.model
        }

    @classmethod
    def from_dict(cls, data: dict):
        task = cls(
            name=data.get("taskName", ""),
            prompt=data.get("taskPrompt", ""),
            model=data.get("taskModel", "")
        )
        task.id = data.get("id")
        return task