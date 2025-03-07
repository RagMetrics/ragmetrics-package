from .api import RagMetricsObject  # This is your HTTP client wrapper for RagMetrics

class Task(RagMetricsObject):
    object_type = "task" 

    def __init__(self, name, generator_model="", system_prompt=""):
        self.name = name
        self.generator_model = generator_model
        self.system_prompt = system_prompt
        self.id = None

    def to_dict(self):
        return {
            "taskName": self.name,
            "taskPrompt": self.system_prompt,
            "taskModel": self.generator_model
        }

    @classmethod
    def from_dict(cls, data: dict):
        task = cls(
            name=data.get("taskName", ""),
            system_prompt=data.get("taskPrompt", ""),
            generator_model=data.get("taskModel", "")
        )
        task.id = data.get("id")
        return task
