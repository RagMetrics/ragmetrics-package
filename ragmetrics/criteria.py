from .api import RagMetricsObject

class Criteria(RagMetricsObject):
    object_type = "criteria"

    def __init__(self, name, phase="", description="", prompt="",
                 bool_true="", bool_false="",
                 output_type="", header="",
                 likert_score_1="", likert_score_2="", likert_score_3="",
                 likert_score_4="", likert_score_5="",
                 criteria_type="llm_judge", function_name="",
                 match_type="", match_pattern="", test_string="",
                 validation_status="", case_sensitive=False):
        """
        Parameters:
          - name (str): The criteria name (required).
          - phase (str): Either "retrieval" or "generation".
          - description (str): For prompt output type.
          - prompt (str): For prompt output type.
          - bool_true (str): True description (for Boolean output type).
          - bool_false (str): False description (for Boolean output type).
          - template_type (str): Output type, e.g., "5-point", "bool", or "prompt".
          - header (str): For 5-point or Boolean output types.
          - likert_score_1..5 (str): Labels for a 5-point Likert scale.
          - implementation_type (str): "llm_judge" or "function".
          - function_name (str): If implementation_type is "function", the name of the function.
          - match_type (str): For string_match function (e.g., "starts_with", "ends_with", "contains", "regex_match").
          - match_pattern (str): The pattern used for matching.
          - test_string (str): A sample test string.
          - validation_status (str): "valid" or "invalid".
          - case_sensitive (bool): Whether matching is case sensitive.
        """
        self.name = name
        self.phase = phase
        self.description = description
        self.prompt = prompt
        self.bool_true = bool_true
        self.bool_false = bool_false
        self.output_type = output_type
        self.header = header
        self.likert_score_1 = likert_score_1
        self.likert_score_2 = likert_score_2
        self.likert_score_3 = likert_score_3
        self.likert_score_4 = likert_score_4
        self.likert_score_5 = likert_score_5
        self.criteria_type = criteria_type
        self.function_name = function_name
        self.match_type = match_type
        self.match_pattern = match_pattern
        self.test_string = test_string
        self.validation_status = validation_status
        self.case_sensitive = case_sensitive
        self.id = None

    def to_dict(self):
        data = {
            "criteria_name": self.name,
            "type": self.phase,
            "implementation_type": self.criteria_type,
            "template_type": self.output_type,
        }
        # For LLM as Judge, output depends on template_type.
        if self.output_type == "5-point":
            data["header"] = self.header
            data["likert_score_1"] = self.likert_score_1
            data["likert_score_2"] = self.likert_score_2
            data["likert_score_3"] = self.likert_score_3
            data["likert_score_4"] = self.likert_score_4
            data["likert_score_5"] = self.likert_score_5
        elif self.output_type == "bool":
            data["header"] = self.header
            data["bool_true"] = self.bool_true
            data["bool_false"] = self.bool_false
        elif self.output_type == "prompt":
            data["description"] = self.description
            data["prompt"] = self.prompt

        # For function-based criteria, include function details.
        if self.criteria_type == "function":
            data["function_name"] = self.function_name
            if self.function_name == "string_match":
                data["match_type"] = self.match_type
                data["match_pattern"] = self.match_pattern
                data["test_string"] = self.test_string
                data["case_sensitive"] = self.case_sensitive

        return data

    @classmethod
    def from_dict(cls, data: dict):
        crit = cls(
            name=data.get("name", ""),
            phase=data.get("type", ""),
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            bool_true=data.get("bool_true", ""),
            bool_false=data.get("bool_false", ""),
            output_type=data.get("template_type", ""),
            header=data.get("header", ""),
            likert_score_1=data.get("likert_score_1", ""),
            likert_score_2=data.get("likert_score_2", ""),
            likert_score_3=data.get("likert_score_3", ""),
            likert_score_4=data.get("likert_score_4", ""),
            likert_score_5=data.get("likert_score_5", ""),
            criteria_type=data.get("implementation_type", "llm_judge"),
            function_name=data.get("function_name", ""),
            match_type=data.get("match_type", ""),
            match_pattern=data.get("match_pattern", ""),
            test_string=data.get("test_string", ""),
            validation_status=data.get("validation_status", ""),
            case_sensitive=data.get("case_sensitive", False)
        )
        crit.id = data.get("id")
        return crit

