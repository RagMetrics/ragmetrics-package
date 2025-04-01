from .api import RagMetricsObject

class Criteria(RagMetricsObject):
    """
    Defines evaluation criteria for assessing LLM responses.
    
    Criteria specify how to evaluate LLM responses in experiments and reviews.
    They can operate in two different modes:
    
    1. LLM-based evaluation: Uses another LLM to judge responses based on
       specified rubrics like Likert scales, boolean judgments, or custom prompts.
       
    2. Function-based evaluation: Uses programmatic rules like string matching
       to automatically evaluate responses.
       
    Criteria can be applied to either the retrieval phase (evaluating context)
    or the generation phase (evaluating final answers).
    """

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
        Initialize a new Criteria instance.
        
        Example - Creating a 5-point Likert scale criteria:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Criteria
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Create a relevance criteria using a 5-point Likert scale
                relevance = Criteria(
                    name="Relevance",
                    phase="generation",
                    output_type="5-point",
                    criteria_type="llm_judge",
                    header="How relevant is the response to the question?",
                    likert_score_1="Not relevant at all",
                    likert_score_2="Slightly relevant",
                    likert_score_3="Moderately relevant",
                    likert_score_4="Very relevant",
                    likert_score_5="Completely relevant"
                )
                relevance.save()
            
        Example - Creating a boolean criteria:
        
            .. code-block:: python
            
                # Create a factual correctness criteria using a boolean judgment
                factual = Criteria(
                    name="Factually Correct",
                    phase="generation",
                    output_type="bool",
                    criteria_type="llm_judge",
                    header="Is the response factually correct based on the provided context?",
                    bool_true="Yes, the response is factually correct and consistent with the context",
                    bool_false="No, the response contains factual errors or contradicts the context"
                )
                factual.save()
            
        Example - Creating a string matching criteria (automated):
        
            .. code-block:: python
            
                # Create an automated criteria that checks if a response contains a date
                contains_date = Criteria(
                    name="Contains Date",
                    phase="generation",
                    output_type="bool",
                    criteria_type="function",
                    function_name="string_match",
                    match_type="regex_match",
                    match_pattern=r"\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}",
                    test_string="The event occurred on 12/25/2023",
                    case_sensitive=False
                )
                contains_date.save()
            
        Example - Creating a custom prompt criteria:
        
            .. code-block:: python
            
                # Create a criteria with a custom prompt for more flexible evaluation
                custom_eval = Criteria(
                    name="Reasoning Quality",
                    phase="generation",
                    output_type="prompt",
                    criteria_type="llm_judge",
                    description="Evaluate the quality of reasoning in the response",
                    prompt=(
                        "On a scale of 1-10, rate the quality of reasoning in the response.\n"
                        "Consider these factors:\n"
                        "- Logical flow of arguments\n"
                        "- Use of evidence\n"
                        "- Consideration of alternatives\n"
                        "- Absence of fallacies\n\n"
                        "First explain your reasoning, then provide a final score between 1-10."
                    )
                )
                custom_eval.save()

    
    Args:
            name (str): The criteria name (required).
            phase (str): Either "retrieval" or "generation" (default: "").
            description (str): Description for prompt output type (default: "").
            prompt (str): Prompt for prompt output type (default: "").
            bool_true (str): True description for Boolean output type (default: "").
            bool_false (str): False description for Boolean output type (default: "").
            output_type (str): Output type, e.g., "5-point", "bool", or "prompt" (default: "").
            header (str): Header for 5-point or Boolean output types (default: "").
            likert_score_1..5 (str): Labels for a 5-point Likert scale (default: "").
            criteria_type (str): Implementation type, "llm_judge" or "function" (default: "llm_judge").
            function_name (str): Name of the function if criteria_type is "function" (default: "").
            match_type (str): For string_match function (e.g., "starts_with", "ends_with", "contains", "regex_match") (default: "").
            match_pattern (str): The pattern used for matching (default: "").
            test_string (str): A sample test string (default: "").
            validation_status (str): "valid" or "invalid" (default: "").
            case_sensitive (bool): Whether matching is case sensitive (default: False).
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
        
    @classmethod
    def download(cls, name=None, criteria_id=None):
        """
        Download a Criteria from the RagMetrics server.
        
        You must provide either a name or a criteria_id to identify the criteria to download.
        
        Example - Download by name:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Criteria
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Download a common evaluation criteria by name
                accuracy = Criteria.download(name="Accuracy")
                
                # Use the downloaded criteria in an experiment
                experiment = Experiment(
                    name="Evaluation",
                    dataset=dataset,
                    task=task,
                    cohorts=cohorts,
                    criteria=[accuracy, "Relevance"],  # Mix of objects and names
                    judge_model="gpt-4"
                )
            
        Example - Download multiple criteria:
        
            .. code-block:: python
            
                # Download multiple criteria that you'll use frequently
                accuracy = Criteria.download(name="Accuracy")
                relevance = Criteria.download(name="Relevance")
                helpfulness = Criteria.download(name="Helpfulness")
                
                # Use them in experiments
                experiment = Experiment(
                    name="Comprehensive Evaluation",
                    dataset=dataset,
                    task=task,
                    cohorts=cohorts,
                    criteria=[accuracy, relevance, helpfulness],
                    judge_model="gpt-4"
                )

    
    Args:
            name (str, optional): The name of the criteria to download.
            criteria_id (str, optional): The ID of the criteria to download.

    
    Returns:
            Criteria: The downloaded Criteria instance.

    
    Raises:
            ValueError: If neither name nor criteria_id is provided, or if the criteria
                is not found on the server.
        """
        from .api import api_client
        
        if not name and not criteria_id:
            raise ValueError("Either name or criteria_id must be provided")
        
        if criteria_id:
            response = api_client.get(f"/criteria/{criteria_id}")
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve criteria: {response.text}")
            return cls.from_dict(response.json())
        
        # Search by name
        response = api_client.get("/criteria")
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve criteria: {response.text}")
        
        all_criteria = response.json()
        for criteria in all_criteria:
            if criteria.get("name") == name:
                return cls.from_dict(criteria)
        
        raise ValueError(f"Criteria with name '{name}' not found")

    def save(self):
        """
        Save the Criteria to the RagMetrics server.
        
        If a criteria with the same name already exists, it will be updated.
        
        Example - Creating and saving a criteria:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Criteria
                
                # Login
                ragmetrics.login("your-api-key")
                
                # Create a criteria for measuring conciseness
                conciseness = Criteria(
                    name="Conciseness",
                    phase="generation",
                    output_type="5-point",
                    criteria_type="llm_judge",
                    header="How concise is the response?",
                    likert_score_1="Extremely verbose with lots of unnecessary information",
                    likert_score_2="Somewhat verbose with some unnecessary information",
                    likert_score_3="Moderately concise with balanced detail",
                    likert_score_4="Very concise with good information density",
                    likert_score_5="Perfectly concise with no wasted words"
                )
                
                # Save the criteria to use in experiments
                criteria_id = conciseness.save()
                print(f"Saved criteria with ID: {criteria_id}")

    
    Returns:
            str: The ID of the saved criteria.

    
    Raises:
            Exception: If saving fails due to network issues or invalid data.
        """
        from .api import api_client
        
        payload = self.to_dict()
        
        if self.id:
            # Update existing criteria
            response = api_client.put(f"/criteria/{self.id}", json=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to update criteria: {response.text}")
            self.id = response.json().get("id")
        else:
            # Create new criteria
            response = api_client.post("/criteria", json=payload)
            if response.status_code != 201:
                raise Exception(f"Failed to create criteria: {response.text}")
            self.id = response.json().get("id")
            
        return self.id

    def to_dict(self):
        """
        Convert the criteria object to a dictionary format for API communication.
        
        The specific fields included in the dictionary depend on the criteria's
        output_type and criteria_type.

    
    Returns:
            dict: Dictionary representation of the criteria, including all relevant
                 fields based on the output_type and criteria_type.
        """
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
        """
        Create a Criteria instance from a dictionary.
        
        Used internally when downloading criteria from the RagMetrics API.

    
    Args:
            data (dict): Dictionary containing criteria data.

    
    Returns:
            Criteria: A new Criteria instance with the specified data.
        """
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

