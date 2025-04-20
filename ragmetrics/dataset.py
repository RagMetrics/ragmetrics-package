from .api import RagMetricsObject

class Example:
    """
    A single example in a dataset for evaluation.
    
    Each Example represents one test case consisting of a question, the ground truth
    context that contains the answer, and the expected ground truth answer.
    
    Examples are used in experiments to evaluate how well a model or RAG pipeline
    performs on specific questions.
    """

    def __init__(self, question, ground_truth_context=None, ground_truth_answer=None):
        """
        Initialize a new Example instance.
        
        Example:
        
            .. code-block:: python
            
                # Simple example with string context
                example = Example(
                    question="What is the capital of France?",
                    ground_truth_context="France is a country in Western Europe. Its capital is Paris.",
                    ground_truth_answer="Paris"
                )
                
                # Example with a list of context strings
                example_multi_context = Example(
                    question="Is NYC beautiful?",
                    ground_truth_context=[
                        "NYC is the biggest city in the east of US.",
                        "NYC is on the eastern seaboard.",
                        "NYC is a very beautiful city"
                    ],
                    ground_truth_answer="Yes"
                )
                
                # Example with only question
                example_question_only = Example(
                    question="What is the population of London?"
                )

    
    Args:
            question (str): The question to be answered.
            ground_truth_context (str or list, optional): The context containing the answer. Can be a string or list of strings.
            ground_truth_answer (str, optional): The expected answer to the question.
        """
        self.question = question
        self.ground_truth_context = ground_truth_context
        self.ground_truth_answer = ground_truth_answer

    def to_dict(self):
        """
        Convert the Example instance into a dictionary for API requests.

    
    Returns:
            dict: Dictionary containing the example's question, context, and answer.
        """
        result = {"question": self.question}
        
        if self.ground_truth_context is not None:
            result["ground_truth_context"] = self.ground_truth_context
            
        if self.ground_truth_answer is not None:
            result["ground_truth_answer"] = self.ground_truth_answer
            
        return result

class Dataset(RagMetricsObject):
    """
    A collection of examples for evaluation.
    
    Datasets are used in experiments to test models and RAG pipelines against a
    consistent set of questions. They provide the questions and ground truth
    information needed for systematic evaluation.
    
    Datasets can be created programmatically, uploaded from files, or downloaded
    from the RagMetrics platform.
    """

    object_type = "dataset"

    def __init__(self, name, examples=[], source_type="", source_file="", questions_qty=0):
        """
        Initialize a new Dataset instance.
        
        Example - Creating and saving a dataset:
        
            .. code-block:: python
            
                import ragmetrics
                from ragmetrics import Example, Dataset
                
                # Login to RagMetrics
                ragmetrics.login("your-api-key")
                
                # Create examples
                examples = [
                    Example(
                        question="What is the capital of France?",
                        ground_truth_context="France is a country in Western Europe. Its capital is Paris.",
                        ground_truth_answer="Paris"
                    ),
                    Example(
                        question="Who wrote Hamlet?",
                        ground_truth_context="Hamlet is a tragedy written by William Shakespeare.",
                        ground_truth_answer="William Shakespeare"
                    )
                ]
                
                # Create dataset
                dataset = Dataset(name="Geography and Literature QA", examples=examples)
                
                # Save to RagMetrics platform
                dataset.save()
                print(f"Dataset saved with ID: {dataset.id}")
            
        Example - Downloading and using an existing dataset:
        
            .. code-block:: python
            
                # Download dataset by name
                dataset = Dataset.download(name="Geography and Literature QA")
                
                # Or download by ID
                # dataset = Dataset.download(id=12345)
                
                # Iterate through examples
                for example in dataset:
                    print(f"Question: {example.question}")
                    print(f"Answer: {example.ground_truth_answer}")
                    
                # Access example count
                print(f"Dataset contains {len(dataset.examples)} examples")

    
    Args:
            name (str): The name of the dataset.
            examples (list): List of Example instances (default: []).
            source_type (str): Type of the data source (default: "").
            source_file (str): Path to the source file (default: "").
            questions_qty (int): Number of questions in the dataset (default: 0).
        """
        self.name = name
        self.examples = examples 
        self.source_type = source_type
        self.source_file = source_file
        self.questions_qty = questions_qty
        self.id = None

    def to_dict(self):
        """
        Convert the Dataset instance into a dictionary for API communication.

    
    Returns:
            dict: Dictionary containing the dataset name, source, examples, and quantity.
        """
        return {
            "datasetName": self.name,
            "datasetSource": "DA",
            "examples": [ex.to_dict() for ex in self.examples],
            "datasetQty": len(self.examples)
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Dataset instance from a dictionary.
        
        Used internally when downloading datasets from the RagMetrics API.

    
    Args:
            data (dict): Dictionary containing dataset information.

    
    Returns:
            Dataset: A new Dataset instance with the specified data.
        """
        examples = [
            Example(**{k: v for k, v in ex.items() if k in ['question', 'ground_truth_context', 'ground_truth_answer']})
            for ex in data.get("examples", [])
        ]
        ds = cls(
            name=data.get("name", ""),
            examples=examples,
            source_type=data.get("source_type", ""),
            source_file=data.get("source_file", ""),
            questions_qty=data.get("questions_qty", 0)
        )
        ds.id = data.get("id")
        return ds
    
    def __iter__(self):
        """
        Make the Dataset instance iterable over its examples.
        
        This allows using a dataset in a for loop to iterate through examples.
        
        Example:
        
            .. code-block:: python
            
                dataset = Dataset.download(name="my-dataset")
                for example in dataset:
                    print(example.question)

    
    Returns:
            iterator: An iterator over the dataset's examples.
        """
        return iter(self.examples)