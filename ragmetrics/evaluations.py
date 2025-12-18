import requests
import logging
from ragmetrics.api import ragmetrics_client

logger = logging.getLogger(__name__)

class Evaluation:
    """
    A class representing a single evaluation.

    Allows running individual evaluations against the RagMetrics API.
    """

    def __init__(self, eval_group_id):
        """
        Initialize a new Evaluation instance.

        Args:
            eval_group_id (str): The ID of the evaluation group to associate with.
        """
        self.eval_group_id = eval_group_id

    def compute(self, question, answer, ground_truth=None, conversation_id=None, model=None):
        """
        Run a single evaluation.

        Args:
            question (str): The question being evaluated.
            answer (str): The answer to evaluate.
            ground_truth (str, optional): The ground truth answer. Required for criteria-based evaluation.
            conversation_id (str, optional): ID to group related evaluations.
            model (str, optional): The model ID to use for judging (if different from group default).

        Returns:
            dict: The evaluation results from the API.
        """
        payload = {
            "eval_group_id": self.eval_group_id,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "conversation_id": conversation_id,
            "type": "C" if ground_truth else "S",  # 'C' for criteria-based (needs GT), 'S' for single/simple
            "model": model
        }

        # Filter out None values to let API defaults handle them
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._call_api(
            endpoint="/api/v2/single-evaluation/",
            data=payload
        )

    def _call_api(self, endpoint, data):
        """
        Helper method to make API calls using the ragmetrics_client configuration.
        """
        if not ragmetrics_client.access_token:
             # Try to login if token is missing, assuming env var might be set or login called previously
            try:
                if not ragmetrics_client.login(key=None): # Will try to load from ENV
                     raise ValueError("Not logged in. Please call ragmetrics.login() first.")
            except Exception as e:
                 raise ValueError(f"Not logged in and failed to auto-login: {str(e)}")

        url = f"{ragmetrics_client.base_url}{endpoint}"
        headers = {
            "Authorization": f"Token {ragmetrics_client.access_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
             # Try to extract detailed error message from response
            try:
                error_detail = e.response.json()
                raise Exception(f"API Error: {error_detail.get('message', str(e))}")
            except:
                raise e
        except Exception as e:
            logger.error(f"Error calling RagMetrics API: {str(e)}")
            raise e
