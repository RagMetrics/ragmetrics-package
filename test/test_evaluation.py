import unittest
from unittest.mock import patch, MagicMock
from ragmetrics import Evaluation
from ragmetrics.api import ragmetrics_client

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        # Mock the login to avoid actual authentication check failure if not logged in
        ragmetrics_client.access_token = "mock_token"
        ragmetrics_client.base_url = "https://mock-api.ragmetrics.ai"

    @patch('ragmetrics.evaluations.requests.post')
    def test_compute_criteria_evaluation(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "score": 0.8,
            "reason": "Good answer"
        }
        mock_post.return_value = mock_response

        # Initialize Evaluation
        evaluator = Evaluation(eval_group_id="test_group_123")
        
        # Run compute
        result = evaluator.compute(
            question="What is 2+2?",
            answer="4",
            ground_truth="4"
        )

        # Verify request
        self.assertEqual(result['score'], 0.8)
        
        # Check if post was called with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        self.assertEqual(args[0], "https://mock-api.ragmetrics.ai/api/v2/single-evaluation/")
        self.assertEqual(kwargs['headers']['Authorization'], "Token mock_token")
        
        expected_payload = {
            "eval_group_id": "test_group_123",
            "question": "What is 2+2?",
            "answer": "4",
            "ground_truth": "4",
            "type": "C"
        }
        self.assertEqual(kwargs['json'], expected_payload)

    @patch('ragmetrics.evaluations.requests.post')
    def test_compute_single_evaluation_no_gt(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response

        evaluator = Evaluation(eval_group_id="test_group_123")
        
        result = evaluator.compute(
            question="What is the meaning of life?",
            answer="42"
        )
        
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['type'], "S")
        self.assertNotIn("ground_truth", kwargs['json'])

if __name__ == '__main__':
    unittest.main()
