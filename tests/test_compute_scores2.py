import unittest
from unittest.mock import patch, Mock

class TestComputeScores2(unittest.TestCase):
    def mock_spacy_pipeline(self):
        mock_nlp = Mock()

        # Define a mock component
        class MockComponent:
            def _init_(self):
                pass

        # Add the mock component to the mock nlp object
        mock_nlp.add_pipe.return_value = MockComponent()

        return mock_nlp

    @patch('spacy.load', side_effect=mock_spacy_pipeline)  # Mock the spacy.load function
    def test_apply_ranking(self, mock_load):
        from compute_scores2 import apply_ranking # noqa: F401
        print('test')
